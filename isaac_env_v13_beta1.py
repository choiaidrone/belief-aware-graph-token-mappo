"""
isaac_env_v13.py  —  Pure NumPy RL Environment  (Isaac Sim 완전 제거)
======================================================================
v12 → v13 변경사항:

  [Isaac Sim / omni 완전 제거]
  - SimulationApp, World, VisualSphere, VisualCuboid, UsdGeom 등 제거
  - self.world, drone_prims, set_world_pose() 제거
  - 타겟/드론 위치: numpy array 직접 관리 (기존과 동일 로직 유지)
  - Depth: heightmap ray-cast만 사용 (학습 모드에서 이미 이렇게 동작)
  - 지형: heightmap numpy array 그대로 유지 (USD 시각화 제거)

  [버그 수정]
  - AntiAir 하드코딩 2개 → num_anti_air개로 수정
    기존: [(45,57),(12,22)], [(45,57),(-22,-12)] 2개 고정
    수정: num_anti_air개를 y 대칭 배치로 동적 생성
    → scale 실험 (8x40, 12x60) 시 target_types 길이 불일치 해결

  [인터페이스 완전 동일]
  - obs shape, reward, done, info 구조 v12와 100% 동일
  - 기존 학습된 체크포인트 (MLP-MAPPO, Token-MAPPO) 그대로 사용 가능

  [사용법]
  - env = DroneSwarmEnv(randomize_terrain=False, terrain_seed=42)
  - obs = env.reset()
  - obs, rewards, dones, info = env.step(actions)
  ※ world.reset() / world.step() 호출 불필요

obs 구조 (484차원, per drone):
  [  0:300] DS 맵  (E/F/U 10×10 mean-pooled)
  [300:308] Depth  (8방향 장애물 거리, heightmap ray-cast)
  [308:310] 자기 위치 (x/y, 0~1 정규화)
  [310:410] Fused Priority Map 10×10 (100dim)
  [410:482] Active Track K_ACTIVE×6 (72dim)
  [482:484] 배터리 [battery_norm, dist_to_base_norm]
"""

import numpy as np
import random


# ============================================================
# 전역 상수  (v12와 완전히 동일)
# ============================================================

DS_N_INTERNAL = 100
DS_N_RL       = 10
DS_POOL_SIZE  = DS_N_INTERNAL // DS_N_RL    # 10

DS_MAP_X_MIN  = -20.0
DS_MAP_X_MAX  =  80.0
DS_MAP_Y_MIN  = -50.0
DS_MAP_Y_MAX  =  50.0
DS_CELL_W     = (DS_MAP_X_MAX - DS_MAP_X_MIN) / DS_N_INTERNAL
DS_CELL_H     = (DS_MAP_Y_MAX - DS_MAP_Y_MIN) / DS_N_INTERNAL

DS_DECAY_RATE    = 0.0005
DS_CAMO_Q_FACTOR = 0.15

Q_MAX  = 0.8
Q_BETA = 0.205
OBS_RADIUS = 2

NUM_DRONES     = 4
NUM_TARGETS    = 20
K_ACTIVE       = 12
DRONE_ALTITUDE = 12.0
DRONE_STEP_M   = 3.0
MAX_STEPS      = 3000

DRONE_SAFE_DIST = 3.0

BATTERY_INIT         = 1.0
BATTERY_COST_MOVE    = 0.0004
BATTERY_COST_TRACK   = 0.0002
BATTERY_SAFETY_FACTOR = 1.3
BASE_ARRIVAL_DIST    = 5.0

ANTIAIR_KILL_RADIUS  = 10.0
ANTIAIR_P_MAX        = 0.02
ANTIAIR_KILL_PENALTY = 0.0

DEPTH_MAX_RANGE = 30.0
DEPTH_DIRS      = [0, 45, 90, 135, 180, 225, 270, 315]

REWARD_UNCERTAINTY_REDUCE = 1.0
PENALTY_TERRAIN           = 0.0
PENALTY_DRONE_COLLISION   = -1.0

LAMBDA_TRACK     = 0.5
LAMBDA_STALE     = 0.02
LAMBDA_REACQUIRE = 0.5
T_STALE          = 50

R_TRACK_M      = 12.0
DWELL_FULL     = 3
DWELL_DECAY    = 0.7

EKF_Q_POS   = 0.5
EKF_Q_VEL   = 1.0
EKF_Q_POS_TANK = 2.0
EKF_Q_VEL_TANK = 4.0
EKF_R_DWELL = 1.0
EKF_R_INIT  = 4.0
EKF_R_CAMO  = 16.0

IMM_TANK_TRANS  = np.array([[0.92, 0.08], [0.08, 0.92]], dtype=np.float64)
IMM_TANK_OMEGA  = 0.15
IMM_ARTIL_TRANS = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64)
IMM_INF_TRANS   = np.array([[0.90, 0.10], [0.10, 0.90]], dtype=np.float64)

EKF_Q_POS_SLOW   = 0.1
EKF_Q_VEL_SLOW   = 0.5
EKF_Q_POS_STATIC = 0.02
EKF_Q_VEL_STATIC = 0.001
EKF_Q_POS_TANK_IMM = 0.5
EKF_Q_VEL_TANK_IMM = 1.0

ARTILLERY_RELO_INTERVAL = 400
ARTILLERY_RELO_JITTER   = 200
ARTILLERY_RELO_SPEED    = 1.5
ARTILLERY_RELO_PROB     = 0.7
TANK_X_RANGE    = (20, 80)
TANK_Y_RANGE    = (-30, 30)

P_INIT_POS   = 100.0
P_INIT_VEL   = 10.0
TR_P_MAX            = 2 * P_INIT_POS + 2 * P_INIT_VEL   # 220
TR_P_TRACKED_THRESH = 0.6

TARGET_SPAWN_PRIOR = {
    0: (12.5, 0.0),   # INFANTRY
    1: (51.0, 0.0),   # ARTILLERY
    2: (31.0, 0.0),   # TANK
    3: (51.0, 0.0),   # ANTI_AIR
}

OBS_DS_DIM        = 300
OBS_DEPTH_DIM     = 8
OBS_SELF_DIM      = 2
OBS_INTENSITY_DIM = 100
OBS_ACTIVE_DIM    = K_ACTIVE * 6
OBS_BATTERY_DIM   = 2
OBS_TOTAL_DIM     = (OBS_DS_DIM + OBS_DEPTH_DIM + OBS_SELF_DIM
                     + OBS_INTENSITY_DIM + OBS_ACTIVE_DIM + OBS_BATTERY_DIM)
# 300+8+2+100+72+2 = 484dim

OBS_BELIEF_DIM    = K_ACTIVE * 6
OBS_BELIEF_START  = OBS_DS_DIM + OBS_DEPTH_DIM + OBS_SELF_DIM + OBS_INTENSITY_DIM

FUSED_BETA = 1.0   # [beta sensitivity] target intensity 강하게 반영


# ── 타겟 상태 (4단계 enum) ──────────────────────────────────
class TargetState:
    UNCOMMITTED = 0
    CANDIDATE   = 1
    COMMITTED   = 2
    STALE       = 3

TR_P_CANDIDATE_THRESH  = 0.9
TR_P_COMMITTED_THRESH  = 0.6
TR_P_RECOVER_THRESH    = 0.7
DWELL_FOR_COMMIT       = 2

PRIORITY_BASE = {
    TargetState.STALE:       2.0,
    TargetState.COMMITTED:   1.0,
    TargetState.CANDIDATE:   0.5,
    TargetState.UNCOMMITTED: 0.0,
}

INTENSITY_W = {
    TargetState.COMMITTED:   1.0,
    TargetState.STALE:       1.5,
    TargetState.CANDIDATE:   0.5,
    TargetState.UNCOMMITTED: 0.0,
}
INTENSITY_SIGMA_STALE_MULT = 3.0


# ── 타겟 종류 ────────────────────────────────────────────────
class TargetType:
    INFANTRY  = 0
    ARTILLERY = 1
    TANK      = 2
    ANTI_AIR  = 3

TARGET_PARAMS = {
    TargetType.INFANTRY: {
        "camouflage_prob"  : 0.08,
        "reveal_prob"      : 0.05,
        "camouflage_check" : 30,
    },
    TargetType.ARTILLERY: {
        "camouflage_prob"  : 0.04,
        "reveal_prob"      : 0.03,
        "camouflage_check" : 30,
    },
    TargetType.TANK: {
        "camouflage_prob"  : 0.0,
        "reveal_prob"      : 1.0,
        "camouflage_check" : 9999999,
    },
    TargetType.ANTI_AIR: {
        "camouflage_prob"  : 0.02,
        "reveal_prob"      : 0.02,
        "camouflage_check" : 30,
    },
}

TANK_OU_THETA      = 0.15
TANK_OU_SIGMA      = 0.8
TANK_SPEED_MAX     = 3.0
TANK_SOFT_BOUNDARY = 8.0

INF_SPEED_MIN  = 0.3
INF_SPEED_MAX  = 0.8
INF_P_STOP     = 0.05
INF_P_MOVE     = 0.08
INF_TURN_SIGMA = 0.3


# ============================================================
# 관측 품질 함수
# ============================================================

def compute_q(d: float) -> float:
    return float(Q_MAX * np.exp(-Q_BETA * d))


# ============================================================
# DS Belief Map  (v12와 완전히 동일)
# ============================================================

class DSBeliefMap:
    def __init__(self, heightmap=None,
                 hm_origin_x=-10.0, hm_origin_y=-60.0):
        self.E = np.zeros((DS_N_INTERNAL, DS_N_INTERNAL), dtype=np.float32)
        self.F = np.zeros((DS_N_INTERNAL, DS_N_INTERNAL), dtype=np.float32)
        self.U = np.ones ((DS_N_INTERNAL, DS_N_INTERNAL), dtype=np.float32)
        self.terrain_mask = np.zeros(
            (DS_N_INTERNAL, DS_N_INTERNAL), dtype=bool)
        if heightmap is not None:
            self._init_terrain(heightmap, hm_origin_x, hm_origin_y)

    def _init_terrain(self, heightmap, hm_ox, hm_oy):
        hm_h, hm_w = heightmap.shape
        for ci in range(DS_N_INTERNAL):
            wx = DS_MAP_X_MIN + (ci + 0.5) * DS_CELL_W
            xi = int(wx - hm_ox)
            for cj in range(DS_N_INTERNAL):
                wy = DS_MAP_Y_MIN + (cj + 0.5) * DS_CELL_H
                yi = int(wy - hm_oy)
                if 0 <= xi < hm_h and 0 <= yi < hm_w:
                    if heightmap[xi, yi] > 0.5:
                        self.terrain_mask[ci, cj] = True
                        self.E[ci, cj] = 1.0
                        self.F[ci, cj] = 0.0
                        self.U[ci, cj] = 0.0

    def world_to_cell(self, wx, wy):
        ci = int((wx - DS_MAP_X_MIN) / DS_CELL_W)
        cj = int((wy - DS_MAP_Y_MIN) / DS_CELL_H)
        if 0 <= ci < DS_N_INTERNAL and 0 <= cj < DS_N_INTERNAL:
            return ci, cj
        return None

    def cell_center(self, ci, cj):
        wx = DS_MAP_X_MIN + (ci + 0.5) * DS_CELL_W
        wy = DS_MAP_Y_MIN + (cj + 0.5) * DS_CELL_H
        return wx, wy

    @staticmethod
    def _ds_combine(E1, F1, U1, E2, F2, U2):
        EE = E1*E2 + E1*U2 + U1*E2
        FF = F1*F2 + F1*U2 + U1*F2
        UU = U1*U2
        K  = E1*F2 + F1*E2
        norm = 1.0 - K
        if norm < 1e-9:
            return 0.0, 0.0, 1.0
        return float(EE/norm), float(FF/norm), float(UU/norm)

    def update_cell(self, ci, cj, q, has_target):
        if self.terrain_mask[ci, cj]:
            return
        E0 = float(self.E[ci, cj])
        F0 = float(self.F[ci, cj])
        U0 = float(self.U[ci, cj])
        if has_target:
            E_ev, F_ev, U_ev = 0.0, q, 1.0 - q
        else:
            E_ev, F_ev, U_ev = q, 0.0, 1.0 - q
        E1, F1, U1 = self._ds_combine(E0, F0, U0, E_ev, F_ev, U_ev)
        self.E[ci, cj] = float(np.clip(E1, 0.0, 1.0))
        self.F[ci, cj] = float(np.clip(F1, 0.0, 1.0))
        self.U[ci, cj] = float(np.clip(U1, 0.0, 1.0))

    def update_from_drone(self, drone_x, drone_y,
                          target_positions, target_is_camouflaged):
        center = self.world_to_cell(drone_x, drone_y)
        if center is None:
            return
        ci_c, cj_c = center
        for dci in range(-OBS_RADIUS, OBS_RADIUS + 1):
            for dcj in range(-OBS_RADIUS, OBS_RADIUS + 1):
                ci = ci_c + dci
                cj = cj_c + dcj
                if not (0 <= ci < DS_N_INTERNAL and 0 <= cj < DS_N_INTERNAL):
                    continue
                if self.terrain_mask[ci, cj]:
                    continue
                cx, cy = self.cell_center(ci, cj)
                d = float(np.sqrt((drone_x - cx)**2 + (drone_y - cy)**2))
                q = compute_q(d)
                has_target = False
                for t_idx, t_pos in enumerate(target_positions):
                    t_cell = self.world_to_cell(t_pos[0], t_pos[1])
                    if t_cell is not None and t_cell == (ci, cj):
                        has_target = True
                        if target_is_camouflaged[t_idx]:
                            q = q * DS_CAMO_Q_FACTOR
                        break
                self.update_cell(ci, cj, float(np.clip(q, 0.0, 1.0)), has_target)

    def decay(self):
        mask = ~self.terrain_mask
        self.E[mask] *= (1.0 - DS_DECAY_RATE)
        self.F[mask] *= (1.0 - DS_DECAY_RATE)
        self.U[mask]  = np.clip(1.0 - self.E[mask] - self.F[mask], 0.0, 1.0)

    def _pool(self, arr):
        n = DS_POOL_SIZE
        result = np.zeros((DS_N_RL, DS_N_RL), dtype=np.float32)
        for i in range(DS_N_RL):
            for j in range(DS_N_RL):
                result[i, j] = arr[i*n:(i+1)*n, j*n:(j+1)*n].mean()
        return result

    def belief_vector(self) -> np.ndarray:
        return np.concatenate([
            self._pool(self.E).flatten(),
            self._pool(self.F).flatten(),
            self._pool(self.U).flatten(),
        ]).astype(np.float32)

    def total_uncertainty(self) -> float:
        return float(self.U[~self.terrain_mask].sum())

    def reset(self):
        mask = ~self.terrain_mask
        self.E[mask] = 0.0
        self.F[mask] = 0.0
        self.U[mask] = 1.0


# ============================================================
# TargetIntensityMap  (v12와 완전히 동일)
# ============================================================

class TargetIntensityMap:
    _CELL_W  = (DS_MAP_X_MAX - DS_MAP_X_MIN) / DS_N_RL
    _CELL_H  = (DS_MAP_Y_MAX - DS_MAP_Y_MIN) / DS_N_RL
    _ci_arr  = np.arange(DS_N_RL, dtype=np.float32)
    _cj_arr  = np.arange(DS_N_RL, dtype=np.float32)
    _wx_grid = (DS_MAP_X_MIN + (_ci_arr + 0.5) * _CELL_W).reshape(DS_N_RL, 1)
    _wy_grid = (DS_MAP_Y_MIN + (_cj_arr + 0.5) * _CELL_H).reshape(1, DS_N_RL)

    def __init__(self):
        self.grid = np.zeros((DS_N_RL, DS_N_RL), dtype=np.float32)

    def update(self, belief_tracker, target_states: np.ndarray):
        one_minus_T = np.ones((DS_N_RL, DS_N_RL), dtype=np.float32)
        for k in range(belief_tracker.num_targets):
            state = target_states[k]
            w = INTENSITY_W.get(state, 0.0)
            if w == 0.0:
                continue
            mx = float(belief_tracker.mu[k, 0])
            my = float(belief_tracker.mu[k, 1])
            P2 = belief_tracker.P[k][:2, :2].copy()
            if state == TargetState.STALE:
                P2 *= INTENSITY_SIGMA_STALE_MULT ** 2
            sigma2 = max(float(np.trace(P2)) / 2.0, self._CELL_W ** 2)
            dx2 = (self._wx_grid - mx) ** 2
            dy2 = (self._wy_grid - my) ** 2
            contrib = w * np.exp(-0.5 * (dx2 + dy2) / sigma2).astype(np.float32)
            contrib = np.clip(contrib, 0.0, 1.0)
            one_minus_T *= (1.0 - contrib)
        self.grid = 1.0 - one_minus_T

    def get_intensity_obs(self) -> np.ndarray:
        return self.grid.flatten().astype(np.float32)

    def get_obs(self) -> np.ndarray:
        return self.get_intensity_obs()

    def get_fused_priority(self, U_pool: np.ndarray,
                            beta: float = FUSED_BETA) -> np.ndarray:
        U = U_pool.reshape(DS_N_RL, DS_N_RL).astype(np.float32)
        T = self.grid
        fused = U * (1.0 + beta * T) / (1.0 + beta)
        return fused.flatten().astype(np.float32)

    def get_zone_stats(self, zone_idx: int, n_zones: int = 16) -> tuple:
        zi = zone_idx // 4
        zj = zone_idx % 4
        sz = max(DS_N_RL // 4, 1)
        block = self.grid[zi*sz:(zi+1)*sz, zj*sz:(zj+1)*sz]
        return float(block.mean()), float(block.max())

    def reset(self):
        self.grid[:] = 0.0


# ============================================================
# TargetBeliefTracker  (v12와 완전히 동일)
# ============================================================

class TargetBeliefTracker:
    def __init__(self, num_targets: int, target_types: tuple,
                 target_positions_init: np.ndarray):
        self.num_targets  = num_targets
        self.target_types = target_types

        self.mu = np.zeros((num_targets, 4), dtype=np.float64)
        for k in range(num_targets):
            t = target_types[k]
            cx, cy = TARGET_SPAWN_PRIOR.get(t, (30.0, 0.0))
            self.mu[k, 0] = cx + np.random.randn() * np.sqrt(P_INIT_POS) * 0.3
            self.mu[k, 1] = cy + np.random.randn() * np.sqrt(P_INIT_POS) * 0.3
            self.mu[k, 2] = 0.0
            self.mu[k, 3] = 0.0

        self.P = np.zeros((num_targets, 4, 4), dtype=np.float64)
        for k in range(num_targets):
            self.P[k] = np.diag([P_INIT_POS, P_INIT_POS, P_INIT_VEL, P_INIT_VEL])

        self.track_age   = np.full(num_targets, T_STALE + 1, dtype=np.int32)
        self.dwell_count = np.zeros(num_targets, dtype=np.int32)

        self.imm_mu_probs = np.zeros((num_targets, 2), dtype=np.float64)
        self.imm_x = np.zeros((num_targets, 2, 4), dtype=np.float64)
        self.imm_P = np.zeros((num_targets, 2, 4, 4), dtype=np.float64)
        for k in range(num_targets):
            t = target_types[k]
            self.imm_mu_probs[k] = [0.8, 0.2] if t == TargetType.TANK else [0.9, 0.1]
            for j in range(2):
                self.imm_x[k, j] = self.mu[k].copy()
                self.imm_P[k, j] = self.P[k].copy()

        self.prev_tr_P = np.array(
            [np.trace(self.P[k]) for k in range(num_targets)], dtype=np.float64)
        self.target_states = np.full(num_targets, TargetState.UNCOMMITTED, dtype=np.int32)
        self.prev_states   = np.full(num_targets, TargetState.UNCOMMITTED, dtype=np.int32)
        self.tracked_count = 0

    @staticmethod
    def _make_F_cv(dt=1.0):
        return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)

    @staticmethod
    def _make_F_static():
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=np.float64)

    @staticmethod
    def _make_F_ct(omega, dt=1.0):
        if abs(omega) < 1e-6:
            return TargetBeliefTracker._make_F_cv(dt)
        s, c = np.sin(omega*dt), np.cos(omega*dt)
        return np.array([
            [1, 0,  s/omega, -(1-c)/omega],
            [0, 1, (1-c)/omega,  s/omega ],
            [0, 0,    c,          -s      ],
            [0, 0,    s,           c      ],
        ], dtype=np.float64)

    @staticmethod
    def _make_F_slowcv(decay=0.85):
        return np.array([[1,0,1,0],[0,1,0,1],[0,0,decay,0],[0,0,0,decay]], dtype=np.float64)

    @staticmethod
    def _make_Q(q_pos, q_vel):
        return np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)

    def _raw_ekf_update(self, k, z, R):
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        y = z - H @ self.mu[k]
        S = H @ self.P[k] @ H.T + R
        K = self.P[k] @ H.T @ np.linalg.inv(S)
        self.mu[k] = self.mu[k] + K @ y
        self.P[k]  = (np.eye(4) - K @ H) @ self.P[k]

    def _compute_R(self, k, n_obs, is_camo):
        dc = self.dwell_count[k]
        if is_camo:
            r_base = EKF_R_CAMO
        elif dc >= DWELL_FULL:
            r_base = EKF_R_DWELL
        else:
            r_base = EKF_R_INIT * (DWELL_DECAY ** dc)
        r_eff = r_base / max(1, n_obs)
        return np.diag([r_eff, r_eff]).astype(np.float64)

    def _get_imm_config(self, k):
        t = self.target_types[k]
        if t == TargetType.TANK:
            Fs    = [self._make_F_cv(), self._make_F_ct(IMM_TANK_OMEGA)]
            Qs    = [self._make_Q(EKF_Q_POS_TANK_IMM, EKF_Q_VEL_TANK_IMM)] * 2
            Trans = IMM_TANK_TRANS
        elif t == TargetType.ARTILLERY:
            Fs    = [self._make_F_static(), self._make_F_slowcv()]
            Qs    = [self._make_Q(EKF_Q_POS_STATIC, EKF_Q_VEL_STATIC),
                     self._make_Q(EKF_Q_POS_SLOW,   EKF_Q_VEL_SLOW)]
            Trans = IMM_ARTIL_TRANS
        else:
            Fs    = [self._make_F_static(), self._make_F_slowcv()]
            Qs    = [self._make_Q(EKF_Q_POS_STATIC, EKF_Q_VEL_STATIC),
                     self._make_Q(EKF_Q_POS_SLOW,   EKF_Q_VEL_SLOW)]
            Trans = IMM_INF_TRANS
        return Fs, Qs, Trans

    def _imm_predict(self, k):
        Fs, Qs, Trans = self._get_imm_config(k)
        N = 2; mu_p = self.imm_mu_probs[k]
        c_j = np.maximum(Trans.T @ mu_p, 1e-10)
        new_mu_p = c_j / c_j.sum()
        x_mix = np.zeros((N, 4), dtype=np.float64)
        P_mix = np.zeros((N, 4, 4), dtype=np.float64)
        for j in range(N):
            for i in range(N):
                w = Trans[i, j] * mu_p[i] / c_j[j]
                x_mix[j] += w * self.imm_x[k, i]
            for i in range(N):
                w = Trans[i, j] * mu_p[i] / c_j[j]
                d = (self.imm_x[k, i] - x_mix[j]).reshape(4, 1)
                P_mix[j] += w * (self.imm_P[k, i] + d @ d.T)
        for j in range(N):
            self.imm_x[k, j] = Fs[j] @ x_mix[j]
            self.imm_P[k, j] = Fs[j] @ P_mix[j] @ Fs[j].T + Qs[j]
        self.imm_mu_probs[k] = new_mu_p
        self.mu[k] = np.sum(new_mu_p[:, None] * self.imm_x[k], axis=0)
        P_fused = np.zeros((4, 4), dtype=np.float64)
        for j in range(N):
            d = (self.imm_x[k, j] - self.mu[k]).reshape(4, 1)
            P_fused += new_mu_p[j] * (self.imm_P[k, j] + d @ d.T)
        self.P[k] = P_fused

    def _imm_update(self, k, z, R):
        N = 2
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        likelihoods = np.zeros(N, dtype=np.float64)
        x_upd = np.zeros((N, 4), dtype=np.float64)
        P_upd = np.zeros((N, 4, 4), dtype=np.float64)
        for j in range(N):
            y_j = z - H @ self.imm_x[k, j]
            S_j = H @ self.imm_P[k, j] @ H.T + R
            det_S = max(np.linalg.det(S_j), 1e-10)
            likelihoods[j] = (
                np.exp(-0.5 * y_j @ np.linalg.inv(S_j) @ y_j)
                / np.sqrt((2*np.pi)**2 * det_S)
            )
            K_j = self.imm_P[k, j] @ H.T @ np.linalg.inv(S_j)
            x_upd[j] = self.imm_x[k, j] + K_j @ y_j
            P_upd[j] = (np.eye(4) - K_j @ H) @ self.imm_P[k, j]
        new_mu_p = likelihoods * self.imm_mu_probs[k]
        new_mu_p = new_mu_p / max(new_mu_p.sum(), 1e-10)
        self.imm_mu_probs[k] = new_mu_p
        self.imm_x[k] = x_upd
        self.imm_P[k] = P_upd
        self.mu[k] = np.sum(new_mu_p[:, None] * x_upd, axis=0)
        P_fused = np.zeros((4, 4), dtype=np.float64)
        for j in range(N):
            d = (x_upd[j] - self.mu[k]).reshape(4, 1)
            P_fused += new_mu_p[j] * (P_upd[j] + d @ d.T)
        self.P[k] = P_fused

    def predict_all(self):
        self.prev_tr_P = np.array(
            [np.trace(self.P[k]) for k in range(self.num_targets)], dtype=np.float64)
        Q_stat = self._make_Q(EKF_Q_POS_STATIC, EKF_Q_VEL_STATIC)
        for k in range(self.num_targets):
            t = self.target_types[k]
            if t == TargetType.ANTI_AIR:
                F = self._make_F_static()
                self.mu[k] = F @ self.mu[k]
                self.P[k]  = F @ self.P[k] @ F.T + Q_stat
                self.mu[k, 2:] = 0.0
                self.P[k, 2:, 2:] = np.eye(2) * 0.01
            else:
                self._imm_predict(k)
            self.track_age[k] += 1
            self.P[k] = np.clip(self.P[k], -1e4, 1e4)

    def update(self, k, z_true, is_camouflaged, n_obs):
        R = self._compute_R(k, n_obs, is_camouflaged)
        r_noise = np.sqrt(np.diag(R)[:2])
        z_noisy = z_true.astype(np.float64) + np.random.randn(2) * r_noise
        t = self.target_types[k]
        if t == TargetType.ANTI_AIR:
            self._raw_ekf_update(k, z_noisy, R)
        else:
            self._imm_update(k, z_noisy, R)
        self.dwell_count[k] = min(self.dwell_count[k] + 1, DWELL_FULL)
        self.track_age[k] = 0

    def reset_dwell(self, k):
        self.dwell_count[k] = 0

    def get_tr_P(self, k):
        return float(np.trace(self.P[k]))

    def get_tr_P_norm(self, k):
        return float(np.clip(np.trace(self.P[k]) / TR_P_MAX, 0.0, 1.0))

    def get_age_norm(self, k):
        return float(np.clip(self.track_age[k] / (T_STALE * 2), 0.0, 1.0))

    def update_target_states(self):
        self.prev_states[:] = self.target_states[:]
        for k in range(self.num_targets):
            trP = self.get_tr_P_norm(k)
            age = self.track_age[k]
            dw  = self.dwell_count[k]
            st  = self.target_states[k]
            if st == TargetState.UNCOMMITTED:
                if trP < TR_P_CANDIDATE_THRESH and age == 0:
                    self.target_states[k] = TargetState.CANDIDATE
            elif st == TargetState.CANDIDATE:
                if trP < TR_P_COMMITTED_THRESH and dw >= DWELL_FOR_COMMIT:
                    self.target_states[k] = TargetState.COMMITTED
                elif age > T_STALE:
                    self.target_states[k] = TargetState.UNCOMMITTED
            elif st == TargetState.COMMITTED:
                if age > T_STALE:
                    self.target_states[k] = TargetState.STALE
            elif st == TargetState.STALE:
                if age == 0 and trP < TR_P_RECOVER_THRESH:
                    self.target_states[k] = TargetState.COMMITTED

    def get_active_tracks(self, k_active=K_ACTIVE):
        scores = []
        for k in range(self.num_targets):
            st   = self.target_states[k]
            base = PRIORITY_BASE.get(st, 0.0)
            trP  = self.get_tr_P_norm(k)
            if st == TargetState.STALE:
                score = base + 1.0 * trP
            elif st == TargetState.COMMITTED:
                recent = 1.0 if self.track_age[k] == 0 else 0.0
                score  = base + 0.5 * recent + 0.5 * (1.0 - trP)
            elif st == TargetState.CANDIDATE:
                recent = 1.0 if self.track_age[k] == 0 else 0.0
                score  = base + 0.5 * recent + 0.5 * (1.0 - trP)
            else:
                score = 0.0
            scores.append((score, k))
        scores.sort(reverse=True)
        active = [k for _, k in scores[:k_active]]
        while len(active) < k_active:
            active.append(-1)
        return active

    def reset(self, target_positions_init=None):
        for k in range(self.num_targets):
            self.P[k] = np.diag([P_INIT_POS, P_INIT_POS, P_INIT_VEL, P_INIT_VEL])
            t = self.target_types[k]
            cx, cy = TARGET_SPAWN_PRIOR.get(t, (30.0, 0.0))
            self.mu[k, 0] = cx + np.random.randn() * np.sqrt(P_INIT_POS) * 0.3
            self.mu[k, 1] = cy + np.random.randn() * np.sqrt(P_INIT_POS) * 0.3
            self.mu[k, 2:] = 0.0
            if t == TargetType.TANK:
                self.imm_mu_probs[k] = [0.8, 0.2]
            elif t == TargetType.ANTI_AIR:
                self.imm_mu_probs[k] = [1.0, 0.0]
            else:
                self.imm_mu_probs[k] = [0.9, 0.1]
            for j in range(2):
                self.imm_x[k, j] = self.mu[k].copy()
                self.imm_P[k, j] = self.P[k].copy()
        self.track_age[:]   = T_STALE + 1
        self.dwell_count[:] = 0
        self.prev_tr_P = np.array(
            [np.trace(self.P[k]) for k in range(self.num_targets)], dtype=np.float64)
        self.target_states[:] = TargetState.UNCOMMITTED
        self.prev_states[:]   = TargetState.UNCOMMITTED
        self.tracked_count = 0

    def active_track_obs_vector(self, drone_pos, map_w, map_h, active_tracks):
        vec = np.zeros(len(active_tracks) * 6, dtype=np.float32)
        V_MAX = 5.0
        for i, k in enumerate(active_tracks):
            if k < 0:
                continue
            base = i * 6
            vec[base+0] = float(np.clip((self.mu[k,0]-drone_pos[0])/map_w, -1.0, 1.0))
            vec[base+1] = float(np.clip((self.mu[k,1]-drone_pos[1])/map_h, -1.0, 1.0))
            vec[base+2] = float(np.clip(self.mu[k,2]/V_MAX, -1.0, 1.0))
            vec[base+3] = float(np.clip(self.mu[k,3]/V_MAX, -1.0, 1.0))
            vec[base+4] = self.get_tr_P_norm(k)
            vec[base+5] = self.get_age_norm(k)
        return vec

    def belief_obs_vector(self, drone_pos, map_w, map_h):
        active = self.get_active_tracks(K_ACTIVE)
        return self.active_track_obs_vector(drone_pos, map_w, map_h, active)


# ============================================================
# DroneSwarmEnv  v13  (Isaac Sim 완전 제거)
# ============================================================

class DroneSwarmEnv:
    """
    Pure NumPy 드론 군집 RL 환경 v13.

    Isaac Sim / omni 의존성 완전 제거.
    obs / reward / done / info 구조는 v12와 100% 동일.
    """

    GRID_MAP_X    = 65
    GRID_MAP_Y    = 60
    GRID_ORIGIN_X = -10.0
    GRID_ORIGIN_Y = -30.0

    TERRAIN_NUM_MOUNTAINS_RANGE = (6, 12)
    TERRAIN_X_RANGE             = (5, 60)
    TERRAIN_Y_RANGE             = (-27, 27)
    TERRAIN_RADIUS_RANGE        = (4, 10)
    TERRAIN_HEIGHT_RANGE        = (4, 10)

    DRONE_STARTS = [
        np.array([ 0.0,  0.0, DRONE_ALTITUDE]),
        np.array([ 0.0,  8.0, DRONE_ALTITUDE]),
        np.array([ 8.0,  0.0, DRONE_ALTITUDE]),
        np.array([ 8.0,  8.0, DRONE_ALTITUDE]),
    ]

    ACTION_MOVES = {
        0: np.array([ DRONE_STEP_M,  0.0]),
        1: np.array([-DRONE_STEP_M,  0.0]),
        2: np.array([ 0.0,  DRONE_STEP_M]),
        3: np.array([ 0.0, -DRONE_STEP_M]),
    }

    def __init__(self,
                 randomize_terrain: bool = False,
                 terrain_seed: int = 42,
                 num_drones: int   = NUM_DRONES,
                 num_targets: int  = NUM_TARGETS,
                 k_active: int     = K_ACTIVE):

        self.randomize_terrain = randomize_terrain
        self.terrain_seed      = terrain_seed if not randomize_terrain else None
        self.num_drones        = num_drones
        self.k_active          = k_active

        # 타겟 수 (v12 동일 비율, AntiAir는 항상 2개 고정 → num_anti_air로 수정)
        base = num_targets // 20
        self.num_infantry  = 8  * max(1, base)
        self.num_artillery = 8  * max(1, base)
        self.num_tanks     = 2  * max(1, base)
        self.num_anti_air  = 2  * max(1, base)   # ← v13 버그 수정: 하드코딩 2개 → 가변
        self.num_targets   = (self.num_infantry + self.num_artillery
                              + self.num_tanks + self.num_anti_air)

        # 드론 시작 위치 (최대 12대 지원)
        _full_grid = [
            np.array([ 0.0,  0.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 0.0,  8.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 8.0,  0.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 8.0,  8.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 0.0, -8.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 8.0, -8.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 0.0, 16.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 8.0, 16.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 0.0,-16.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 8.0,-16.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 0.0, 24.0, DRONE_ALTITUDE], dtype=np.float32),
            np.array([ 8.0, 24.0, DRONE_ALTITUDE], dtype=np.float32),
        ]
        _starts = [_full_grid[i] if i < len(_full_grid) else _full_grid[-1]
                   for i in range(num_drones)]

        # 드론 상태
        self.drone_positions = np.array([p.copy() for p in _starts], dtype=np.float32)
        self.drone_active    = np.ones(num_drones, dtype=bool)
        self.drone_battery   = np.full(num_drones, BATTERY_INIT, dtype=np.float32)
        self.drone_returning = np.zeros(num_drones, dtype=bool)
        self.base_positions  = np.array([p[:2].copy() for p in _starts], dtype=np.float32)

        # 타겟 상태
        self.target_positions      = np.zeros((self.num_targets, 3), dtype=np.float32)
        self.target_types          = []
        self.target_velocities     = np.zeros((self.num_targets, 3), dtype=np.float32)
        self.target_values         = []
        self.target_is_camouflaged = np.zeros(self.num_targets, dtype=bool)
        self.target_found          = np.zeros(self.num_targets, dtype=bool)

        # v12: 포병 진지변환
        self.artil_relo_target = {}
        self.artil_relo_timer  = {}

        # v12: 보병 이동 상태
        self.inf_moving  = np.zeros(self.num_targets, dtype=bool)
        self.inf_heading = np.zeros(self.num_targets, dtype=np.float32)
        self.inf_speed   = np.zeros(self.num_targets, dtype=np.float32)

        # v13: 전차 patrol_bounds (Isaac Sim prim 속성 대신 dict 배열로 관리)
        self.tank_patrol_bounds = {}

        self.current_step      = 0
        self._prev_uncertainty = 0.0
        self.belief_tracker    = None

        # 지형 생성 (Isaac Sim 없이 heightmap만)
        self._setup_scene()

        # DS 맵
        self.ds_map = DSBeliefMap(
            heightmap   = self.heightmap,
            hm_origin_x = self.GRID_ORIGIN_X,
            hm_origin_y = self.GRID_ORIGIN_Y,
        )
        self.intensity_map = TargetIntensityMap()

        print(f"\nDroneSwarmEnv v13 (Pure NumPy) 초기화 완료")
        print(f"  지형   : {'랜덤' if randomize_terrain else f'고정(seed={terrain_seed})'}")
        print(f"  드론   : {num_drones}대  |  타겟 : {self.num_targets}개  |  K_active : {k_active}")
        print(f"  관측   : {OBS_TOTAL_DIM}차원 (DS300+depth8+self2+fused100+active{k_active*6}+bat2)")
        print(f"  Isaac Sim : 완전 제거 (Pure NumPy)")

    # ── 지형 생성 ─────────────────────────────────────────────

    @staticmethod
    def _generate_mountain_specs(seed=None):
        rng = np.random.default_rng(seed)
        n   = int(rng.integers(*DroneSwarmEnv.TERRAIN_NUM_MOUNTAINS_RANGE))
        specs = []
        for _ in range(n):
            specs.append((
                float(rng.uniform(*DroneSwarmEnv.TERRAIN_X_RANGE)),
                float(rng.uniform(*DroneSwarmEnv.TERRAIN_Y_RANGE)),
                int(rng.integers(*DroneSwarmEnv.TERRAIN_RADIUS_RANGE)),
                float(rng.uniform(*DroneSwarmEnv.TERRAIN_HEIGHT_RANGE)),
            ))
        return specs

    def _build_heightmap(self, rng, specs):
        hm = np.zeros((self.GRID_MAP_X, self.GRID_MAP_Y), dtype=np.float32)
        for (mx, my, radius, max_h) in specs:
            gx = int(mx - self.GRID_ORIGIN_X)
            gy = int(my - self.GRID_ORIGIN_Y)
            for dx in range(-radius-3, radius+4):
                for dy in range(-radius-3, radius+4):
                    xi, yi = gx+dx, gy+dy
                    if not (0 <= xi < self.GRID_MAP_X and 0 <= yi < self.GRID_MAP_Y):
                        continue
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > radius:
                        continue
                    h  = max_h * np.exp(-0.5 * (dist / (radius * 0.5))**2)
                    h *= float(rng.uniform(0.75, 1.15))
                    hm[xi, yi] = max(hm[xi, yi], h)
        return np.clip(hm, 0.0, 10.0)

    def _setup_scene(self):
        seed = self.terrain_seed if self.terrain_seed is not None else random.randint(0, 99999)
        specs = self._generate_mountain_specs(seed=seed)
        rng   = np.random.default_rng(seed)
        self.heightmap = self._build_heightmap(rng, specs)
        print(f"  지형 블록 생성 완료 (seed={seed}, 산={len(specs)}개)")
        self._place_targets()
        print("Scene 설정 완료")

    # ── 지형 유틸 ─────────────────────────────────────────────

    def _get_terrain_height(self, x, y) -> float:
        xi = int(x - self.GRID_ORIGIN_X)
        yi = int(y - self.GRID_ORIGIN_Y)
        if not (0 <= xi < self.GRID_MAP_X and 0 <= yi < self.GRID_MAP_Y):
            return 0.0
        return float(self.heightmap[xi, yi])

    def _is_terrain(self, x, y) -> bool:
        return self._get_terrain_height(x, y) > 0.5

    def _is_clear(self, x, y, min_d=5) -> bool:
        xi_c = int(x - self.GRID_ORIGIN_X)
        yi_c = int(y - self.GRID_ORIGIN_Y)
        rc   = int(np.ceil(min_d))
        for dxi in range(-rc, rc+1):
            for dyi in range(-rc, rc+1):
                if np.sqrt(dxi*dxi + dyi*dyi) > min_d:
                    continue
                xi, yi = xi_c+dxi, yi_c+dyi
                if not (0 <= xi < self.GRID_MAP_X and 0 <= yi < self.GRID_MAP_Y):
                    continue
                if self.heightmap[xi, yi] > 0.5:
                    return False
        return True

    def _find_clear_pos(self, xr, yr, z=0.0, attempts=50):
        for _ in range(attempts):
            x = float(round(random.uniform(*xr)))
            y = float(round(random.uniform(*yr)))
            if self._is_clear(x, y):
                return np.array([x, y, z], dtype=np.float32)
        return np.array([float(round(random.uniform(*xr))),
                         float(round(random.uniform(*yr))), z], dtype=np.float32)

    # ── 타겟 배치 ─────────────────────────────────────────────

    def _place_targets(self):
        self.target_types      = []
        self.target_positions  = np.zeros((self.num_targets, 3), dtype=np.float32)
        self.target_velocities = np.zeros((self.num_targets, 3), dtype=np.float32)
        self.target_values     = []
        self.tank_patrol_bounds = {}
        idx = 0

        # Infantry
        for i in range(self.num_infantry):
            yr  = (10, 30) if i < 4 else (-30, -10)
            pos = self._find_clear_pos((5, 20), yr, z=0.5)
            self.target_types.append(TargetType.INFANTRY)
            self.target_positions[idx] = pos
            self.target_values.append(random.uniform(1.0, 1.5))
            idx += 1

        # Tank
        for i in range(self.num_tanks):
            pos   = self._find_clear_pos(TANK_X_RANGE, TANK_Y_RANGE, z=0.5)
            angle = random.uniform(0, 2*np.pi)
            self.tank_patrol_bounds[idx] = {
                'x_min': TANK_X_RANGE[0], 'x_max': TANK_X_RANGE[1],
                'y_min': TANK_Y_RANGE[0], 'y_max': TANK_Y_RANGE[1],
            }
            self.target_types.append(TargetType.TANK)
            self.target_positions[idx] = pos
            self.target_velocities[idx] = [2*np.cos(angle), 2*np.sin(angle), 0]
            self.target_values.append(random.uniform(2.1, 3.0))
            idx += 1

        # Artillery
        for i in range(self.num_artillery):
            yr  = (10, 25) if i < self.num_artillery // 2 else (-25, -10)
            pos = self._find_clear_pos((42, 60), yr, z=0.4)
            self.target_types.append(TargetType.ARTILLERY)
            self.target_positions[idx] = pos
            self.target_values.append(random.uniform(3.0, 3.5))
            idx += 1

        # AntiAir — v13 버그 수정: num_anti_air개 동적 생성 (v12는 하드코딩 2개)
        # y 대칭 배치: 짝수 → 양수 y, 홀수 → 음수 y
        aa_y_pairs = [
            ((45, 57), (12, 22)),
            ((45, 57), (-22, -12)),
            ((50, 57), (14, 20)),
            ((45, 52), (-20, -14)),
            ((46, 56), (16, 22)),
            ((46, 56), (-22, -16)),
        ]
        for i in range(self.num_anti_air):
            if i < len(aa_y_pairs):
                xr, yr = aa_y_pairs[i]
            else:
                # 추가 AA: x(45,57) y 교대
                half = (i // 2) + 1
                sign = 1 if i % 2 == 0 else -1
                xr = (45, 57)
                yr = (sign * 12, sign * 22) if sign > 0 else (sign * 22, sign * 12)
            pos = self._find_clear_pos(xr, yr, z=0.6)
            self.target_types.append(TargetType.ANTI_AIR)
            self.target_positions[idx] = pos
            self.target_values.append(random.uniform(1.7, 2.0))
            idx += 1

        self.target_types = tuple(self.target_types)

        # 포병 relocation 타이머
        self.artil_relo_target = {}
        self.artil_relo_timer  = {}
        for i, t in enumerate(self.target_types):
            if t == TargetType.ARTILLERY:
                jitter = random.randint(0, ARTILLERY_RELO_JITTER)
                self.artil_relo_timer[i] = ARTILLERY_RELO_INTERVAL + jitter

        print(f"  타겟 {self.num_targets}개 배치 완료")

        self.belief_tracker = TargetBeliefTracker(
            num_targets          = self.num_targets,
            target_types         = self.target_types,
            target_positions_init= self.target_positions,
        )

    def _reset_targets(self):
        idx = 0

        for i in range(self.num_infantry):
            yr  = (10, 30) if i < 4 else (-30, -10)
            pos = self._find_clear_pos((5, 20), yr, z=0.5)
            self.target_positions[idx] = pos
            idx += 1

        for i in range(self.num_tanks):
            pos   = self._find_clear_pos(TANK_X_RANGE, TANK_Y_RANGE, z=0.5)
            angle = random.uniform(0, 2*np.pi)
            self.target_positions[idx] = pos
            self.target_velocities[idx] = [2*np.cos(angle), 2*np.sin(angle), 0]
            idx += 1

        for i in range(self.num_artillery):
            yr  = (10, 25) if i < self.num_artillery // 2 else (-25, -10)
            pos = self._find_clear_pos((42, 60), yr, z=0.4)
            self.target_positions[idx] = pos
            idx += 1

        aa_y_pairs = [
            ((45, 57), (12, 22)), ((45, 57), (-22, -12)),
            ((50, 57), (14, 20)), ((45, 52), (-20, -14)),
            ((46, 56), (16, 22)), ((46, 56), (-22, -16)),
        ]
        for i in range(self.num_anti_air):
            if i < len(aa_y_pairs):
                xr, yr = aa_y_pairs[i]
            else:
                half = (i // 2) + 1
                sign = 1 if i % 2 == 0 else -1
                xr = (45, 57)
                yr = (sign * 12, sign * 22) if sign > 0 else (sign * 22, sign * 12)
            pos = self._find_clear_pos(xr, yr, z=0.6)
            self.target_positions[idx] = pos
            idx += 1

        self.target_is_camouflaged = np.zeros(self.num_targets, dtype=bool)
        self.target_found          = np.zeros(self.num_targets, dtype=bool)

        self.artil_relo_target = {}
        self.artil_relo_timer  = {}
        for i, t in enumerate(self.target_types):
            if t == TargetType.ARTILLERY:
                jitter = random.randint(0, ARTILLERY_RELO_JITTER)
                self.artil_relo_timer[i] = ARTILLERY_RELO_INTERVAL + jitter

        self.inf_moving[:]  = False
        self.inf_heading[:] = 0.0
        self.inf_speed[:]   = 0.0

        if self.belief_tracker is not None:
            self.belief_tracker.reset()

    # ── 타겟 이동 ─────────────────────────────────────────────

    def _update_infantry(self, step):
        for i in range(self.num_infantry):
            if self.target_is_camouflaged[i]:
                continue
            pos = self.target_positions[i].copy()
            if self.inf_moving[i]:
                if random.random() < INF_P_STOP:
                    self.inf_moving[i]  = False
                    self.inf_speed[i]   = 0.0
                    self.target_velocities[i, :2] = 0.0
            else:
                if random.random() < INF_P_MOVE:
                    self.inf_moving[i]   = True
                    self.inf_heading[i]  = float(random.uniform(0, 2 * np.pi))
                    self.inf_speed[i]    = float(random.uniform(INF_SPEED_MIN, INF_SPEED_MAX))
            if self.inf_moving[i]:
                self.inf_heading[i] += float(np.random.randn() * INF_TURN_SIGMA)
                vx = self.inf_speed[i] * np.cos(self.inf_heading[i])
                vy = self.inf_speed[i] * np.sin(self.inf_heading[i])
                nx = pos[0] + vx
                ny = pos[1] + vy
                if (self._is_terrain(nx, ny) or not (5 <= nx <= 20)
                        or not (-30 <= ny <= 30)):
                    self.inf_heading[i] += np.pi
                    nx, ny = float(pos[0]), float(pos[1])
                    self.inf_moving[i] = False
                    self.inf_speed[i]  = 0.0
                self.target_velocities[i, 0] = float(vx)
                self.target_velocities[i, 1] = float(vy)
                self.target_positions[i] = np.array([nx, ny, 0.5], dtype=np.float32)

    def _update_tanks(self):
        dt = 1.0
        LOOKAHEAD = 3
        for i, t_type in enumerate(self.target_types):
            if t_type != TargetType.TANK:
                continue
            vel = self.target_velocities[i].copy()
            pos = self.target_positions[i].copy()
            dW = np.random.randn(2) * np.sqrt(dt)
            vel[:2] += -TANK_OU_THETA * vel[:2] * dt + TANK_OU_SIGMA * dW
            # patrol bounds (v13: dict에서 읽음)
            b = self.tank_patrol_bounds.get(i, {
                'x_min': TANK_X_RANGE[0], 'x_max': TANK_X_RANGE[1],
                'y_min': TANK_Y_RANGE[0], 'y_max': TANK_Y_RANGE[1],
            })
            for axis, lo, hi in [(0, b['x_min'], b['x_max']),
                                  (1, b['y_min'], b['y_max'])]:
                if pos[axis] < lo + TANK_SOFT_BOUNDARY:
                    vel[axis] += (lo + TANK_SOFT_BOUNDARY - pos[axis]) / TANK_SOFT_BOUNDARY * 1.5
                if pos[axis] > hi - TANK_SOFT_BOUNDARY:
                    vel[axis] -= (pos[axis] - (hi - TANK_SOFT_BOUNDARY)) / TANK_SOFT_BOUNDARY * 1.5
            speed = np.linalg.norm(vel[:2])
            if speed > TANK_SPEED_MAX:
                vel[:2] = vel[:2] / speed * TANK_SPEED_MAX
            ahead_blocked = False
            if speed > 0.1:
                for step_la in range(1, LOOKAHEAD + 1):
                    lx = pos[0] + vel[0] / speed * step_la * 1.5
                    ly = pos[1] + vel[1] / speed * step_la * 1.5
                    if self._is_terrain(lx, ly):
                        ahead_blocked = True
                        break
            if ahead_blocked:
                ang = np.arctan2(vel[1], vel[0])
                turned = False
                for delta in [np.pi/2, -np.pi/2, np.pi*2/3, -np.pi*2/3]:
                    new_ang = ang + delta
                    test_vx = np.cos(new_ang) * speed
                    test_vy = np.sin(new_ang) * speed
                    if not self._is_terrain(pos[0]+test_vx, pos[1]+test_vy):
                        vel[:2] = [test_vx, test_vy]
                        turned = True
                        break
                if not turned:
                    vel[:2] *= -0.8
            nx = pos[0] + vel[0] * dt
            ny = pos[1] + vel[1] * dt
            if self._is_terrain(nx, pos[1]): vel[0] *= -0.6; nx = float(pos[0])
            if self._is_terrain(pos[0], ny): vel[1] *= -0.6; ny = float(pos[1])
            if self._is_terrain(nx, ny):     vel *= -0.6;     nx, ny = float(pos[0]), float(pos[1])
            nx = float(np.clip(nx, TANK_X_RANGE[0], TANK_X_RANGE[1]))
            ny = float(np.clip(ny, TANK_Y_RANGE[0], TANK_Y_RANGE[1]))
            self.target_velocities[i] = vel
            self.target_positions[i]  = np.array([nx, ny, 0.5], dtype=np.float32)

    def _update_artillery(self, step):
        artil_start = self.num_infantry + self.num_tanks
        artil_end   = artil_start + self.num_artillery
        for i in range(artil_start, artil_end):
            pos = self.target_positions[i].copy()
            if i in self.artil_relo_target:
                tx, ty = self.artil_relo_target[i]
                dx, dy = tx - pos[0], ty - pos[1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < ARTILLERY_RELO_SPEED * 1.5:
                    self.target_positions[i] = np.array([tx, ty, 0.4], dtype=np.float32)
                    self.target_velocities[i, :2] = 0.0
                    del self.artil_relo_target[i]
                    jitter = random.randint(0, ARTILLERY_RELO_JITTER)
                    self.artil_relo_timer[i] = ARTILLERY_RELO_INTERVAL + jitter
                else:
                    nx = pos[0] + dx / dist * ARTILLERY_RELO_SPEED
                    ny = pos[1] + dy / dist * ARTILLERY_RELO_SPEED
                    if self._is_terrain(nx, ny):
                        new_target = self._find_clear_pos(
                            (42, 60), (pos[1]-15, pos[1]+15), z=0.4)
                        self.artil_relo_target[i] = (new_target[0], new_target[1])
                    else:
                        vx = dx / dist * ARTILLERY_RELO_SPEED
                        vy = dy / dist * ARTILLERY_RELO_SPEED
                        self.target_velocities[i, 0] = float(vx)
                        self.target_velocities[i, 1] = float(vy)
                        self.target_positions[i] = np.array([nx, ny, 0.4], dtype=np.float32)
            else:
                self.target_velocities[i, :2] = 0.0
                self.artil_relo_timer[i] = self.artil_relo_timer.get(i, ARTILLERY_RELO_INTERVAL) - 1
                if self.artil_relo_timer[i] <= 0:
                    if random.random() < ARTILLERY_RELO_PROB:
                        xr = (max(42, pos[0]-20), min(60, pos[0]+20))
                        yr = (max(-25, pos[1]-15), min(25, pos[1]+15))
                        new_target = self._find_clear_pos(xr, yr, z=0.4)
                        if np.linalg.norm(new_target[:2] - pos[:2]) > 5.0:
                            self.artil_relo_target[i] = (new_target[0], new_target[1])
                    jitter = random.randint(0, ARTILLERY_RELO_JITTER)
                    self.artil_relo_timer[i] = ARTILLERY_RELO_INTERVAL + jitter

    def _update_camouflage(self, step):
        for i, t_type in enumerate(self.target_types):
            params = TARGET_PARAMS[t_type]
            if step % params["camouflage_check"] != 0:
                continue
            if self.target_is_camouflaged[i]:
                if random.random() < params["reveal_prob"]:
                    self.target_is_camouflaged[i] = False
            else:
                if random.random() < params["camouflage_prob"]:
                    self.target_is_camouflaged[i] = True

    # ── AntiAir / 충돌 ────────────────────────────────────────

    def _check_antiair(self) -> list:
        shot_down = []
        aa_idx_list = [i for i, t in enumerate(self.target_types)
                       if t == TargetType.ANTI_AIR]
        for d_idx in range(self.num_drones):
            if not self.drone_active[d_idx]:
                continue
            dx = float(self.drone_positions[d_idx, 0])
            dy = float(self.drone_positions[d_idx, 1])
            for aa_idx in aa_idx_list:
                aa   = self.target_positions[aa_idx]
                dist = float(np.sqrt((dx-aa[0])**2 + (dy-aa[1])**2))
                if dist < ANTIAIR_KILL_RADIUS:
                    p_kill = ANTIAIR_P_MAX * (1.0 - dist / ANTIAIR_KILL_RADIUS)
                    if np.random.random() < p_kill:
                        if d_idx not in shot_down:
                            shot_down.append(d_idx)
                        break
        return shot_down

    def _check_drone_collisions(self) -> list:
        pairs = []
        for i in range(self.num_drones):
            if not self.drone_active[i]:
                continue
            for j in range(i+1, self.num_drones):
                if not self.drone_active[j]:
                    continue
                dist = float(np.linalg.norm(
                    self.drone_positions[i, :2] - self.drone_positions[j, :2]))
                if dist < DRONE_SAFE_DIST:
                    pairs.append((i, j))
        return pairs

    # ── Depth (heightmap ray-cast) ────────────────────────────

    def _depth_from_heightmap(self, drone_idx: int) -> np.ndarray:
        depth_obs = np.ones(8, dtype=np.float32)
        px = float(self.drone_positions[drone_idx, 0])
        py = float(self.drone_positions[drone_idx, 1])
        for k, angle_deg in enumerate(DEPTH_DIRS):
            a_rad = np.radians(angle_deg)
            dx_r  = np.cos(a_rad)
            dy_r  = np.sin(a_rad)
            min_dist = DEPTH_MAX_RANGE
            for r in np.linspace(1.0, DEPTH_MAX_RANGE, 30):
                wx = px + dx_r * r
                wy = py + dy_r * r
                if self._get_terrain_height(wx, wy) > DRONE_ALTITUDE:
                    min_dist = r
                    break
            depth_obs[k] = float(np.clip(min_dist / DEPTH_MAX_RANGE, 0.0, 1.0))
        return depth_obs

    # ── 드론 이동 ─────────────────────────────────────────────

    def _move_drone(self, drone_idx: int, action: int):
        i   = drone_idx
        pos = self.drone_positions[i]
        nz  = DRONE_ALTITUDE

        if self.drone_returning[i]:
            base = self.base_positions[i]
            d    = base - pos[:2]
            dist = float(np.linalg.norm(d))
            if dist < BASE_ARRIVAL_DIST:
                self.drone_active[i]    = False
                self.drone_returning[i] = False
                return False, False
            step_vec = d / dist * min(DRONE_STEP_M, dist)
            nx = float(np.clip(pos[0] + step_vec[0], DS_MAP_X_MIN, DS_MAP_X_MAX))
            ny = float(np.clip(pos[1] + step_vec[1], DS_MAP_Y_MIN, DS_MAP_Y_MAX))
            self.drone_positions[i] = np.array([nx, ny, nz], dtype=np.float32)
            self.drone_battery[i]   = max(0.0, self.drone_battery[i] - BATTERY_COST_MOVE)
            return False, False

        move = self.ACTION_MOVES[action]
        nx   = float(pos[0] + move[0])
        ny   = float(pos[1] + move[1])

        if not (DS_MAP_X_MIN <= nx <= DS_MAP_X_MAX and DS_MAP_Y_MIN <= ny <= DS_MAP_Y_MAX):
            return False, True

        if self._get_terrain_height(nx, ny) >= nz:
            return True, False

        self.drone_positions[i] = np.array([nx, ny, nz], dtype=np.float32)
        self.drone_battery[i]   = max(0.0, self.drone_battery[i] - BATTERY_COST_MOVE)
        return False, False

    def _get_dist_to_base(self, drone_idx: int) -> float:
        return float(np.linalg.norm(
            self.drone_positions[drone_idx, :2] - self.base_positions[drone_idx]))

    def _rtb_threshold(self, drone_idx: int) -> float:
        dist = self._get_dist_to_base(drone_idx)
        return dist * (BATTERY_COST_MOVE / DRONE_STEP_M) * BATTERY_SAFETY_FACTOR

    # ── 관측 벡터 ─────────────────────────────────────────────

    def _get_obs_single(self, drone_idx: int) -> np.ndarray:
        ds_vec    = self.ds_map.belief_vector()
        depth_vec = self._depth_from_heightmap(drone_idx)

        px = float(np.clip((self.drone_positions[drone_idx, 0] - DS_MAP_X_MIN)
                           / (DS_MAP_X_MAX - DS_MAP_X_MIN), 0.0, 1.0))
        py = float(np.clip((self.drone_positions[drone_idx, 1] - DS_MAP_Y_MIN)
                           / (DS_MAP_Y_MAX - DS_MAP_Y_MIN), 0.0, 1.0))
        self_pos = np.array([px, py], dtype=np.float32)

        map_w = DS_MAP_X_MAX - DS_MAP_X_MIN
        map_h = DS_MAP_Y_MAX - DS_MAP_Y_MIN

        U_pool    = self.ds_map._pool(self.ds_map.U)
        fused_vec = self.intensity_map.get_fused_priority(U_pool)

        drone_xy      = self.drone_positions[drone_idx, :2]
        active_tracks = self.belief_tracker.get_active_tracks(self.k_active)
        active_vec    = self.belief_tracker.active_track_obs_vector(
            drone_pos=drone_xy, map_w=map_w, map_h=map_h, active_tracks=active_tracks)

        bat_norm  = float(np.clip(self.drone_battery[drone_idx], 0.0, 1.0))
        diag      = float(np.sqrt(map_w**2 + map_h**2))
        dist_base = self._get_dist_to_base(drone_idx)
        dist_norm = float(np.clip(dist_base / diag, 0.0, 1.0))
        battery_vec = np.array([bat_norm, dist_norm], dtype=np.float32)

        return np.concatenate(
            [ds_vec, depth_vec, self_pos, fused_vec, active_vec, battery_vec]
        ).astype(np.float32)

    def get_obs(self) -> np.ndarray:
        return np.stack([self._get_obs_single(i) for i in range(self.num_drones)], axis=0)

    # ── RL 인터페이스 ─────────────────────────────────────────

    def set_eval_mode(self, eval_mode: bool = True):
        self.randomize_terrain = eval_mode

    def reset(self) -> np.ndarray:
        if self.randomize_terrain:
            new_seed = random.randint(0, 99999)
            specs = self._generate_mountain_specs(seed=new_seed)
            rng   = np.random.default_rng(new_seed)
            self.heightmap = self._build_heightmap(rng, specs)
            self.ds_map = DSBeliefMap(
                heightmap   = self.heightmap,
                hm_origin_x = self.GRID_ORIGIN_X,
                hm_origin_y = self.GRID_ORIGIN_Y,
            )

        # 드론 초기화 (base_positions에서 복원)
        for i in range(self.num_drones):
            self.drone_positions[i, :2] = self.base_positions[i]
            self.drone_positions[i, 2]  = DRONE_ALTITUDE
        self.drone_active[:]    = True
        self.drone_battery[:]   = BATTERY_INIT
        self.drone_returning[:] = False

        # 타겟 초기화
        if self.belief_tracker is None:
            self._place_targets()
        else:
            self._reset_targets()

        self.ds_map.reset()
        self.intensity_map.reset()
        self._prev_uncertainty = self.ds_map.total_uncertainty()
        self.current_step = 0

        return self.get_obs()

    def step(self, actions: np.ndarray):
        self.current_step += 1
        rewards = np.zeros(self.num_drones, dtype=np.float32)
        dones   = np.zeros(self.num_drones, dtype=bool)
        info    = {
            "step"             : self.current_step,
            "shot_down"        : [],
            "terrain_hit"      : [],
            "boundary_hit"     : [],
            "drone_collision"  : [],
            "total_uncertainty": 0.0,
            "active_drones"    : int(self.drone_active.sum()),
            "mean_tr_P"        : 0.0,
            "stale_count"      : 0,
            "r_track"          : 0.0,
            "r_stale"          : 0.0,
            "r_unc"            : 0.0,
            "returning_drones" : [],
            "depleted_drones"  : [],
            "mean_battery"     : float(self.drone_battery.mean()),
        }

        # 1. 드론 이동
        for i in range(self.num_drones):
            if not self.drone_active[i]: continue
            t_hit, b_hit = self._move_drone(i, int(actions[i]))
            if t_hit: info["terrain_hit"].append(i)
            if b_hit: info["boundary_hit"].append(i)

        # 1b. 배터리 RTB
        for i in range(self.num_drones):
            if not self.drone_active[i]: continue
            if self.drone_battery[i] <= 0.0:
                self.drone_active[i]    = False
                self.drone_returning[i] = False
                info["depleted_drones"].append(i)
                continue
            if (not self.drone_returning[i]
                    and self.drone_battery[i] <= self._rtb_threshold(i)):
                self.drone_returning[i] = True
                info["returning_drones"].append(i)

        # 2. 드론 간 충돌
        for (i, j) in self._check_drone_collisions():
            rewards[i] += PENALTY_DRONE_COLLISION
            rewards[j] += PENALTY_DRONE_COLLISION
            info["drone_collision"].append((i, j))

        # 3. AntiAir
        for d_idx in self._check_antiair():
            self.drone_active[d_idx] = False
            rewards[d_idx]          += ANTIAIR_KILL_PENALTY
            dones[d_idx]             = True
            info["shot_down"].append(d_idx)

        # 4. DS 맵 업데이트
        for i in range(self.num_drones):
            if not self.drone_active[i]: continue
            self.ds_map.update_from_drone(
                drone_x               = float(self.drone_positions[i, 0]),
                drone_y               = float(self.drone_positions[i, 1]),
                target_positions      = self.target_positions,
                target_is_camouflaged = self.target_is_camouflaged,
            )

        # 5. Target Belief Update
        self.belief_tracker.predict_all()
        obs_map = {k: [] for k in range(self.num_targets)}
        for i in range(self.num_drones):
            if not self.drone_active[i]: continue
            drone_xy = self.drone_positions[i, :2]
            for k in range(self.num_targets):
                if float(np.linalg.norm(drone_xy - self.target_positions[k, :2])) < R_TRACK_M:
                    obs_map[k].append(i)

        tracking_drones = set(i for obs in obs_map.values() for i in obs)
        for i in tracking_drones:
            if self.drone_active[i]:
                self.drone_battery[i] = max(0.0, self.drone_battery[i] - BATTERY_COST_TRACK)

        for k in range(self.num_targets):
            if obs_map[k]:
                self.belief_tracker.update(
                    k, self.target_positions[k, :2],
                    bool(self.target_is_camouflaged[k]), len(obs_map[k]))
            else:
                self.belief_tracker.reset_dwell(k)

        # 6. 타겟 상태 전이
        self.belief_tracker.update_target_states()
        for k in range(self.num_targets):
            st = self.belief_tracker.target_states[k]
            self.target_found[k] = (st == TargetState.COMMITTED or st == TargetState.STALE)

        # 7. Intensity Map
        self.intensity_map.update(self.belief_tracker, self.belief_tracker.target_states)

        # 8. Active Track
        active_tracks = self.belief_tracker.get_active_tracks(self.k_active)

        # 9. Tracking Reward
        track_terms = []
        for k in range(self.num_targets):
            st = self.belief_tracker.target_states[k]
            if st not in (TargetState.COMMITTED, TargetState.STALE): continue
            if not obs_map[k]: continue
            prev_tr = float(self.belief_tracker.prev_tr_P[k])
            curr_tr = float(np.trace(self.belief_tracker.P[k]))
            delta_tr_norm = np.clip((prev_tr - curr_tr) / TR_P_MAX, 0.0, 1.0)
            if delta_tr_norm > 0:
                track_terms.append((k, delta_tr_norm))

        r_track_total = 0.0
        n_terms = max(1, len(track_terms))
        for k, delta_tr_norm in track_terms:
            r_k = LAMBDA_TRACK * (delta_tr_norm / n_terms)
            r_track_total += r_k
            share = r_k / len(obs_map[k])
            for i in obs_map[k]:
                if self.drone_active[i]:
                    rewards[i] += float(share)

        # 10. Stale Penalty
        r_stale_total = 0.0
        stale_count   = 0
        n_active = max(1, int(self.drone_active.sum()))
        for k in range(self.num_targets):
            if self.belief_tracker.target_states[k] == TargetState.STALE:
                r_stale_total -= LAMBDA_STALE
                stale_count   += 1
        if r_stale_total < 0:
            per_drone = r_stale_total / n_active
            for i in range(self.num_drones):
                if self.drone_active[i]:
                    rewards[i] += per_drone

        # 11. Reacquire Bonus
        r_reacquire_total = 0.0
        for k in range(self.num_targets):
            if (self.belief_tracker.prev_states[k] == TargetState.STALE
                    and self.belief_tracker.target_states[k] == TargetState.COMMITTED
                    and obs_map[k]):
                r_bonus = LAMBDA_REACQUIRE
                r_reacquire_total += r_bonus
                share = r_bonus / len(obs_map[k])
                for i in obs_map[k]:
                    if self.drone_active[i]:
                        rewards[i] += float(share)

        # 12. DS Uncertainty 보상
        curr_u  = self.ds_map.total_uncertainty()
        delta_u = self._prev_uncertainty - curr_u
        r_unc_total = 0.0
        if delta_u > 0:
            r_unc_total = REWARD_UNCERTAINTY_REDUCE * delta_u
            r_shared    = r_unc_total / n_active
            for i in range(self.num_drones):
                if self.drone_active[i]:
                    rewards[i] += r_shared
        self._prev_uncertainty = curr_u

        # 13. 타겟 이동 / 카모플라주 / Decay
        self._update_infantry(self.current_step)
        self._update_tanks()
        self._update_artillery(self.current_step)
        self._update_camouflage(self.current_step)
        self.ds_map.decay()

        # 14. 종료 조건
        if self.current_step >= MAX_STEPS:
            dones[:] = True
        if not self.drone_active.any():
            dones[:] = True

        # 15. info 업데이트
        mean_tr_P = float(np.mean([
            self.belief_tracker.get_tr_P_norm(k) for k in range(self.num_targets)]))
        states = self.belief_tracker.target_states
        info.update({
            "total_uncertainty": float(curr_u),
            "active_drones"    : int(self.drone_active.sum()),
            "mean_tr_P"        : mean_tr_P,
            "stale_count"      : stale_count,
            "r_track"          : r_track_total,
            "r_stale"          : r_stale_total,
            "r_unc"            : r_unc_total,
            "r_reacquire"      : r_reacquire_total,
            "tracked_targets"  : int(self.target_found.sum()),
            "committed_count"  : int(np.sum(states == TargetState.COMMITTED)),
            "stale_committed"  : int(np.sum(states == TargetState.STALE)),
            "candidate_count"  : int(np.sum(states == TargetState.CANDIDATE)),
            "active_track_ids" : active_tracks,
            "mean_battery"     : float(self.drone_battery.mean()),
            "returning_count"  : int(self.drone_returning.sum()),
            "batteries"        : self.drone_battery.tolist(),
            "artil_relocating" : len(self.artil_relo_target),
        })
        tank_idxs = [k for k,t in enumerate(self.target_types) if t == TargetType.TANK]
        if tank_idxs:
            mu_p = self.belief_tracker.imm_mu_probs[tank_idxs[0]]
            info["tank0_imm_cv"] = float(mu_p[0])
            info["tank0_imm_ct"] = float(mu_p[1])

        return self.get_obs(), np.clip(rewards, -10.0, 10.0), dones, info

    @property
    def action_space(self) -> int:
        return 4

    @property
    def obs_space(self) -> int:
        return OBS_TOTAL_DIM

    def close(self):
        pass

    def render_summary(self):
        print(f"\n── Step {self.current_step}/{MAX_STEPS} ──")
        print(f"  활성 드론 : {self.drone_active.sum()}/{self.num_drones}  "
              f"복귀 중: {self.drone_returning.sum()}")
        print(f"  불확실성  : {self.ds_map.total_uncertainty():.1f}")
        mean_trP = np.mean([self.belief_tracker.get_tr_P_norm(k)
                             for k in range(self.num_targets)])
        stale_n  = int(np.sum(self.belief_tracker.track_age > T_STALE))
        print(f"  mean_tr_P : {mean_trP:.3f}  |  stale: {stale_n}/{self.num_targets}")
        for i in range(self.num_drones):
            status = ("활성" + (" [RTB]" if self.drone_returning[i] else "")
                      if self.drone_active[i] else "비활성")
            p   = self.drone_positions[i]
            bat = self.drone_battery[i]
            print(f"  드론 {i} [{status}]: ({p[0]:.1f}, {p[1]:.1f})  배터리={bat:.3f}")


# ============================================================
# 환경 테스트
# ============================================================

def test_env():
    print("=" * 60)
    print("DroneSwarmEnv v13 (Pure NumPy) 테스트")
    print("=" * 60)

    env = DroneSwarmEnv(randomize_terrain=False, terrain_seed=42)
    print(f"\n[reset]")
    obs = env.reset()
    print(f"  obs shape : {obs.shape}")
    assert obs.shape == (NUM_DRONES, OBS_TOTAL_DIM), f"Expected {(NUM_DRONES, OBS_TOTAL_DIM)}, got {obs.shape}"

    print(f"\n[step × 5]")
    for step in range(5):
        actions = np.array([random.randint(0, 3) for _ in range(env.num_drones)])
        obs, rewards, dones, info = env.step(actions)
        print(f"  step={step+1}  rewards={np.round(rewards,2)}  "
              f"tracked={info['tracked_targets']}  U={info['total_uncertainty']:.1f}")

    env.render_summary()
    print("\n[Scale 테스트: 8 drones / 38 targets]")
    env2 = DroneSwarmEnv(num_drones=8, num_targets=38)
    obs2 = env2.reset()
    print(f"  obs shape: {obs2.shape}  (expected: (8, {OBS_TOTAL_DIM}))")
    print("\n테스트 완료 ✓")


if __name__ == "__main__":
    test_env()
