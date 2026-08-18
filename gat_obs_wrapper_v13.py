"""
gat_obs_wrapper_v13.py
======================
isaac_env_v13 위에 얹는 GAT 논문식 observation wrapper.

논문(Zhao et al. 2025)에서 각 드론은 FOV 내에서 직접 관측한
Agent / Target / Obstacle 노드만 볼 수 있다.

제거 (proposed 요소):
  - DS E/F/U map
  - fused priority map
  - IMM-EKF belief_tracker
  - active track slot
  - target intensity map

사용 (논문 허용):
  - 드론 자신의 위치, 배터리
  - FOV 내 다른 드론의 상대 위치
  - FOV 내 직접 탐지된 타겟의 상대 위치
  - FOV 내 지형 위험 셀(obstacle proxy)의 상대 위치

Observation shape:
  각 드론별 고정 크기 flat vector:
    self_feat:      5  (self_x_norm, self_y_norm, battery, returning, step_norm)
    neighbor_feat:  MAX_NEIGHBOR_DRONE × 4  (rel_x, rel_y, dist_norm, active)
    target_feat:    MAX_OBS_TARGET    × 4  (rel_x, rel_y, dist_norm, valid)
    obstacle_feat:  MAX_OBSTACLE      × 3  (rel_x, rel_y, dist_norm)

    총 = 5 + 11×4 + 8×4 + 16×3 = 5 + 44 + 32 + 48 = 129 차원

사용법:
    from gat_obs_wrapper_v13 import GATObsWrapper

    base_env = DroneSwarmEnv(...)
    env = GATObsWrapper(base_env)
    obs = env.reset()          # shape (num_drones, GAT_OBS_DIM)
    obs, r, done, info = env.step(actions)
"""

import numpy as np
from typing import Optional

# ── 상수 ──────────────────────────────────────────────────────────────
MAX_NEIGHBOR_DRONE = 11    # 최대 이웃 드론 수 (num_drones - 1 = 최대 11)
MAX_OBS_TARGET     = 8     # FOV 내 타겟 슬롯
MAX_OBSTACLE       = 16    # FOV 내 지형 위험 슬롯

SELF_DIM     = 5
NEIGHBOR_DIM = MAX_NEIGHBOR_DRONE * 4
TARGET_DIM   = MAX_OBS_TARGET    * 4
OBSTACLE_DIM = MAX_OBSTACLE      * 3
GAT_OBS_DIM  = SELF_DIM + NEIGHBOR_DIM + TARGET_DIM + OBSTACLE_DIM   # 129

# 맵 정규화용 상수
MAP_X_MIN = -20.0; MAP_X_MAX = 80.0
MAP_Y_MIN = -50.0; MAP_Y_MAX = 50.0
MAP_W     = MAP_X_MAX - MAP_X_MIN   # 100
MAP_H     = MAP_Y_MAX - MAP_Y_MIN   # 100

# 환경 상수 (isaac_env_v13 동일)
R_TRACK_M  = 12.0    # 드론 센서 탐지 반경 (m)
FOV_RADIUS = R_TRACK_M   # agent / obstacle FOV도 같은 반경
DRONE_STEP = 3.0


def _norm_x(x): return (x - MAP_X_MIN) / MAP_W
def _norm_y(y): return (y - MAP_Y_MIN) / MAP_H
def _norm_d(d): return float(d) / FOV_RADIUS   # [0,1] — FOV 반경 기준


class GATObsWrapper:
    """
    isaac_env_v13 DroneSwarmEnv를 감싸는 GAT 논문식 obs wrapper.

    reset() / step() / 주요 속성은 base_env에 위임.
    obs만 GAT graph obs로 교체.
    """

    def __init__(self, base_env, max_ep_steps: int = 3000):
        self.env          = base_env
        self.max_ep_steps = max_ep_steps
        self._step_count  = 0

        # base_env 속성 위임
        self.num_drones  = base_env.num_drones
        self.num_targets = base_env.num_targets

        # 지형 obstacle 위치 캐시 (reset 후 갱신)
        self._obstacle_positions: Optional[np.ndarray] = None  # (N_obs, 2)

    # ── 속성 위임 ────────────────────────────────────────────────────

    @property
    def drone_positions(self):  return self.env.drone_positions
    @property
    def drone_active(self):     return self.env.drone_active
    @property
    def drone_battery(self):    return self.env.drone_battery
    @property
    def base_positions(self):   return self.env.base_positions
    @property
    def target_positions(self): return self.env.target_positions
    @property
    def belief_tracker(self):   return getattr(self.env, 'belief_tracker', None)
    @property
    def ds_map(self):           return getattr(self.env, 'ds_map', None)

    # ── reset / step ─────────────────────────────────────────────────

    def reset(self):
        obs = self.env.reset()
        self._step_count = 0
        self._build_obstacle_cache()
        return self._build_gat_obs()

    def step(self, actions):
        self._step_count += 1
        _, rewards, dones, info = self.env.step(actions)
        return self._build_gat_obs(), rewards, dones, info

    # ── obstacle 위치 캐시 ───────────────────────────────────────────

    def _build_obstacle_cache(self):
        """
        지형 위험 셀 중심 좌표를 미리 추출.
        DroneSwarmEnv.heightmap 기반으로 지형 블록 위치 파악.
        """
        env = self.env
        positions = []

        if hasattr(env, 'heightmap') and env.heightmap is not None:
            hm = env.heightmap
            ox = env.GRID_ORIGIN_X
            oy = env.GRID_ORIGIN_Y
            # 높이 > 0.5인 셀을 obstacle로 간주 (5칸 간격으로 샘플링해서 최대 수 제한)
            step = 3
            for xi in range(0, hm.shape[0], step):
                for yi in range(0, hm.shape[1], step):
                    if hm[xi, yi] > 0.5:
                        wx = float(xi) + ox
                        wy = float(yi) + oy
                        if MAP_X_MIN <= wx <= MAP_X_MAX and MAP_Y_MIN <= wy <= MAP_Y_MAX:
                            positions.append([wx, wy])

        if positions:
            self._obstacle_positions = np.array(positions, dtype=np.float32)
        else:
            self._obstacle_positions = np.zeros((0, 2), dtype=np.float32)

    # ── GAT obs 구성 ─────────────────────────────────────────────────

    def _build_gat_obs(self) -> np.ndarray:
        """
        Returns: shape (num_drones, GAT_OBS_DIM)
        """
        env  = self.env
        N    = env.num_drones
        step_norm = min(self._step_count / max(self.max_ep_steps, 1), 1.0)
        obs  = np.zeros((N, GAT_OBS_DIM), dtype=np.float32)

        for i in range(N):
            obs[i] = self._build_drone_obs(i, step_norm)

        return obs

    def _build_drone_obs(self, i: int, step_norm: float) -> np.ndarray:
        env   = self.env
        pos_i = env.drone_positions[i, :2]
        vec   = np.zeros(GAT_OBS_DIM, dtype=np.float32)
        ptr   = 0

        # ── 1. self feature (5차원) ──────────────────────────────────
        vec[ptr]   = _norm_x(pos_i[0])
        vec[ptr+1] = _norm_y(pos_i[1])
        vec[ptr+2] = float(env.drone_battery[i])
        vec[ptr+3] = float(getattr(env, 'drone_returning', np.zeros(env.num_drones))[i])
        vec[ptr+4] = step_norm
        ptr += SELF_DIM

        # ── 2. neighbor drone features (MAX_NEIGHBOR_DRONE × 4) ─────
        slot = 0
        for j in range(env.num_drones):
            if j == i: continue                  # 자기 자신 건너뜀
            if slot >= MAX_NEIGHBOR_DRONE: break  # 슬롯 꽉 찼을 때만 종료
            if not env.drone_active[j]: continue
            pos_j = env.drone_positions[j, :2]
            diff  = pos_j - pos_i
            dist  = float(np.linalg.norm(diff))
            if dist > FOV_RADIUS: continue
            base = ptr + slot * 4
            vec[base]   = diff[0] / FOV_RADIUS
            vec[base+1] = diff[1] / FOV_RADIUS
            vec[base+2] = _norm_d(dist)
            vec[base+3] = 1.0   # valid flag
            slot += 1
        ptr += NEIGHBOR_DIM

        # ── 3. observed target features (MAX_OBS_TARGET × 4) ────────
        # 직접 탐지 — IMM-EKF 없음, FOV 내 ground-level 위치만
        slot = 0
        # 거리 순 정렬해서 가까운 것부터
        t_pos = env.target_positions[:, :2]
        dists = np.linalg.norm(t_pos - pos_i, axis=1)
        order = np.argsort(dists)
        camouflage = getattr(env, 'target_is_camouflaged',
                             np.zeros(env.num_targets, dtype=bool))
        for k in order:
            if slot >= MAX_OBS_TARGET: break
            if dists[k] > R_TRACK_M: continue
            if camouflage[k]: continue           # camouflaged → 직접 감지 불가
            diff = t_pos[k] - pos_i
            base = ptr + slot * 4
            vec[base]   = diff[0] / R_TRACK_M
            vec[base+1] = diff[1] / R_TRACK_M
            vec[base+2] = _norm_d(dists[k])
            vec[base+3] = 1.0   # valid flag
            slot += 1
        ptr += TARGET_DIM

        # ── 4. obstacle (terrain) features (MAX_OBSTACLE × 3) ───────
        if self._obstacle_positions is not None and len(self._obstacle_positions) > 0:
            o_pos  = self._obstacle_positions
            o_diff = o_pos - pos_i
            o_dist = np.linalg.norm(o_diff, axis=1)
            in_fov = o_dist <= FOV_RADIUS
            idx    = np.where(in_fov)[0]
            idx    = idx[np.argsort(o_dist[idx])][:MAX_OBSTACLE]
            for si, k in enumerate(idx):
                base = ptr + si * 3
                vec[base]   = o_diff[k, 0] / FOV_RADIUS
                vec[base+1] = o_diff[k, 1] / FOV_RADIUS
                vec[base+2] = _norm_d(o_dist[k])

        return vec
