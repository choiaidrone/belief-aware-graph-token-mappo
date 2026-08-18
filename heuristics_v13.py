"""
heuristics_v13.py
=================
isaac_env_v13.py 용 휴리스틱 베이스라인 구현체.

ssm_heuristic_v13.py / ccmpso_heuristic_v13.py 는 이 파일로 통합됨.
  → from heuristics_v13 import SSMHeuristic, SSMConfig
  → from heuristics_v13 import CCMPSOHeuristic, CCMPSOConfig

두 가지 알고리즘:
  1. SSMHeuristic   — Zhang et al. (2023) "Simultaneous Search and Monitoring"
                      의 Heuristic Reactive Policy를 v13 환경에 맞게 포팅.
  2. CCMPSOHeuristic — Li et al. (2024) "CC-MPSO" 알고리즘을
                       v13 환경(4방향 이산 액션)에 맞게 포팅.

환경 인터페이스 가정 (baseline — proposed 요소 미사용):
  - env.drone_positions:       shape (num_drones, 3)
  - env.drone_active:          shape (num_drones,)   bool
  - env.drone_battery:         shape (num_drones,)   float
  - env.ds_map.U:              shape (100,100)       DS uncertainty only
  - env.ds_map._pool(arr):     -> (10,10)
  - env.target_positions:      shape (num_targets, 3)  센서 탐지용
  - env.target_is_camouflaged: shape (num_targets,) bool  [optional, 없으면 전부 False]
  - DS_MAP 좌표계: x ∈ [-20, 80], y ∈ [-50, 50]

  ※ 사용 금지 (proposed 요소):
    env.belief_tracker, env.intensity_map, get_fused_priority,
    active track obs, tr(P), fused priority map

  ※ DS uncertainty map (env.ds_map.U) 사용 기준:
    DS map 은 공통 환경 uncertainty 로 허용 (기준 A).
    SSM / CC-MPSO 모두 search planning 에 이 map 을 사용하되,
    proposed fused priority map / IMM-KF belief / active target belief 는
    일절 사용하지 않는다.
    논문 기술: "SSM and CC-MPSO baselines use the common environment-level
    uncertainty map for search planning, but do not use the proposed fused
    priority map, IMM-KF belief tracker, or active target belief."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── 공개 API (ssm_heuristic_v13 / ccmpso_heuristic_v13 대체) ────────────────
__all__ = [
    "SSMHeuristic", "SSMConfig",
    "CCMPSOHeuristic", "CCMPSOConfig",
]

# ── 환경 상수 (isaac_env_v13 와 동일값 하드코딩, import 불필요하게) ──────
DS_MAP_X_MIN = -20.0
DS_MAP_X_MAX =  80.0
DS_MAP_Y_MIN = -50.0
DS_MAP_Y_MAX =  50.0
DS_N_RL      = 10          # pooled grid 한 변 크기
CELL_W       = (DS_MAP_X_MAX - DS_MAP_X_MIN) / DS_N_RL   # 10.0 m
CELL_H       = (DS_MAP_Y_MAX - DS_MAP_Y_MIN) / DS_N_RL   # 10.0 m
DRONE_STEP_M = 3.0
R_TRACK_M    = 12.0        # 탐지 반경
TR_P_MAX     = 220.0       # 2*P_INIT_POS + 2*P_INIT_VEL

# TargetState 값 (환경 import 없이 사용)
TS_UNCOMMITTED = 0
TS_CANDIDATE   = 1
TS_COMMITTED   = 2
TS_STALE       = 3

# 4방향 액션 → delta (x, y)
ACTION_DELTA = np.array([
    [ DRONE_STEP_M,  0.0],   # 0: +x
    [-DRONE_STEP_M,  0.0],   # 1: -x
    [ 0.0,  DRONE_STEP_M],   # 2: +y
    [ 0.0, -DRONE_STEP_M],   # 3: -y
], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  공통 유틸
# ═══════════════════════════════════════════════════════════════════════════

def _best_action_toward(pos: np.ndarray, goal: np.ndarray) -> int:
    """pos → goal 방향으로 가장 가까운 4방향 액션 인덱스를 반환."""
    diff = goal - pos
    if np.linalg.norm(diff) < 1e-3:
        return np.random.randint(4)
    # 절댓값이 큰 축 우선
    if abs(diff[0]) >= abs(diff[1]):
        return 0 if diff[0] >= 0 else 1
    else:
        return 2 if diff[1] >= 0 else 3


def _cell_center(ci: int, cj: int) -> np.ndarray:
    """pooled grid 셀 (ci, cj) 의 월드 좌표 중심."""
    x = DS_MAP_X_MIN + (ci + 0.5) * CELL_W
    y = DS_MAP_Y_MIN + (cj + 0.5) * CELL_H
    return np.array([x, y], dtype=np.float32)


def _world_to_cell(wx: float, wy: float) -> Tuple[int, int]:
    """월드 좌표 → pooled 10×10 셀 인덱스 (clamp)."""
    ci = int(np.clip((wx - DS_MAP_X_MIN) / CELL_W, 0, DS_N_RL - 1))
    cj = int(np.clip((wy - DS_MAP_Y_MIN) / CELL_H, 0, DS_N_RL - 1))
    return ci, cj


def _safe_best_action_toward(env, i: int, goal: np.ndarray) -> int:
    """
    boundary + inter-UAV 안전거리를 고려한 action 선택.

    _best_action_toward() 의 safe 버전:
      1. 맵 경계 밖으로 나가는 action 은 후보에서 제외.
      2. 다른 active drone 과 5.0 m 미만으로 접근하는 action 은 제외.
      3. 위 조건을 만족하는 후보 중 goal 에 가장 가까운 action 선택.
      4. 조건을 만족하는 action 이 없으면 preferred action 그대로 반환
         (환경의 hard constraint 에 위임).
    """
    preferred = _best_action_toward(env.drone_positions[i, :2], goal)
    pos = env.drone_positions[i, :2]

    best_a, best_score = preferred, np.inf
    found = False
    # preferred 를 앞에 두어 동점이면 preferred 가 선택되도록
    for a in [preferred, 0, 1, 2, 3]:
        npos = pos + ACTION_DELTA[a]
        # ── 경계 체크 ──────────────────────────────────────────────────
        if not (DS_MAP_X_MIN <= npos[0] <= DS_MAP_X_MAX and
                DS_MAP_Y_MIN <= npos[1] <= DS_MAP_Y_MAX):
            continue
        # ── 드론 간 안전거리 체크 ──────────────────────────────────────
        too_close = False
        for j in range(env.num_drones):
            if j == i or not env.drone_active[j]:
                continue
            if float(np.linalg.norm(npos - env.drone_positions[j, :2])) < 5.0:
                too_close = True
                break
        if too_close:
            continue
        score = float(np.linalg.norm(npos - goal))
        if score < best_score:
            best_score, best_a = score, a
            found = True

    # 안전한 후보가 없으면 preferred 그대로 (환경 hard constraint 에 위임)
    return best_a


# ═══════════════════════════════════════════════════════════════════════════
#  SSMHeuristic  (Zhang et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SSMConfig:
    # confidence 이 이하이면 target lost 처리
    lose_thresh:           float = 0.25
    # 관측 없을 때 스텝당 confidence 감쇠율
    confidence_decay:      float = 0.05
    # 드론당 최대 모니터링 타겟 수
    max_monitor_per_drone: int   = 2
    rtb_battery_thresh:    float = 0.15


class SSMHeuristic:
    """
    Zhang et al. (2023) Heuristic Reactive Policy — paper-faithful 버전.

    proposed 요소 없음:
      - env.belief_tracker 접근 없음
      - fused_priority / intensity_map 없음

    자체 상태:
      _last_seen_pos[k]: 마지막 관측 위치
      _confidence[k]:    belief probability (직접 관측 → 1.0, 미관측 시 decay)
      _known_set:        currently known target IDs
    """

    def __init__(self, env=None, config: SSMConfig = SSMConfig()):
        self.env = env
        self.cfg = config
        self.n   = env.num_drones if env is not None else 4
        self._init_internal()

    def _init_internal(self):
        self._last_seen_pos: dict = {}
        self._confidence:    dict = {}
        self._known_set:     set  = set()
        self._assigned:      List[int]                  = [-1] * self.n
        self._search_goal:   List[Optional[np.ndarray]] = [None] * self.n
        self._step = 0

    # ── 외부 인터페이스 ──────────────────────────────────────────────────

    def reset(self, env=None):
        if env is not None:
            self.env = env
            self.n   = env.num_drones
        self._init_internal()

    def compute_actions(self, env=None) -> np.ndarray:
        if env is not None:
            self.env = env
        return self.act()

    def act(self) -> np.ndarray:
        self._step += 1
        self._update_belief()
        self._update_assignments()
        actions = np.zeros(self.n, dtype=np.int32)
        for i in range(self.n):
            if not self.env.drone_active[i]:
                actions[i] = 0; continue
            if self._should_rtb(i):
                actions[i] = self._rtb_action(i); continue
            k = self._assigned[i]
            actions[i] = self._monitor_action(i, k) if k >= 0 else self._search_action(i)
        return actions

    # ── 내부 belief 갱신 ─────────────────────────────────────────────────

    def _update_belief(self):
        """
        센서 범위(R_TRACK_M) 내 직접 감지로만 belief 갱신.
        env.detection_events 있으면 사용, 없으면 target_positions 거리 체크.
        """
        env = self.env
        detected = set()

        ev_list = getattr(env, 'detection_events', None)
        if ev_list is not None:
            for ev in ev_list:
                if len(ev) >= 3:
                    k   = int(ev[1])
                    pos = np.asarray(ev[2], dtype=np.float32)[:2]
                    self._last_seen_pos[k] = pos
                    self._confidence[k]    = 1.0
                    self._known_set.add(k)
                    detected.add(k)
        else:
            t_pos = getattr(env, 'target_positions', None)
            if t_pos is not None:
                # camouflage 체크: proposed target_is_camouflaged 가 있으면 사용,
                # 없으면 전부 False (AM/GAT wrapper 와 동일 기준 적용)
                camou = getattr(env, 'target_is_camouflaged',
                                np.zeros(t_pos.shape[0], dtype=bool))
                for i in range(self.n):
                    if not env.drone_active[i]: continue
                    dp = env.drone_positions[i, :2]
                    for k in range(t_pos.shape[0]):
                        if camou[k]:          # 위장 타겟은 탐지 불가
                            continue
                        if float(np.linalg.norm(dp - t_pos[k, :2])) < R_TRACK_M:
                            self._last_seen_pos[k] = t_pos[k, :2].copy()
                            self._confidence[k]    = 1.0
                            self._known_set.add(k)
                            detected.add(k)

        # confidence decay for undetected known targets
        to_lose = []
        for k in list(self._known_set):
            if k not in detected:
                self._confidence[k] = max(
                    0.0, self._confidence.get(k, 0.0) - self.cfg.confidence_decay)
                if self._confidence[k] < self.cfg.lose_thresh:
                    to_lose.append(k)
        for k in to_lose:
            self._known_set.discard(k)
            for i in range(self.n):
                if self._assigned[i] == k:
                    self._assigned[i] = -1

    def _update_assignments(self):
        covered = set(k for k in self._assigned if k >= 0)
        uncovered = sorted(
            [k for k in self._known_set if k not in covered],
            key=lambda k: self._confidence.get(k, 0.0)
        )
        for k in uncovered:
            if sum(1 for a in self._assigned if a == k) >= self.cfg.max_monitor_per_drone:
                continue
            tpos = self._last_seen_pos.get(k)
            if tpos is None: continue
            best_i, best_d = -1, np.inf
            for i in range(self.n):
                if not self.env.drone_active[i]: continue
                if self._should_rtb(i): continue
                if self._assigned[i] >= 0: continue
                d = float(np.linalg.norm(self.env.drone_positions[i, :2] - tpos))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                self._assigned[best_i] = k
                covered.add(k)

    # ── action helpers ────────────────────────────────────────────────────

    def _should_rtb(self, i: int) -> bool:
        return float(self.env.drone_battery[i]) < self.cfg.rtb_battery_thresh

    def _rtb_action(self, i: int) -> int:
        base = np.asarray(self.env.base_positions[i], dtype=np.float32)[:2]
        return _safe_best_action_toward(self.env, i, base)

    def _monitor_action(self, i: int, k: int) -> int:
        tpos = self._last_seen_pos.get(k, self.env.drone_positions[i, :2])
        return _safe_best_action_toward(self.env, i, tpos)

    def _search_action(self, i: int) -> int:
        pos  = self.env.drone_positions[i, :2]
        goal = self._search_goal[i]
        if goal is None or np.linalg.norm(pos - goal) < DRONE_STEP_M:
            goal = self._pick_search_cell(i)
            self._search_goal[i] = goal
        return _safe_best_action_toward(self.env, i, goal)

    def _pick_search_cell(self, drone_i: int) -> np.ndarray:
        """
        DS uncertainty map 만 사용 (proposed fused_priority / intensity_map 없음).
        DS map 은 공통 환경 uncertainty 로 허용 — 논문 기준 A.
        타 드론의 현재 목표 셀은 priority 를 절반으로 감쇄하여 분산 탐색 유도.
        """
        priority = self.env.ds_map._pool(self.env.ds_map.U).copy()
        for j, g in enumerate(self._search_goal):
            if j == drone_i or g is None: continue
            ci, cj = _world_to_cell(float(g[0]), float(g[1]))
            priority[ci, cj] *= 0.5
        ci, cj = divmod(int(np.argmax(priority)), DS_N_RL)
        return _cell_center(ci, cj)


@dataclass
class CCMPSOConfig:
    n_particles:  int   = 20
    max_iters:    int   = 30
    horizon:      int   = 8
    w:            float = 0.6
    c1:           float = 1.8
    c2:           float = 1.8
    v_max:        float = 1.5
    conv_eps:     float = 0.2
    rtb_battery_thresh: float = 0.15
    # 자체 Bayesian prob map 파라미터
    visit_decay:  float = 0.5   # 방문 후 prob 감소 비율


class CCMPSOHeuristic:
    """
    Li et al. (2024) CC-MPSO — paper-faithful 버전.

    proposed 요소 없음:
      - env.belief_tracker 접근 없음
      - fused_priority / intensity_map 없음

    자체 상태: _prob_map (10×10) — DS uncertainty + 방문 횟수 기반 Bayesian map.
    Fitness: _prob_map 위에서 누적 탐지 확률 (논문 식 14/15).

    CC 프레임워크:
      - UAV i 전용 swarm i 가 각자 독립 최적화.
      - fitness 계산 시 나머지 swarm 들의 global best와 결합 (중복 페널티).
    """

    # 4방향 angle → delta
    _DIRS = np.array([
        [ DRONE_STEP_M,  0.0],   # 0° → +x
        [ 0.0,  DRONE_STEP_M],   # 90° → +y
        [-DRONE_STEP_M,  0.0],   # 180° → -x
        [ 0.0, -DRONE_STEP_M],   # 270° → -y
    ], dtype=np.float32)
    # angle (rad) → 가장 가까운 4방향 인덱스
    _DIR_ANGLES = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])

    def __init__(self, env=None, config: CCMPSOConfig = CCMPSOConfig()):
        self.env = env
        self.cfg = config
        self.n   = env.num_drones if env is not None else 4

        # swarm 별 상태 초기화는 reset() 에서
        self._swarm_pos: List[np.ndarray]  = []   # (N, T) continuous angle
        self._swarm_vel: List[np.ndarray]  = []
        self._pbest:     List[np.ndarray]  = []   # personal best
        self._gbest:     List[np.ndarray]  = []   # global best per swarm
        self._gbest_fit: List[float]       = []

        self.reset()

    # ── 외부 인터페이스 ──────────────────────────────────────────────────

    def reset(self, env=None):
        if env is not None:
            self.env = env
            self.n   = env.num_drones
        cfg = self.cfg
        N, T = cfg.n_particles, cfg.horizon
        self._swarm_pos  = []
        self._swarm_vel  = []
        self._pbest      = []
        self._gbest      = []
        self._gbest_fit  = []

        # 자체 Bayesian target probability map
        self._prob_map    = np.ones((DS_N_RL, DS_N_RL), dtype=np.float32) / (DS_N_RL * DS_N_RL)
        self._visit_count = np.zeros((DS_N_RL, DS_N_RL), dtype=np.float32)

        for i in range(self.n):
            pos = np.random.uniform(0, 2*np.pi, size=(N, T)).astype(np.float64)
            vel = np.random.uniform(-1.0, 1.0, size=(N, T)).astype(np.float64)
            self._swarm_pos.append(pos)
            self._swarm_vel.append(vel)
            self._pbest.append(pos.copy())
            self._gbest.append(pos[0].copy())
            self._gbest_fit.append(-np.inf)

    def compute_actions(self, env=None) -> np.ndarray:
        """eval 인터페이스용 래퍼. act() 와 동일."""
        if env is not None:
            self.env = env
        return self.act()

    def act(self) -> np.ndarray:
        """매 스텝 PSO 최적화 후 각 드론의 첫 번째 액션 반환."""
        self._optimize()
        actions = np.zeros(self.n, dtype=np.int32)
        for i in range(self.n):
            if not self.env.drone_active[i]:
                actions[i] = 0
                continue
            if self.env.drone_battery[i] < self.cfg.rtb_battery_thresh:
                actions[i] = _safe_best_action_toward(
                    self.env, i,
                    np.asarray(self.env.base_positions[i], dtype=np.float32)[:2])
                continue
            # PSO gbest 의 첫 번째 action → safe 보정
            # PSO 는 연속 angle 공간에서 최적화하므로 boundary / 드론 간
            # 안전거리를 직접 고려하지 않음.
            # gbest 첫 step 의 목표 위치를 goal 로 변환 후
            # _safe_best_action_toward 로 보정.
            angle = float(self._gbest[i][0])
            a     = self._angle_to_action(angle)
            pos   = self.env.drone_positions[i, :2]
            goal  = pos + ACTION_DELTA[a]
            actions[i] = _safe_best_action_toward(self.env, i, goal)
        return actions

    # ── PSO 최적화 ───────────────────────────────────────────────────────

    def _optimize(self):
        cfg = self.cfg
        N, T = cfg.n_particles, cfg.horizon
        env  = self.env

        # 자체 Bayesian prob map 갱신 후 priority로 사용
        self._update_prob_map()
        priority = self._prob_map

        for it in range(cfg.max_iters):
            for k in range(self.n):
                if not env.drone_active[k]: continue

                # stagnation 확인 → reset
                if it > 0 and self._check_stagnation(k):
                    self._reset_swarm(k)

                # gbest of other swarms — (drone_id, path) 튜플로 전달
                other_paths = [(j, self._gbest[j]) for j in range(self.n) if j != k]

                N_k = len(self._swarm_pos[k])
                fit_vals = np.empty(N_k, dtype=np.float64)
                for p_idx in range(N_k):
                    path = self._swarm_pos[k][p_idx]
                    fit_vals[p_idx] = self._fitness(k, path, other_paths, priority)

                # personal best update
                for p_idx in range(N_k):
                    old_fit = self._fitness(k, self._pbest[k][p_idx], other_paths, priority)
                    if fit_vals[p_idx] > old_fit:
                        self._pbest[k][p_idx] = self._swarm_pos[k][p_idx].copy()

                # global best update
                best_idx = int(np.argmax(fit_vals))
                if fit_vals[best_idx] > self._gbest_fit[k]:
                    self._gbest_fit[k] = float(fit_vals[best_idx])
                    self._gbest[k]     = self._swarm_pos[k][best_idx].copy()

                # velocity & position update (standard PSO)
                r1 = np.random.rand(N_k, T)
                r2 = np.random.rand(N_k, T)
                self._swarm_vel[k] = (
                    cfg.w  * self._swarm_vel[k]
                    + cfg.c1 * r1 * (self._pbest[k] - self._swarm_pos[k])
                    + cfg.c2 * r2 * (self._gbest[k] - self._swarm_pos[k])
                )
                # velocity clamping
                self._swarm_vel[k] = np.clip(
                    self._swarm_vel[k], -cfg.v_max, cfg.v_max)
                self._swarm_pos[k] = self._swarm_pos[k] + self._swarm_vel[k]
                # angle wrap to [0, 2π)
                self._swarm_pos[k] = self._swarm_pos[k] % (2 * np.pi)

    def _fitness(self,
                 drone_idx: int,
                 path: np.ndarray,
                 other_paths: List[Tuple[int, np.ndarray]],
                 priority: np.ndarray) -> float:
        """
        경로(path: T개 연속 angle)를 따라가며
        자체 Bayesian probability map 위에서 누적 탐지 확률을 계산.

        F = 1 - Π(1 - p_cell_t)   (논문 식 14/15 의 근사)

        other_paths 가 방문한 셀은 이미 커버됐다고 보고 패널티.
        """
        T   = self.cfg.horizon
        pos = self.env.drone_positions[drone_idx, :2].copy()

        # other drones 가 방문하는 셀 집합 — 각 드론의 실제 현재 위치에서 출발
        other_cells: set = set()
        for j, op in other_paths:
            op_pos = self.env.drone_positions[j, :2].copy()
            for t in range(min(T, len(op))):
                d_idx = self._angle_to_dir(float(op[t]))
                op_pos = op_pos + self._DIRS[d_idx]
                ci, cj = _world_to_cell(float(op_pos[0]), float(op_pos[1]))
                other_cells.add((ci, cj))
        log_no_detect = 0.0   # log(1 - F)
        for t in range(T):
            d_idx = self._angle_to_dir(float(path[t]))
            pos   = pos + self._DIRS[d_idx]
            # boundary clamp (맵 밖 이동 페널티)
            if not (DS_MAP_X_MIN <= pos[0] <= DS_MAP_X_MAX and
                    DS_MAP_Y_MIN <= pos[1] <= DS_MAP_Y_MAX):
                log_no_detect += np.log(0.99)  # 작은 페널티
                continue
            ci, cj = _world_to_cell(float(pos[0]), float(pos[1]))
            p_cell = float(priority[ci, cj])
            # 이미 다른 드론이 커버 → p 절반으로
            if (ci, cj) in other_cells:
                p_cell *= 0.5
            # 수치 안정성
            p_cell = np.clip(p_cell, 1e-6, 1.0 - 1e-6)
            log_no_detect += np.log(1.0 - p_cell)

        return float(1.0 - np.exp(log_no_detect))

    def _update_prob_map(self):
        """
        자체 Bayesian target probability map 갱신.

        DS uncertainty (env.ds_map.U) × 방문 횟수 역비례.
        DS map 은 공통 환경 uncertainty 로 허용 (논문 기준 A).
        proposed fused priority map / IMM-KF / active track 은 사용하지 않음.
        """
        env = self.env
        U_pool = env.ds_map._pool(env.ds_map.U).astype(np.float32)

        for i in range(self.n):
            if not env.drone_active[i]: continue
            ci, cj = _world_to_cell(
                float(env.drone_positions[i, 0]),
                float(env.drone_positions[i, 1]))
            self._visit_count[ci, cj] += 1.0

        visit_norm   = self._visit_count / (self._visit_count.max() + 1e-6)
        self._prob_map = U_pool * (1.0 - self.cfg.visit_decay * visit_norm)
        total = self._prob_map.sum()
        if total > 1e-8:
            self._prob_map /= total
        else:
            self._prob_map[:] = 1.0 / (DS_N_RL * DS_N_RL)

    def _check_stagnation(self, k: int) -> bool:
        """논문 식 (29)(30): 입자들이 gbest 근방에 몰렸는지 확인."""
        pos   = self._swarm_pos[k]      # (N, T)
        gbest = self._gbest[k]           # (T,)
        # Manhattan distance in angle space (wrap 고려)
        diff     = np.abs(pos - gbest)
        diff     = np.minimum(diff, 2*np.pi - diff)  # wrap
        dist_sum = float(diff.sum(axis=1).mean())    # mean over particles
        # 정규화: 초기 최대 dist ~ T * π
        norm = dist_sum / (self.cfg.horizon * np.pi + 1e-8)
        return norm < self.cfg.conv_eps

    def _reset_swarm(self, k: int):
        """Stagnation 시 swarm k 의 입자를 랜덤 재초기화 (gbest 유지)."""
        N, T = self.cfg.n_particles, self.cfg.horizon
        self._swarm_pos[k] = np.random.uniform(0, 2*np.pi, size=(N, T))
        self._swarm_vel[k] = np.random.uniform(-1.0, 1.0, size=(N, T))
        self._pbest[k]     = self._swarm_pos[k].copy()
        # gbest 는 최소한 1개 입자로 씨드
        self._swarm_pos[k][0] = self._gbest[k].copy()

    # ── 방향 변환 ────────────────────────────────────────────────────────

    @staticmethod
    def _angle_to_dir(angle: float) -> int:
        """연속 angle(rad) → 가장 가까운 4방향 인덱스 (0~3)."""
        angle = angle % (2 * np.pi)
        diffs = np.abs(CCMPSOHeuristic._DIR_ANGLES - angle)
        # wrap 처리
        diffs = np.minimum(diffs, 2*np.pi - diffs)
        return int(np.argmin(diffs))

    @staticmethod
    def _angle_to_action(angle: float) -> int:
        """
        angle → v13 환경 액션 (0=+x, 1=-x, 2=+y, 3=-y).
        내부 dir 인덱스와 환경 액션 인덱스가 다름에 주의.
        dir: 0=+x(0°), 1=+y(90°), 2=-x(180°), 3=-y(270°)
        env: 0=+x, 1=-x, 2=+y, 3=-y
        """
        DIR_TO_ACTION = [0, 2, 1, 3]
        d = CCMPSOHeuristic._angle_to_dir(angle)
        return DIR_TO_ACTION[d]


# ═══════════════════════════════════════════════════════════════════════════
#  간단한 동작 확인용 smoke test
# ═══════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """
    isaac_env_v13 가 있으면 실제 환경으로, 없으면 mock 으로 테스트.
    """
    try:
        import sys
        sys.path.insert(0, "/mnt/user-data/uploads")
        from isaac_env_v13 import DroneSwarmEnv
        env = DroneSwarmEnv(randomize_terrain=False, terrain_seed=42)
        obs = env.reset()
        print(f"[smoke] env reset OK, obs shape = {obs.shape}")
    except Exception as e:
        print(f"[smoke] env load failed ({e}), using mock")
        env = _MockEnv()
        obs = env.reset()

    # SSM
    ssm = SSMHeuristic(env, SSMConfig())
    ssm.reset()
    for step in range(5):
        actions = ssm.act()
        obs, r, done, info = env.step(actions)
        print(f"[SSM  ] step {step+1}  actions={actions}  r={r.round(3)}")
        if done.any(): break

    env.reset()

    # CC-MPSO
    pso = CCMPSOHeuristic(env, CCMPSOConfig(n_particles=10, max_iters=10, horizon=5))
    pso.reset()
    for step in range(5):
        actions = pso.act()
        obs, r, done, info = env.step(actions)
        print(f"[CCPSO] step {step+1}  actions={actions}  r={r.round(3)}")
        if done.any(): break

    print("[smoke] DONE")


class _MockEnv:
    """실제 환경 없을 때 쓰는 최소 mock."""
    num_drones   = 4
    num_targets  = 20
    k_active     = 12

    def __init__(self):
        self.drone_positions        = np.zeros((4, 3), dtype=np.float32)
        self.drone_active           = np.ones(4, dtype=bool)
        self.drone_battery          = np.ones(4, dtype=np.float32)
        self.drone_returning        = np.zeros(4, dtype=bool)
        self.base_positions         = np.zeros((4, 2), dtype=np.float32)
        self.belief_tracker         = _MockTracker(20)
        # camouflage mock: 일부 타겟을 위장 상태로 설정
        self.target_is_camouflaged  = np.zeros(20, dtype=bool)
        self.target_is_camouflaged[::5] = True   # 0, 5, 10, 15번 타겟 위장

        class _DS:
            U = np.ones((100, 100), dtype=np.float32) * 0.5
            def _pool(self, arr):
                n = 10
                r = np.zeros((10,10), dtype=np.float32)
                for i in range(10):
                    for j in range(10):
                        r[i,j] = arr[i*n:(i+1)*n, j*n:(j+1)*n].mean()
                return r

        class _IM:
            grid = np.random.rand(10, 10).astype(np.float32)
            def get_fused_priority(self, U_pool, beta=0.5):
                return (U_pool * (1 + beta * self.grid) / (1 + beta)).flatten()

        self.ds_map       = _DS()
        self.intensity_map = _IM()

    def reset(self): return np.zeros((4, 484), dtype=np.float32)

    def step(self, actions):
        r   = np.zeros(4, dtype=np.float32)
        d   = np.zeros(4, dtype=bool)
        inf = {"step": 0}
        return np.zeros((4, 484), dtype=np.float32), r, d, inf


class _MockTracker:
    def __init__(self, n):
        self.num_targets  = n
        self.target_states = np.zeros(n, dtype=np.int32)
        self.mu            = np.zeros((n, 4), dtype=np.float64)
        # 몇 개는 COMMITTED
        self.target_states[:4] = TS_COMMITTED
        self.mu[:4, 0] = [20, 30, 40, 50]
        self.mu[:4, 1] = [0, 10, -10, 5]

    def get_tr_P_norm(self, k): return float(np.random.rand())
    def get_active_tracks(self, k): return list(range(min(k, self.num_targets)))


if __name__ == "__main__":
    _smoke_test()
