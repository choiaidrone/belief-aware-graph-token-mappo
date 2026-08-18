"""
graph_token_mappo_v13.py
========================
Graph-Token Belief MAPPO — proposed 확장 모델.

구조:
  Token branch  (기존 token_ppo_v3 그대로):
    DS map / fused priority / IMM-EKF active track / global token
    → Entity Transformer (RelationBiasV4)

  Graph branch  (신규, PaperGATBackbone):
    FOV 내 neighbor drone / 직접 관측 target / terrain obstacle
    → 거리 기반 GAT aggregation → graph_drone_h

  Fusion:
    drone_token = graph_fuse(cat[token_drone_h, graph_drone_h])
    → 기존 Entity Transformer 통과

  Critic fusion:
    token global_h + graph global_h → critic_fuse → value

비교표:
  AM-MAPPO-paper     : DS X  fused X  IMM-EKF X  graph X  token X
  GAT-MAPPO-paper    : DS X  fused X  IMM-EKF X  graph O  token X
  Token-MAPPO        : DS O  fused O  IMM-EKF O  graph X  token O
  Graph-Token-MAPPO  : DS O  fused O  IMM-EKF O  graph O  token O  ← 이 파일

주요 수정:
  1. GraphTokenObsWrapper — dual obs (obs484 + gobs129) 생성
  2. RolloutBuffer        — obs484 / gobs 둘 다 저장
  3. EntityMAPPOActor     — graph_fuse(drone token + GAT drone_h)
  4. EntityMAPPOCritic    — critic_fuse(token global + GAT global)
  5. MAPPOAgent           — get_action_and_value / update 수정

거의 그대로 유지:
  TokenBuilder / RelationBiasV4 / EntityTransformerEncoder
  PPO loss / GAE / reward / logger / checkpoint
"""
import os, sys, time
from pathlib import Path
import numpy as np

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque
from tqdm import tqdm

# ── Graph branch: 외부 파일 없이 인라인 정의 ─────────────────────────
# (gat_obs_wrapper_v13 / gat_mappo_paper_v13 import 불필요)

# GAT obs 상수
MAX_NEIGHBOR_DRONE = 11
MAX_OBS_TARGET     = 8
MAX_OBSTACLE       = 16

SELF_DIM     = 5
NEIGHBOR_DIM = MAX_NEIGHBOR_DRONE * 4   # 44
TARGET_DIM   = MAX_OBS_TARGET    * 4   # 32
OBSTACLE_DIM = MAX_OBSTACLE      * 3   # 48
GAT_OBS_DIM  = SELF_DIM + NEIGHBOR_DIM + TARGET_DIM + OBSTACLE_DIM  # 129

_GAT_MAP_X_MIN = -20.0; _GAT_MAP_X_MAX = 80.0
_GAT_MAP_Y_MIN = -50.0; _GAT_MAP_Y_MAX = 50.0
_GAT_FOV_RADIUS = 12.0
_GAT_DRONE_STEP = 3.0

D_SELF_GAT     = SELF_DIM
D_NEIGHBOR_GAT = 4
D_TARGET_GAT   = 4
D_OBSTACLE_GAT = 3


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ============================================================
# 환경 상수 (isaac_env_v11.py 동기화)
# ============================================================
N_DRONE_TRAIN = 4    # 학습 시 드론 수
N_DRONE_MAX   = 12   # 지원 최대 드론 수 (eval scale-up용)
N_DRONE       = N_DRONE_TRAIN   # 하위 호환용 alias
K_ACTIVE   = 8       # K-slot sensitivity variant (base=12)
N_ZONE     = 16
N_GLOBAL   = 1

NUM_DRONES   = N_DRONE_TRAIN
# K-slot sensitivity: OBS_DIM = 300+8+2+100+K*6+2
# K=8 → DS(300)+depth(8)+self(2)+fused_priority(100)+active(48)+battery(2) = 460
OBS_DIM      = 460
ACTION_DIM   = 4
DS_N_RL      = 10
MAP_W        = 100.0
MAP_H        = 100.0

# obs 슬라이스 (v12: other_vec 제거)
OBS_E_START         = 0
OBS_F_START         = 100
OBS_U_START         = 200
OBS_DEPTH_START     = 300
OBS_SELF_START      = 308
# other_vec 슬롯 제거 — Entity Transformer attention이 드론 간 정보 담당
OBS_INTENSITY_START = 310           # 100dim fused priority map (v12: 316→310)
OBS_ACTIVE_START    = 410           # active track 시작 (K와 무관, 고정)
OBS_BATTERY_START   = 458           # K*6=48dim active 다음 (410+48)

# active track 구성
ACTIVE_FEAT_DIM = 6    # dx,dy,vx,vy,trP_norm,age_norm

# 토큰 구성 — 드론 수 가변 (학습=4, eval=8/12 지원)
# N_TOKENS는 실제 드론 수에 따라 동적으로 계산
N_TOKENS_TRAIN = N_DRONE_TRAIN + N_ZONE + K_ACTIVE + N_GLOBAL  # 33 (학습 시)
TOKEN_DIM = 128

def get_token_indices(n_drone: int):
    """실제 드론 수 기준 토큰 인덱스 반환."""
    idx_drone_start  = 0
    idx_drone_end    = n_drone
    idx_zone_start   = n_drone
    idx_zone_end     = n_drone + N_ZONE
    idx_active_start = n_drone + N_ZONE
    idx_active_end   = n_drone + N_ZONE + K_ACTIVE
    idx_global       = n_drone + N_ZONE + K_ACTIVE  # last token
    n_tokens         = n_drone + N_ZONE + K_ACTIVE + N_GLOBAL
    return (idx_drone_start, idx_drone_end,
            idx_zone_start, idx_zone_end,
            idx_active_start, idx_active_end,
            idx_global, n_tokens)

# 학습 시 기본값 (하위 호환)
(IDX_DRONE_START, IDX_DRONE_END,
 IDX_ZONE_START,  IDX_ZONE_END,
 IDX_ACTIVE_START,IDX_ACTIVE_END,
 IDX_GLOBAL, N_TOKENS) = get_token_indices(N_DRONE_TRAIN)

# Zone 16개 partition (10×10 grid 기준)
_SX = [(0,2),(2,5),(5,8),(8,10)]
_SY = [(0,2),(2,5),(5,8),(8,10)]
ZONE_TABLE: List[Tuple] = [(_SX[xi], _SY[yi]) for xi in range(4) for yi in range(4)]

def _zone_centers_norm() -> torch.Tensor:
    DS_MAP_X_MIN = -20.0; DS_MAP_Y_MIN = -50.0
    cell_w = MAP_W / DS_N_RL; cell_h = MAP_H / DS_N_RL
    centers = []
    for (x0,x1),(y0,y1) in ZONE_TABLE:
        world_cx = DS_MAP_X_MIN + (x0+x1)/2.0*cell_w
        world_cy = DS_MAP_Y_MIN + (y0+y1)/2.0*cell_h
        cx = (world_cx - DS_MAP_X_MIN) / MAP_W
        cy = (world_cy - DS_MAP_Y_MIN) / MAP_H
        centers.append([cx, cy])
    return torch.tensor(centers, dtype=torch.float32)

ZONE_CENTERS_NORM = _zone_centers_norm()  # (16,2)

# ============================================================
# Seed 고정 (training-seed robustness 실험용)
# ============================================================
def set_global_seed(seed: int):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 가능한 범위에서 재현성 강화
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ============================================================
# TrainConfig
# ============================================================
@dataclass
class TrainConfig:
    total_episodes   : int   = 3000
    max_steps        : int   = 3000   # v9: 배터리 constraint 3000step
    rollout_steps    : int   = 3000
    token_dim        : int   = TOKEN_DIM
    num_heads        : int   = 4
    num_layers       : int   = 3
    ffn_dim          : int   = 256
    dropout          : float = 0.1
    lr_actor         : float = 3e-4
    lr_critic        : float = 1e-4
    gamma            : float = 0.99
    gae_lambda       : float = 0.95
    clip_param       : float = 0.2
    ppo_epochs       : int   = 4
    ppo_batch_size   : int   = 256
    entropy_coef     : float = 0.02
    vf_coef          : float = 0.5
    max_grad_norm    : float = 1.0
    save_dir         : str   = str(REPO_ROOT / "graph_token_mappo_v13" / "stage1")
    log_interval     : int   = 10
    save_interval    : int   = 100
    attn_log_interval: int   = 50
    device           : str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed             : int   = 0
    # ── graph branch ─────────────────────────────────────────────────
    graph_layers : int   = 2      # PaperGATLayer 수
    beta_agent   : float = 2.0
    beta_target  : float = 2.0
    beta_obstacle: float = 2.0

# ============================================================
# CurriculumConfig
# ============================================================
@dataclass
class CurriculumConfig:
    stage         : int = 1
    base_save_dir : str = str(REPO_ROOT / "graph_token_mappo_v13")
    stage_episodes: dict = field(default_factory=lambda: {1:1500, 2:2000, 3:2000})
    stage_goals   : dict = field(default_factory=lambda: {
        1: {"metric":"uncertainty_pct","threshold":20.0,"window":50,"direction":"below"},
        2: {"metric":"terrain_hits",   "threshold":5.0, "window":50,"direction":"below"},
        3: None,
    })
    # v11: boundary는 hard constraint only — PENALTY_BOUNDARY 제거
    # r_unc + r_track + r_stale + r_terrain + r_collision 합산 스칼라
    stage_rewards : dict = field(default_factory=lambda: {
        1: {"LAMBDA_TRACK":0.3, "LAMBDA_STALE":0.01,
            "REWARD_UNCERTAINTY_REDUCE":1.5,
            "PENALTY_TERRAIN":-0.05,
            "PENALTY_DRONE_COLLISION":-1.0, "ANTIAIR_KILL_PENALTY":0.0,
            "BATTERY_COST_TRACK":0.0001},
        2: {"LAMBDA_TRACK":0.5, "LAMBDA_STALE":0.02,
            "REWARD_UNCERTAINTY_REDUCE":1.0,
            "PENALTY_TERRAIN":-0.3,
            "PENALTY_DRONE_COLLISION":-1.0, "ANTIAIR_KILL_PENALTY":0.0},
        3: {"LAMBDA_TRACK":0.5, "LAMBDA_STALE":0.02,
            "REWARD_UNCERTAINTY_REDUCE":1.0,
            "PENALTY_TERRAIN":-0.1,
            "PENALTY_DRONE_COLLISION":-1.0, "ANTIAIR_KILL_PENALTY":0.0},
    })
    def save_dir(self): return os.path.join(self.base_save_dir, f"stage{self.stage}")
    def prev_stage_dir(self):
        return None if self.stage<=1 else os.path.join(self.base_save_dir, f"stage{self.stage-1}")

# ============================================================
# StageMonitor
# ============================================================
class StageMonitor:
    def __init__(self, goal):
        self.goal = goal
        self.history = deque(maxlen=goal["window"])
        self.achieved = False
    def update(self, value):
        if self.goal is None: return False
        self.history.append(value)
        if len(self.history) < self.goal["window"]: return False
        avg  = float(np.mean(self.history))
        done = (avg < self.goal["threshold"] if self.goal["direction"]=="below"
                else avg > self.goal["threshold"])
        if done and not self.achieved:
            self.achieved = True
            print(f"\n  Stage 달성! {self.goal['metric']} {avg:.2f}")
        return done

# ============================================================
# TokenBuilder  (v9: 41 tokens)
# ============================================================
# TokenBuilder  (v2: 33 tokens — Drone×4 + Zone×16 + ActiveTrack×12 + Global×1)

# ============================================================
# GraphTokenObsWrapper  — dual obs 생성 (base_env 한 번만 step)
# ============================================================

def _gat_norm_x(x): return (x - _GAT_MAP_X_MIN) / (_GAT_MAP_X_MAX - _GAT_MAP_X_MIN)
def _gat_norm_y(y): return (y - _GAT_MAP_Y_MIN) / (_GAT_MAP_Y_MAX - _GAT_MAP_Y_MIN)
def _gat_norm_d(d): return float(d) / _GAT_FOV_RADIUS


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
                        if _GAT_MAP_X_MIN <= wx <= _GAT_MAP_X_MAX and _GAT_MAP_Y_MIN <= wy <= _GAT_MAP_Y_MAX:
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
        vec[ptr]   = _gat_norm_x(pos_i[0])
        vec[ptr+1] = _gat_norm_y(pos_i[1])
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
            if dist > _GAT_FOV_RADIUS: continue
            base = ptr + slot * 4
            vec[base]   = diff[0] / _GAT_FOV_RADIUS
            vec[base+1] = diff[1] / _GAT_FOV_RADIUS
            vec[base+2] = _gat_norm_d(dist)
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
            if dists[k] > _GAT_FOV_RADIUS: continue
            if camouflage[k]: continue           # camouflaged → 직접 감지 불가
            diff = t_pos[k] - pos_i
            base = ptr + slot * 4
            vec[base]   = diff[0] / _GAT_FOV_RADIUS
            vec[base+1] = diff[1] / _GAT_FOV_RADIUS
            vec[base+2] = _gat_norm_d(dists[k])
            vec[base+3] = 1.0   # valid flag
            slot += 1
        ptr += TARGET_DIM

        # ── 4. obstacle (terrain) features (MAX_OBSTACLE × 3) ───────
        if self._obstacle_positions is not None and len(self._obstacle_positions) > 0:
            o_pos  = self._obstacle_positions
            o_diff = o_pos - pos_i
            o_dist = np.linalg.norm(o_diff, axis=1)
            in_fov = o_dist <= _GAT_FOV_RADIUS
            idx    = np.where(in_fov)[0]
            idx    = idx[np.argsort(o_dist[idx])][:MAX_OBSTACLE]
            for si, k in enumerate(idx):
                base = ptr + si * 3
                vec[base]   = o_diff[k, 0] / _GAT_FOV_RADIUS
                vec[base+1] = o_diff[k, 1] / _GAT_FOV_RADIUS
                vec[base+2] = _gat_norm_d(o_dist[k])

        return vec


def parse_gat_obs(obs):
    """flat GAT obs → 구조화된 텐서. obs: (B, N, GAT_OBS_DIM)"""
    B, N, _ = obs.shape
    ptr = 0
    self_f = obs[..., ptr:ptr+SELF_DIM]; ptr += SELF_DIM
    nbr_f  = obs[..., ptr:ptr+NEIGHBOR_DIM].reshape(B, N, MAX_NEIGHBOR_DRONE, D_NEIGHBOR_GAT); ptr += NEIGHBOR_DIM
    tgt_f  = obs[..., ptr:ptr+TARGET_DIM].reshape(B, N, MAX_OBS_TARGET, D_TARGET_GAT); ptr += TARGET_DIM
    obs_f  = obs[..., ptr:ptr+OBSTACLE_DIM].reshape(B, N, MAX_OBSTACLE, D_OBSTACLE_GAT)
    nbr_pos = nbr_f[..., :2]; tgt_pos = tgt_f[..., :2]; obs_pos = obs_f[..., :2]
    nbr_mask = nbr_f[..., -1]
    tgt_mask = tgt_f[..., -1]
    obs_mask = (obs_f.norm(dim=-1) > 1e-6).float()
    return self_f, nbr_f, tgt_f, obs_f, nbr_pos, tgt_pos, obs_pos, nbr_mask, tgt_mask, obs_mask


class PaperGATLayer(nn.Module):
    """논문 수식 (12)~(21): 3타입 거리 기반 GAT aggregation."""
    def __init__(self, d, ffn_dim, beta_agent, beta_target, beta_obstacle):
        super().__init__()
        self.beta_a = beta_agent; self.beta_t = beta_target; self.beta_o = beta_obstacle
        self.W_agent    = nn.Linear(d, d, bias=False)
        self.W_target   = nn.Linear(d, d, bias=False)
        self.W_obstacle = nn.Linear(d, d, bias=False)
        self.out_proj   = nn.Sequential(nn.Linear(d * 4, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, d))
        self.norm       = nn.LayerNorm(d)

    @staticmethod
    def _aggregate(rel_pos, value, mask, beta):
        dist  = rel_pos.norm(dim=-1).clamp(min=1e-6)
        score = torch.exp(-beta * dist) * mask.float()
        alpha = score / score.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (alpha.unsqueeze(-1) * value).sum(dim=2)

    def forward(self, drone_h, nbr_h, tgt_h, obs_h,
                nbr_pos, tgt_pos, obs_pos, nbr_mask, tgt_mask, obs_mask):
        h_a = self._aggregate(nbr_pos, self.W_agent(nbr_h),    nbr_mask, self.beta_a)
        h_t = self._aggregate(tgt_pos, self.W_target(tgt_h),   tgt_mask, self.beta_t)
        h_o = self._aggregate(obs_pos, self.W_obstacle(obs_h), obs_mask, self.beta_o)
        cat = torch.cat([drone_h, h_a, h_t, h_o], dim=-1)
        return self.norm(drone_h + self.out_proj(cat))


class GraphTokenObsWrapper:
    """
    base_env를 감싸서 Token branch용 obs484와
    Graph branch용 gobs129를 동시에 반환.

    중요: GATObsWrapper.step()을 호출하지 않음.
    base_env.step() 한 번 → gat_wrapper._build_gat_obs()로 graph obs 생성.
    """

    def __init__(self, base_env, max_ep_steps: int = 3000):
        self.env         = base_env
        self.gat_wrapper = GATObsWrapper(base_env, max_ep_steps=max_ep_steps)
        self.num_drones  = base_env.num_drones
        self.num_targets = base_env.num_targets

    def reset(self):
        obs484 = self.env.reset()
        self.gat_wrapper._step_count = 0
        self.gat_wrapper._build_obstacle_cache()
        gobs129 = self.gat_wrapper._build_gat_obs()
        return obs484, gobs129

    def step(self, actions):
        obs484, rewards, dones, info = self.env.step(actions)
        self.gat_wrapper._step_count += 1
        gobs129 = self.gat_wrapper._build_gat_obs()
        return (obs484, gobs129), rewards, dones, info

    # ── 속성 위임 ─────────────────────────────────────────────
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


class TokenBuilder(nn.Module):
    """
    flat obs (B, N_D, OBS_DIM) →
        drone_feat  (B, N_D,       15)   (v12: other_vec 제거)
        zone_feat   (B, N_ZONE,    10)
        active_feat (B, K_ACTIVE,   6)
        global_feat (B, 1,         11)
        pos_info    (B, N_D+N_ZONE, 2)
    """
    D_DRONE_RAW  = 15  # pos(2)+depth(8)+local_ds(3)+battery(1)+dist_base(1)  (v12: other 제거)
    D_ZONE_RAW   = 10  # center(2)+E/F/U(3)+f_max(1)+u_max(1)+drone_density(1)+fused_mean(1)+fused_max(1)
    D_ACTIVE_RAW = 6   # dx,dy,vx,vy,trP_norm,age_norm
    D_GLOBAL_RAW = 11  # u_mean,u_std,cov,f_mean,mean_battery,mean_trP,stale_norm,prog,active_count,committed_count,fused_mass

    def __init__(self):
        super().__init__()
        self.register_buffer("zone_centers", ZONE_CENTERS_NORM.clone())

    @staticmethod
    def _reshape_ds(obs: torch.Tensor):
        B, N, _ = obs.shape
        e = obs[..., 0  :100].reshape(B, N, 10, 10)
        f = obs[..., 100:200].reshape(B, N, 10, 10)
        u = obs[..., 200:300].reshape(B, N, 10, 10)
        return torch.stack([e, f, u], dim=2)  # (B,N,3,10,10)

    @staticmethod
    def _reshape_intensity(obs: torch.Tensor):
        """
        fused priority map 슬롯 추출: (B,N,100) → (B,N,10,10)
        v11: 이 슬롯은 raw intensity가 아니라 fused priority map
             = U_pool × (1 + β·T_pool) / (1+β)  [0,1] 보장
        """
        B, N, _ = obs.shape
        return obs[..., OBS_INTENSITY_START:OBS_INTENSITY_START+100].reshape(B, N, 10, 10)

    def _zone_feats(self, ds, intensity, drone_pos):
        """
        v11: zone token 10dim
        슬롯[8]=fused_mean, 슬롯[9]=fused_max
        intensity 인자는 fused priority map (v11 기준)
        """
        B = ds.shape[0]
        feats = torch.zeros(B, N_ZONE, self.D_ZONE_RAW, device=ds.device)
        for z, ((x0, x1), (y0, y1)) in enumerate(ZONE_TABLE):
            patch   = ds[:, :, :, x0:x1, y0:y1].mean(dim=1)   # (B,3,dx,dy)
            e_m     = patch[:, 0].mean((-1, -2))
            f_m     = patch[:, 1].mean((-1, -2))
            u_m     = patch[:, 2].mean((-1, -2))
            f_x     = patch[:, 1].amax((-1, -2))
            u_x     = patch[:, 2].amax((-1, -2))
            # fused priority map: 동일 zone slice (10×10 기준)
            i_patch = intensity[:, :, x0:x1, y0:y1].mean(dim=1)   # (B,dx,dy)
            i_mean  = i_patch.mean((-1, -2))
            i_max   = i_patch.amax((-1, -2))   # fused max (reacquire hotspot proxy)
            zc  = self.zone_centers[z].to(ds.device)
            dist_dz = torch.norm(drone_pos - zc, dim=-1)  # (B,N_DRONE)
            cnt = (dist_dz < 0.15).float().sum(1) / max(dist_dz.shape[1], 1)
            feats[:, z, 0] = self.zone_centers[z, 0]
            feats[:, z, 1] = self.zone_centers[z, 1]
            feats[:, z, 2] = e_m
            feats[:, z, 3] = f_m
            feats[:, z, 4] = u_m
            feats[:, z, 5] = f_x
            feats[:, z, 6] = u_x
            feats[:, z, 7] = cnt
            feats[:, z, 8] = i_mean    # fused_mean
            feats[:, z, 9] = i_max     # fused_max
        return feats  # (B,16,10)

    def forward(self, all_obs: torch.Tensor, ep_progress: float = 0.5):
        """
        all_obs: (B, N_DRONE, OBS_DIM)
        Returns: drone_feat, zone_feat, active_feat, global_feat, pos_info
        """
        B   = all_obs.shape[0]
        dev = all_obs.device

        ds        = self._reshape_ds(all_obs)                           # (B,N,3,10,10)
        intensity = self._reshape_intensity(all_obs).mean(dim=1)        # (B,10,10) 드론 평균

        pos   = all_obs[..., OBS_SELF_START :OBS_SELF_START+2]         # (B,N,2)
        depth = all_obs[..., OBS_DEPTH_START:OBS_DEPTH_START+8]        # (B,N,8)
        # v12: other_vec 제거 — Entity Transformer attention이 드론 간 정보 담당
        bat   = all_obs[..., OBS_BATTERY_START:OBS_BATTERY_START+2]    # (B,N,2)

        # local DS 통계
        l_e = ds[:, :, 0].mean((-1, -2))
        l_f = ds[:, :, 1].mean((-1, -2))
        l_u = ds[:, :, 2].mean((-1, -2))
        local_ds = torch.stack([l_e, l_f, l_u], -1)  # (B,N,3)

        # drone_feat (B, N_DRONE, 15)  v12: other 제거
        drone_feat = torch.cat([pos, depth, local_ds, bat], -1)

        # zone_feat (B, 16, 10)
        N_D_actual = all_obs.shape[1]   # 실제 드론 수 (가변)
        zone_feat = self._zone_feats(ds, intensity.unsqueeze(1).expand(-1, N_D_actual, -1, -1), pos)

        # active_feat (B, K_ACTIVE, 6) — 드론 0번 기준 단일 관점 사용
        # dx,dy는 각 드론 기준 상대좌표라 평균내면 좌표계가 섞임
        # → 드론 0번 obs를 기준으로 고정 (좌표 의미 일관성 유지)
        active_raw  = all_obs[..., OBS_ACTIVE_START:OBS_ACTIVE_START + K_ACTIVE*6]
        # (B, N, K*6) → 드론 0번만 사용
        active_d0   = active_raw[:, 0, :]          # (B, K*6)
        active_feat = active_d0.reshape(B, K_ACTIVE, ACTIVE_FEAT_DIM)  # (B,K,6)
        # trP, age, vx, vy (스칼라 통계량)는 드론 평균이 안전 → 해당 채널만 평균
        # [0]=dx,[1]=dy,[2]=vx,[3]=vy,[4]=trP,[5]=age
        # trP/age는 드론 평균 사용 (위치 무관 스칼라)
        active_scalar_avg = active_raw.mean(dim=1).reshape(B, K_ACTIVE, ACTIVE_FEAT_DIM)
        active_feat = torch.cat([
            active_feat[:, :, :2],             # dx,dy: 드론 0번 기준 (좌표계 통일)
            active_scalar_avg[:, :, 2:],       # vx,vy,trP,age: 드론 평균 (스칼라)
        ], dim=-1)  # (B,K,6)

        # global_feat (B, 1, 11)
        u_map        = ds[:, 0, 2]                              # (B,10,10)
        u_mean       = u_map.mean((-1, -2))
        u_std        = u_map.std((-1, -2)).clamp(0, 1)
        cov          = (u_map < 0.3).float().mean((-1, -2))
        f_mean       = ds[:, 0, 1].mean((-1, -2))
        mean_bat     = bat[:, :, 0].mean(1)                     # (B,)
        mean_trP     = active_feat[:, :, 4].mean(1)             # (B,) trP_norm 평균
        stale_norm   = (active_feat[:, :, 5] > 0.5).float().mean(1)  # age 높은 비율
        prog         = torch.full((B,), ep_progress, device=dev)
        # v11: fused priority map 평균 (DS uncertainty × intensity 결합값)
        #      stale/reacquire 핫스팟 강도의 전역 proxy
        active_count     = (active_feat[:, :, 4] < 1.0).float().mean(1)
        committed_count  = (active_feat[:, :, 4] < 0.6).float().mean(1)
        stale_mass_total = intensity.mean((-1, -2))   # (B,) — fused map 기반

        global_feat = torch.stack([
            u_mean, u_std, cov, f_mean, mean_bat,
            mean_trP, stale_norm, prog,
            active_count, committed_count, stale_mass_total
        ], -1).unsqueeze(1)  # (B,1,11)

        # pos_info: drone(N_D) + zone(16)
        zone_pos = self.zone_centers.unsqueeze(0).expand(B, -1, -1).to(dev)
        pos_info = torch.cat([pos, zone_pos], dim=1)  # (B, N_D+16, 2)

        return drone_feat, zone_feat, active_feat, global_feat, pos_info


# ============================================================
# RelationBiasV3  (DD / DZ / DT / ZZ / TT)
# ============================================================
# ============================================================
# RelationBiasV4  (DD / DZ / DA / ZZ / AA / ZA)
# v2 변경: DT→DA (Target→ActiveTrack), TT→AA, ZA 신규 추가
# NT = 33 (4 drone + 16 zone + 12 active + 1 global)
# ============================================================
class RelationBiasV4(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        H = num_heads
        self.num_heads = H
        # DD: [dist, cos, sin]
        self.dd_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, H))
        # DZ: [dist, cos, sin, zone_u]
        self.dz_bias = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, H))
        # DA: [dist, cos, sin, trP_norm, age_norm, vx_hat, vy_hat]  (v2: DT→DA)
        self.da_bias = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, H))
        # ZZ: [dist, adj, du]
        self.zz_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, H))
        # AA: [dist, dtrP, dage]  (v2: TT→AA)
        self.aa_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, H))
        # ZA: [dist, zone_u, trP_norm, age_norm]  (v2 신규)
        self.za_bias = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, H))

    @staticmethod
    def _geo(pi, pj):
        d    = pj - pi
        r    = torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
        dist = (r / 1.5).clamp(0, 1)
        cos  = d[..., 0:1] / r
        sin  = d[..., 1:2] / r
        return dist, cos, sin

    def forward(self, pos_info: torch.Tensor,
                zone_u: torch.Tensor,
                active_feat: torch.Tensor):
        """
        pos_info:    (B, N_D+16, 2)  drone(N_D)+zone(16) 위치
        zone_u:      (B, 16)         zone uncertainty
        active_feat: (B, K, 6)       active track features
        returns:     (B*H, 33, 33)
        """
        B   = pos_info.shape[0]
        H   = self.num_heads
        ND  = pos_info.shape[1] - N_ZONE   # v12: 실제 드론 수 동적 계산
        NT  = ND + N_ZONE + K_ACTIVE + N_GLOBAL
        dev = pos_info.device
        bias = torch.zeros(B, H, NT, NT, device=dev)

        dp = pos_info[:, :ND]    # (B,N_D,2)
        zp = pos_info[:, ND:]    # (B,16,2)
        ap = active_feat[:, :, :2]    # (B,K,2) dx/dy를 상대 위치 proxy로 사용

        NZ = N_ZONE
        NA = K_ACTIVE

        sz_d = 0;        ez_d = ND
        sz_z = ND;       ez_z = ND + NZ
        sz_a = ND + NZ;  ez_a = ND + NZ + NA

        # ── DD ────────────────────────────────────────
        di = dp.unsqueeze(2).expand(-1, -1, ND, -1)
        dj = dp.unsqueeze(1).expand(-1, ND, -1, -1)
        dist, cos, sin = self._geo(di, dj)
        b = torch.tanh(self.dd_bias(torch.cat([dist, cos, sin], -1)))
        bias[:, :, sz_d:ez_d, sz_d:ez_d] = b.permute(0, 3, 1, 2)

        # ── DZ ────────────────────────────────────────
        di = dp.unsqueeze(2).expand(-1, -1, NZ, -1)
        zj = zp.unsqueeze(1).expand(-1, ND, -1, -1)
        dist, cos, sin = self._geo(di, zj)
        uj = zone_u.unsqueeze(1).expand(-1, ND, -1).unsqueeze(-1)
        b  = torch.tanh(self.dz_bias(torch.cat([dist, cos, sin, uj], -1)))
        bias[:, :, sz_d:ez_d, sz_z:ez_z] = b.permute(0, 3, 1, 2)
        bias[:, :, sz_z:ez_z, sz_d:ez_d] = b.permute(0, 3, 2, 1)

        # ── DA (핵심: 드론-ActiveTrack) ───────────────
        di_xy = dp.unsqueeze(2).expand(-1, -1, NA, -1)   # (B,N_D,K,2)
        aj_xy = ap.unsqueeze(1).expand(-1, ND, -1, -1)   # (B,N_D,K,2)
        d_da  = aj_xy - di_xy
        r_da  = torch.norm(d_da, dim=-1, keepdim=True).clamp(min=1e-6)
        dist_da = (r_da / 1.0).clamp(0, 1)
        cos_da  = d_da[..., 0:1] / r_da
        sin_da  = d_da[..., 1:2] / r_da
        trP  = active_feat[:, None, :, 4:5].expand(-1, ND, -1, -1)
        age  = active_feat[:, None, :, 5:6].expand(-1, ND, -1, -1)
        vx   = active_feat[:, None, :, 2:3].expand(-1, ND, -1, -1)
        vy   = active_feat[:, None, :, 3:4].expand(-1, ND, -1, -1)
        da_in = torch.cat([dist_da, cos_da, sin_da, trP, age, vx, vy], -1)
        b = torch.tanh(self.da_bias(da_in))              # (B,N_D,K,H)
        bias[:, :, sz_d:ez_d, sz_a:ez_a] = b.permute(0, 3, 1, 2)
        bias[:, :, sz_a:ez_a, sz_d:ez_d] = b.permute(0, 3, 2, 1)

        # ── ZZ ────────────────────────────────────────
        zi = zp.unsqueeze(2).expand(-1, -1, NZ, -1)
        zj = zp.unsqueeze(1).expand(-1, NZ, -1, -1)
        dist, _, _ = self._geo(zi, zj)
        adj  = (dist < 0.15).float()
        ui   = zone_u.unsqueeze(2).expand(-1, -1, NZ).unsqueeze(-1)
        uj2  = zone_u.unsqueeze(1).expand(-1, NZ, -1).unsqueeze(-1)
        du   = (uj2 - ui).abs()
        b    = torch.tanh(self.zz_bias(torch.cat([dist, adj, du], -1)))
        bias[:, :, sz_z:ez_z, sz_z:ez_z] = b.permute(0, 3, 1, 2)

        # ── AA (ActiveTrack-ActiveTrack) ──────────────
        ai = ap.unsqueeze(2).expand(-1, -1, NA, -1)
        aj = ap.unsqueeze(1).expand(-1, NA, -1, -1)
        dist_aa, _, _ = self._geo(ai, aj)
        trPi = active_feat[:, :, 4].unsqueeze(2).expand(-1, -1, NA).unsqueeze(-1)
        trPj = active_feat[:, :, 4].unsqueeze(1).expand(-1, NA, -1).unsqueeze(-1)
        dtrP = (trPj - trPi).abs()
        agei = active_feat[:, :, 5].unsqueeze(2).expand(-1, -1, NA).unsqueeze(-1)
        agej = active_feat[:, :, 5].unsqueeze(1).expand(-1, NA, -1).unsqueeze(-1)
        dage = (agej - agei).abs()
        b    = torch.tanh(self.aa_bias(torch.cat([dist_aa, dtrP, dage], -1)))
        bias[:, :, sz_a:ez_a, sz_a:ez_a] = b.permute(0, 3, 1, 2)

        # ── ZA (Zone-ActiveTrack, 신규) ───────────────
        zi_za = zp.unsqueeze(2).expand(-1, -1, NA, -1)   # (B,16,K,2)
        aj_za = ap.unsqueeze(1).expand(-1, NZ, -1, -1)   # (B,16,K,2)
        d_za  = aj_za - zi_za
        r_za  = torch.norm(d_za, dim=-1, keepdim=True).clamp(min=1e-6)
        dist_za = (r_za / 1.0).clamp(0, 1)
        zu_za   = zone_u.unsqueeze(2).expand(-1, -1, NA).unsqueeze(-1)
        trP_za  = active_feat[:, None, :, 4:5].expand(-1, NZ, -1, -1)
        age_za  = active_feat[:, None, :, 5:6].expand(-1, NZ, -1, -1)
        za_in   = torch.cat([dist_za, zu_za, trP_za, age_za], -1)
        b = torch.tanh(self.za_bias(za_in))              # (B,16,K,H)
        bias[:, :, sz_z:ez_z, sz_a:ez_a] = b.permute(0, 3, 1, 2)
        bias[:, :, sz_a:ez_a, sz_z:ez_z] = b.permute(0, 3, 2, 1)

        return bias.reshape(B * H, NT, NT)  # NT = ND+NZ+NA+NG (가변)


# ============================================================
# EntityTransformerLayer / Encoder
# ============================================================
class EntityTransformerLayer(nn.Module):
    def __init__(self, d, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d,ffn_dim), nn.GELU(), nn.Linear(ffn_dim,d))
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, need_weights=False):
        n = self.ln1(x)
        o, w = self.attn(n, n, n, attn_mask=attn_mask,
                         need_weights=need_weights,
                         average_attn_weights=need_weights)
        x = x + self.drop(o)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, w

class EntityTransformerEncoder(nn.Module):
    def __init__(self, d, num_heads, ffn_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EntityTransformerLayer(d, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None, return_attn=False):
        w = None
        for i, layer in enumerate(self.layers):
            nw = return_attn and (i == len(self.layers)-1)
            x, aw = layer(x, attn_mask=attn_mask, need_weights=nw)
            if nw: w = aw
        return x, w

# ============================================================
# _BaseEntityNet  (Actor / Critic 공통 backbone)
# ============================================================
class _BaseEntityNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.token_dim
        self._D = D
        # 타입별 projection  (v2: zone 10dim, active 6dim, global 11dim)
        self.drone_proj  = nn.Linear(TokenBuilder.D_DRONE_RAW,  D)
        self.zone_proj   = nn.Linear(TokenBuilder.D_ZONE_RAW,   D)
        self.active_proj = nn.Linear(TokenBuilder.D_ACTIVE_RAW, D)
        self.global_proj = nn.Linear(TokenBuilder.D_GLOBAL_RAW, D)
        # type embedding: 0=drone, 1=zone, 2=active, 3=global
        self.type_emb = nn.Embedding(4, D)
        # id embedding
        self.drone_id_emb  = nn.Embedding(N_DRONE_MAX, D)  # v12: 최대 드론 수 지원
        self.zone_id_emb   = nn.Embedding(N_ZONE,       D)
        self.active_id_emb = nn.Embedding(K_ACTIVE,     D)
        # position projection
        self.pos_proj = nn.Linear(2, D)
        # transformer
        self.encoder  = EntityTransformerEncoder(
            D, cfg.num_heads, cfg.ffn_dim, cfg.num_layers, cfg.dropout)

    def _build_tokens(self, df, zf, af, gf, pi):
        """
        df: (B, N_D,     15)  drone  (v12: other 제거로 21→15)
        zf: (B, 16,      10)  zone
        af: (B, K_ACTIVE,  6)  active track
        gf: (B,  1,      11)  global
        pi: (B, N_D+16,   2)  pos_info
        returns: (B, N_D+N_ZONE+K_ACTIVE+1, D)  — N_D 가변
        """
        B    = df.shape[0]
        N_D  = df.shape[1]   # 실제 드론 수 (4/8/12)
        dev  = df.device
        D    = self._D

        d_tok = self.drone_proj(df)   # (B,N_D,D)
        z_tok = self.zone_proj(zf)    # (B,16,D)
        a_tok = self.active_proj(af)  # (B,K,D)
        g_tok = self.global_proj(gf)  # (B,1,D)

        # type embedding (N_D 가변)
        type_ids = torch.tensor(
            [0]*N_D + [1]*N_ZONE + [2]*K_ACTIVE + [3]*N_GLOBAL,
            dtype=torch.long, device=dev)
        type_e = self.type_emb(type_ids).unsqueeze(0).expand(B, -1, -1)

        # id embedding (drone_id_emb은 N_DRONE_MAX=12 크기)
        did    = torch.arange(N_D,      device=dev)
        zid    = torch.arange(N_ZONE,   device=dev)
        aid    = torch.arange(K_ACTIVE, device=dev)
        d_id_e = self.drone_id_emb(did).unsqueeze(0).expand(B, -1, -1)
        z_id_e = self.zone_id_emb(zid).unsqueeze(0).expand(B, -1, -1)
        a_id_e = self.active_id_emb(aid).unsqueeze(0).expand(B, -1, -1)
        g_id_e = torch.zeros(B, 1, D, device=dev)

        # position embedding (drone+zone만)
        pos_dz = self.pos_proj(pi)                         # (B,N_D+16,D)
        pos_a  = torch.zeros(B, K_ACTIVE, D, device=dev)
        pos_g  = torch.zeros(B, 1,        D, device=dev)
        pos_e  = torch.cat([pos_dz, pos_a, pos_g], dim=1)

        id_e = torch.cat([d_id_e, z_id_e, a_id_e, g_id_e], dim=1)
        toks = torch.cat([d_tok,  z_tok,  a_tok,  g_tok],  dim=1)
        return toks + type_e + id_e + pos_e  # (B, N_D+N_ZONE+K+1, D)

# ============================================================
# EntityActorNet / EntityCriticNet
# ============================================================
class EntityActorNet(_BaseEntityNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        D = cfg.token_dim
        self.head       = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, ACTION_DIM))
        self.drone_proj = nn.Linear(TokenBuilder.D_DRONE_RAW, D)

    def forward(self, df, zf, af, gf, pi, attn_mask=None, return_attn=False):
        toks = self._build_tokens(df, zf, af, gf, pi)
        out, w = self.encoder(toks, attn_mask=attn_mask, return_attn=return_attn)
        N_D = df.shape[1]
        drone_out = out[:, :N_D, :]
        return self.head(drone_out), w

    def forward_fused(self, df_fused, zf, af, gf, pi, attn_mask=None, return_attn=False):
        """
        df_fused: (B, N_D, D) — 이미 fusion된 drone embedding (projection 불필요)
        _build_tokens에서 drone을 다시 project하는 걸 우회해서 fused 값을 직접 주입.
        """
        toks = self._build_tokens_with_drone_h(df_fused, zf, af, gf, pi)
        out, w = self.encoder(toks, attn_mask=attn_mask, return_attn=return_attn)
        N_D = df_fused.shape[1]
        drone_out = out[:, :N_D, :]
        return self.head(drone_out), w

    def _build_tokens_with_drone_h(self, drone_h, zf, af, gf, pi):
        """_build_tokens와 동일하되 drone은 이미 D차원이므로 projection 건너뜀."""
        B   = drone_h.shape[0]
        N_D = drone_h.shape[1]
        dev = drone_h.device
        D   = drone_h.shape[-1]

        z_tok = self.zone_proj(zf)
        a_tok = self.active_proj(af)
        g_tok = self.global_proj(gf)

        # type embedding (한 번만)
        type_ids = torch.tensor(
            [0] * N_D + [1] * N_ZONE + [2] * K_ACTIVE + [3] * N_GLOBAL,
            dtype=torch.long, device=dev)
        type_e = self.type_emb(type_ids).unsqueeze(0).expand(B, -1, -1)

        # id embedding — drone/zone/active 각각 전용 embedding
        did   = torch.arange(N_D,      device=dev)
        zid   = torch.arange(N_ZONE,   device=dev)
        aid   = torch.arange(K_ACTIVE, device=dev)
        d_id_e = self.drone_id_emb(did).unsqueeze(0).expand(B, -1, -1)
        z_id_e = self.zone_id_emb(zid).unsqueeze(0).expand(B, -1, -1)
        a_id_e = self.active_id_emb(aid).unsqueeze(0).expand(B, -1, -1)
        g_id_e = torch.zeros(B, N_GLOBAL, D, device=dev)
        id_e   = torch.cat([d_id_e, z_id_e, a_id_e, g_id_e], dim=1)

        # position embedding
        pos_dz = self.pos_proj(pi)
        pos_a  = torch.zeros(B, K_ACTIVE, D, device=dev)
        pos_g  = torch.zeros(B, N_GLOBAL, D, device=dev)
        pos_e  = torch.cat([pos_dz, pos_a, pos_g], dim=1)

        toks = torch.cat([drone_h, z_tok, a_tok, g_tok], dim=1)
        return toks + type_e + id_e + pos_e


class EntityCriticNet(_BaseEntityNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        D = cfg.token_dim
        self.head       = nn.Sequential(nn.Linear(D*2, D), nn.GELU(), nn.Linear(D, 1))
        self.drone_proj = nn.Linear(TokenBuilder.D_DRONE_RAW, D)

    def forward(self, df, zf, af, gf, pi, attn_mask=None, return_attn=False):
        toks = self._build_tokens(df, zf, af, gf, pi)
        out, w = self.encoder(toks, attn_mask=attn_mask, return_attn=return_attn)
        N_D    = df.shape[1]
        idx_g  = N_D + N_ZONE + K_ACTIVE
        g_out  = out[:, idx_g:idx_g+1, :]
        pool   = out.mean(dim=1, keepdim=True)
        fused  = torch.cat([g_out, pool], dim=-1)
        v      = self.head(fused).squeeze(-1).expand(-1, N_D)
        return v, w

    def forward_fused_critic(self, df_fused, zf, af, gf, pi,
                              graph_global_h, critic_fuse_layer,
                              attn_mask=None, return_attn=False):
        """
        df_fused: (B, N_D, D) fused drone embedding
        graph_global_h: (B, D) GAT global embedding
        critic_fuse_layer: EntityMAPPOCritic.critic_fuse
        """
        # drone token 직접 주입
        toks = self._build_tokens_with_drone_h(df_fused, zf, af, gf, pi)
        out, w = self.encoder(toks, attn_mask=attn_mask, return_attn=return_attn)
        N_D   = df_fused.shape[1]
        idx_g = N_D + N_ZONE + K_ACTIVE
        # token global + graph global → critic_fuse
        token_global_h = out[:, idx_g, :]       # (B, D)
        fused_global   = critic_fuse_layer(
            torch.cat([token_global_h, graph_global_h], dim=-1))  # (B, D)
        pool   = out.mean(dim=1)                 # (B, D)
        v      = self.head(
            torch.cat([fused_global.unsqueeze(1), pool.unsqueeze(1)], dim=-1)
        ).squeeze(-1).expand(-1, N_D)            # (B, N_D)
        return v, w

    def _build_tokens_with_drone_h(self, drone_h, zf, af, gf, pi):
        """_build_tokens와 동일하되 drone은 이미 D차원이므로 projection 건너뜀."""
        B   = drone_h.shape[0]
        N_D = drone_h.shape[1]
        dev = drone_h.device
        D   = drone_h.shape[-1]

        z_tok = self.zone_proj(zf)
        a_tok = self.active_proj(af)
        g_tok = self.global_proj(gf)

        # type embedding (한 번만)
        type_ids = torch.tensor(
            [0] * N_D + [1] * N_ZONE + [2] * K_ACTIVE + [3] * N_GLOBAL,
            dtype=torch.long, device=dev)
        type_e = self.type_emb(type_ids).unsqueeze(0).expand(B, -1, -1)

        # id embedding — drone/zone/active 각각 전용 embedding
        did   = torch.arange(N_D,      device=dev)
        zid   = torch.arange(N_ZONE,   device=dev)
        aid   = torch.arange(K_ACTIVE, device=dev)
        d_id_e = self.drone_id_emb(did).unsqueeze(0).expand(B, -1, -1)
        z_id_e = self.zone_id_emb(zid).unsqueeze(0).expand(B, -1, -1)
        a_id_e = self.active_id_emb(aid).unsqueeze(0).expand(B, -1, -1)
        g_id_e = torch.zeros(B, N_GLOBAL, D, device=dev)
        id_e   = torch.cat([d_id_e, z_id_e, a_id_e, g_id_e], dim=1)

        # position embedding
        pos_dz = self.pos_proj(pi)
        pos_a  = torch.zeros(B, K_ACTIVE, D, device=dev)
        pos_g  = torch.zeros(B, N_GLOBAL, D, device=dev)
        pos_e  = torch.cat([pos_dz, pos_a, pos_g], dim=1)

        toks = torch.cat([drone_h, z_tok, a_tok, g_tok], dim=1)
        return toks + type_e + id_e + pos_e



# ============================================================
# EntityMAPPOActor / EntityMAPPOCritic
# ============================================================

# ============================================================
# GraphBranch — GAT local relation encoder
# ============================================================
class GraphBranch(nn.Module):
    """
    gat_mappo_paper_v13.PaperGATBackbone 기반.
    gobs (B, N, GAT_OBS_DIM=129) → graph_drone_h (B,N,D), graph_global_h (B,D)
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg.token_dim

        self.self_proj     = nn.Linear(D_SELF_GAT,     D)
        self.neighbor_proj = nn.Linear(D_NEIGHBOR_GAT, D)
        self.target_proj   = nn.Linear(D_TARGET_GAT,   D)
        self.obstacle_proj = nn.Linear(D_OBSTACLE_GAT, D)

        self.layers = nn.ModuleList([
            PaperGATLayer(D, cfg.token_dim * 2,
                          cfg.beta_agent, cfg.beta_target, cfg.beta_obstacle)
            for _ in range(cfg.graph_layers)
        ])
        self.global_proj = nn.Linear(D * 4, D)

    def forward(self, gobs: torch.Tensor):
        """
        gobs: (B, N, GAT_OBS_DIM)
        Returns:
            graph_drone_h:  (B, N, D)
            graph_global_h: (B, D)
        """
        (self_f, nbr_f, tgt_f, obs_f,
         nbr_pos, tgt_pos, obs_pos,
         nbr_mask, tgt_mask, obs_mask) = parse_gat_obs(gobs)

        drone_h = self.self_proj(self_f)
        nbr_h   = self.neighbor_proj(nbr_f)
        tgt_h   = self.target_proj(tgt_f)
        obs_h   = self.obstacle_proj(obs_f)

        for layer in self.layers:
            drone_h = layer(drone_h, nbr_h, tgt_h, obs_h,
                            nbr_pos, tgt_pos, obs_pos,
                            nbr_mask, tgt_mask, obs_mask)

        drone_pool = drone_h.mean(1)

        def _mp(h, m):
            cnt = m.sum(-1, keepdim=True).clamp(min=1)
            return (h * m.unsqueeze(-1)).sum(2) / cnt

        nbr_pool = _mp(nbr_h, nbr_mask).mean(1)
        tgt_pool = _mp(tgt_h, tgt_mask).mean(1)
        obs_pool = _mp(obs_h, obs_mask).mean(1)

        graph_global_h = self.global_proj(
            torch.cat([drone_pool, nbr_pool, tgt_pool, obs_pool], dim=-1))

        return drone_h, graph_global_h


class EntityMAPPOActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tb           = TokenBuilder()
        self.rb           = RelationBiasV4(cfg.num_heads)
        self.net          = EntityActorNet(cfg)
        self.H            = cfg.num_heads
        self.graph_branch = GraphBranch(cfg)
        D = cfg.token_dim
        # drone token fusion: token_drone_h(D) + graph_drone_h(D) → D
        self.graph_fuse   = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.Linear(D, D))

    def _build_mask(self, pi, zone_u, active_feat, B):
        bias = self.rb(pi, zone_u, active_feat)   # (B*H, NT, NT) NT 가변
        eye  = torch.zeros_like(bias)
        # drone 토큰 자기 자신 attend 방지 (앞 N_D 개)
        N_D  = pi.shape[1] - N_ZONE
        for i in range(N_D):
            eye[:, i, i] = -1e9
        return bias + eye

    def forward(self, obs, gobs, ep_prog=0.5, return_attn=False):
        """
        obs:  (B, N, OBS_DIM)   Token branch input
        gobs: (B, N, GAT_OBS_DIM) Graph branch input
        """
        B = obs.shape[0]
        df, zf, af, gf, pi = self.tb(obs, ep_prog)
        zone_u = zf[:, :, 4]
        msk    = self._build_mask(pi, zone_u, af, B)

        # Graph branch → drone embedding 추출
        graph_drone_h, _ = self.graph_branch(gobs)   # (B, N, D)

        # drone token에 graph local relation embedding 주입
        N_D = df.shape[1]
        # EntityActorNet._build_tokens 내부에서 df를 projection하므로
        # df를 graph_fuse 결과로 교체한 뒤 전달
        # df: (B, N, D_DRONE_RAW=15), graph_drone_h: (B, N, D)
        # → 먼저 df를 D로 projection해서 합침
        D = graph_drone_h.shape[-1]
        df_proj = self.net.drone_proj(df)              # (B, N, D)
        df_fused = self.graph_fuse(
            torch.cat([df_proj, graph_drone_h], dim=-1))  # (B, N, D)

        return self.net.forward_fused(
            df_fused, zf, af, gf, pi, attn_mask=msk, return_attn=return_attn)

class EntityMAPPOCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tb            = TokenBuilder()
        self.rb            = RelationBiasV4(cfg.num_heads)
        self.net           = EntityCriticNet(cfg)
        self.H             = cfg.num_heads
        self.graph_branch  = GraphBranch(cfg)
        D = cfg.token_dim
        # critic fusion: token_drone_h(D) + graph_drone_h(D) → D (actor와 동일)
        self.graph_fuse    = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.Linear(D, D))
        # critic global fusion: token_global_h(D) + graph_global_h(D) → D
        self.critic_fuse   = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.Linear(D, D))

    def _build_mask(self, pi, zone_u, active_feat, B):
        bias = self.rb(pi, zone_u, active_feat)
        eye  = torch.zeros_like(bias)
        N_D  = pi.shape[1] - N_ZONE
        for i in range(N_D):
            eye[:, i, i] = -1e9
        return bias + eye

    def forward(self, obs, gobs, ep_prog=0.5, return_attn=False):
        """
        obs:  (B, N, OBS_DIM)    Token branch input
        gobs: (B, N, GAT_OBS_DIM) Graph branch input
        """
        B = obs.shape[0]
        df, zf, af, gf, pi = self.tb(obs, ep_prog)
        zone_u = zf[:, :, 4]
        msk    = self._build_mask(pi, zone_u, af, B)

        # Graph branch
        graph_drone_h, graph_global_h = self.graph_branch(gobs)  # (B,N,D), (B,D)

        # drone token fusion (actor와 동일하게)
        df_proj  = self.net.drone_proj(df)
        df_fused = self.graph_fuse(
            torch.cat([df_proj, graph_drone_h], dim=-1))

        return self.net.forward_fused_critic(
            df_fused, zf, af, gf, pi,
            graph_global_h, self.critic_fuse,
            attn_mask=msk, return_attn=return_attn)


# ============================================================
# RolloutBuffer
# ============================================================
class RolloutBuffer:
    def __init__(self, capacity, num_drones=N_DRONE_TRAIN,
                 obs_dim=OBS_DIM, device="cpu"):
        self.capacity = capacity
        self.device   = device
        self.pos      = 0
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs        = z(capacity, num_drones, obs_dim)
        self.gobs       = z(capacity, num_drones, GAT_OBS_DIM)   # graph obs
        self.actions    = torch.zeros(capacity, num_drones,
                                       dtype=torch.int64, device=device)
        self.logprobs   = z(capacity, num_drones)
        self.rewards    = z(capacity, num_drones)
        self.values     = z(capacity, num_drones)
        self.dones      = z(capacity, num_drones)
        self.returns    = z(capacity, num_drones)
        self.advantages = z(capacity, num_drones)

    def push(self, obs, gobs, actions, logprobs, rewards, values, dones):
        p = self.pos
        self.obs[p]      = obs
        self.gobs[p]     = gobs
        self.actions[p]  = actions
        self.logprobs[p] = logprobs
        self.rewards[p]  = rewards
        self.values[p]   = values
        self.dones[p]    = dones
        self.pos += 1

    def compute_gae(self, next_value, next_done, gamma, lam):
        gae = 0
        for t in reversed(range(self.pos)):
            nt = 1.0 - next_done if t == self.pos-1 else 1.0 - self.dones[t+1]
            nv = next_value       if t == self.pos-1 else self.values[t+1]
            delta = self.rewards[t] + gamma*nv*nt - self.values[t]
            gae   = delta + gamma*lam*nt*gae
            self.advantages[t] = gae
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

    def get_generator(self, batch_size):
        idx = np.random.permutation(self.pos)
        for s in range(0, self.pos, batch_size):
            i = idx[s:s+batch_size]
            yield (self.obs[i], self.gobs[i], self.actions[i],
                   self.logprobs[i], self.returns[i], self.advantages[i])

    def clear(self): self.pos = 0

# ============================================================
# MAPPOAgent
# ============================================================
class MAPPOAgent:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.actor  = EntityMAPPOActor(cfg).to(self.dev)
        self.critic = EntityMAPPOCritic(cfg).to(self.dev)
        self.a_opt  = optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.c_opt  = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.buffer = RolloutBuffer(cfg.rollout_steps+10, device=self.dev)
        # graph branch 파라미터 수 출력
        self.episode    = 0
        self.total_steps= 0
        self.ep_prog    = 0.0
        self.ep_rewards_history  = deque(maxlen=100)
        self.loss_actor_history  = deque(maxlen=100)
        self.loss_critic_history = deque(maxlen=100)
        ap = sum(p.numel() for p in self.actor.parameters())
        cp = sum(p.numel() for p in self.critic.parameters())
        print("MAPPOAgent (v9 Entity Transformer) 초기화 완료")
        print(f"  장치   : {self.dev}")
        print(f"  Actor  : {ap:,} params")
        print(f"  Critic : {cp:,} params")
        print(f"  Tokens : Dronex{N_DRONE_TRAIN}(train)/up to {N_DRONE_MAX}(eval) "
              f"+ Zonex{N_ZONE} + Trackx{K_ACTIVE} + Globalx{N_GLOBAL}")

    @torch.no_grad()
    def get_action_and_value(self, obs, gobs, deterministic=False, return_attn=False):
        obs_t  = torch.FloatTensor(obs).to(self.dev).unsqueeze(0)
        gobs_t = torch.FloatTensor(gobs).to(self.dev).unsqueeze(0)
        n_drone = obs.shape[0]
        logits, aw = self.actor(obs_t, gobs_t, self.ep_prog, return_attn=return_attn)
        actions  = torch.zeros(n_drone, dtype=torch.int64,   device=self.dev)
        logprobs = torch.zeros(n_drone, dtype=torch.float32, device=self.dev)
        for i in range(n_drone):
            dist = Categorical(logits=logits[:, i, :])
            a    = logits[:, i, :].argmax(-1) if deterministic else dist.sample()
            actions[i]  = a.squeeze()
            logprobs[i] = dist.log_prob(a).squeeze()
        v, cw = self.critic(obs_t, gobs_t, self.ep_prog, return_attn=return_attn)
        return actions, logprobs, v.squeeze(0), aw, cw

    def update(self):
        cfg = self.cfg
        buf = self.buffer
        ta = tc = te = tf = nu = 0.0
        total_cgrad = total_agrad = 0.0

        # diag: adv/ret/val std — 전체 버퍼 기준 (update 전 계산)
        T = buf.pos
        diag_adv_std = float(buf.advantages[:T].flatten().std()) if T > 0 else 0.0
        diag_ret_std = float(buf.returns[:T].flatten().std())    if T > 0 else 0.0
        diag_val_std = float(buf.values[:T].flatten().std())     if T > 0 else 0.0

        for _ in range(cfg.ppo_epochs):
            for mb_obs, mb_gobs, mb_act, mb_lp, mb_ret, mb_adv in \
                    buf.get_generator(cfg.ppo_batch_size):
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Critic update
                V, _ = self.critic(mb_obs, mb_gobs, self.ep_prog)
                cl   = cfg.vf_coef * F.mse_loss(V, mb_ret)
                self.c_opt.zero_grad(); cl.backward()
                cgnorm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm)
                self.c_opt.step()
                total_cgrad += float(cgnorm)

                # Actor update
                logits, _ = self.actor(mb_obs, mb_gobs, self.ep_prog)
                n_d = mb_obs.shape[1]   # 실제 드론 수
                al = 0.0; ent = 0.0; cf = 0.0
                for i in range(n_d):
                    d   = Categorical(logits=logits[:, i, :])
                    lp  = d.log_prob(mb_act[:, i])
                    ent += d.entropy().mean().item()
                    r   = torch.exp(lp - mb_lp[:, i])
                    s1  = r * mb_adv[:, i]
                    s2  = torch.clamp(r, 1-cfg.clip_param,
                                         1+cfg.clip_param) * mb_adv[:, i]
                    al  += (-torch.min(s1,s2).mean()
                            - cfg.entropy_coef * d.entropy().mean())
                    cf  += ((r-1).abs() > cfg.clip_param).float().mean().item()
                al = al / max(n_d, 1)
                self.a_opt.zero_grad(); al.backward()
                agnorm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), cfg.max_grad_norm)
                self.a_opt.step()
                total_agrad += float(agnorm)

                ta += al.item(); tc += cl.item()
                te += ent/max(n_d,1); tf += cf/max(n_d,1); nu += 1

        buf.clear()
        n = max(nu, 1)
        info = {
            "actor_loss"      : ta / n,
            "critic_loss"     : tc / n,
            "entropy"         : te / n,
            "clip_frac"       : tf / n,
            "critic_grad_norm": total_cgrad / n,
            "actor_grad_norm" : total_agrad / n,
            "diag_adv_std"    : diag_adv_std,
            "diag_ret_std"    : diag_ret_std,
            "diag_val_std"    : diag_val_std,
        }
        self.loss_actor_history.append(info["actor_loss"])
        self.loss_critic_history.append(info["critic_loss"])
        return info

    def save(self, ep):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        p = os.path.join(self.cfg.save_dir, f"graph_token_ep{ep:05d}.pt")
        torch.save({"episode": ep, "actor": self.actor.state_dict(),
                    "critic":  self.critic.state_dict(),
                    "total_steps": self.total_steps}, p)
        print(f"  [저장] {p}"); return p

    def load(self, path):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ck["actor"], strict=True)
        self.critic.load_state_dict(ck["critic"], strict=True)
        self.total_steps = ck.get("total_steps", 0)
        ep = ck.get("episode", 0)
        print(f"  [로드] {path}  (ep={ep})"); return ep

# ============================================================
# TrainLogger  (v9 확장 TensorBoard)
# ============================================================
class TrainLogger:
    def __init__(self, save_dir, stage=1):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(save_dir, "train_log.csv")
        with open(self.log_path, "w") as f:
            f.write("episode,steps,reward_mean,reward_sum,"
                    "uncertainty_pct,terrain_hits,drone_colls,"
                    "tracked_targets,mean_trP,stale_count,"
                    "mean_battery,rtb_count,active_at_end,"
                    "r_track_mean,r_stale_mean,r_unc_mean,"
                    "critic_loss,actor_loss,entropy,clip_frac,time_s\n")
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = os.path.join(save_dir, "tb_logs", f"stage{stage}")
            self.writer = SummaryWriter(log_dir=tb)
            print(f"  [TensorBoard] {tb}")
            print(f"  [TensorBoard] tensorboard --logdir {tb}")
        except ImportError:
            print("  [경고] tensorboard 미설치"); self.writer = None

    def log_episode(self, ep, steps, rewards, info, loss_info, elapsed):
        """
        rewards: (N_DRONE,) numpy
        info: EpisodeInfoAccumulator.summarize() 반환값
        loss_info: agent.update() 반환값 (grad_norm, diag 포함)
        """
        r_mean   = float(np.mean(rewards))
        r_sum    = float(np.sum(rewards))
        u_pct    = info.get("uncertainty_pct",      0.0)
        t_hits   = info.get("terrain_hits_ep",      0)
        d_colls  = info.get("drone_colls_ep",       0)
        tracked  = info.get("tracked_targets",      0)
        mean_trP = info.get("mean_tr_P",            1.0)
        stale    = info.get("stale_count",          0)
        mean_bat = info.get("mean_battery_ep",      1.0)
        rtb_cnt  = info.get("rtb_count_ep",         0)
        active   = info.get("active_at_end",        NUM_DRONES)
        r_track  = info.get("r_track_ep",           0.0)
        r_stale  = info.get("r_stale_ep",           0.0)
        r_unc    = info.get("r_unc_ep",             0.0)
        cov      = max(0.0, 1.0 - u_pct / 100.0)

        with open(self.log_path, "a") as f:
            f.write(f"{ep},{steps},{r_mean:.4f},{r_sum:.4f},"
                    f"{u_pct:.1f},{t_hits},{d_colls},"
                    f"{tracked},{mean_trP:.4f},{stale},"
                    f"{mean_bat:.4f},{rtb_cnt},{active},"
                    f"{r_track:.4f},{r_stale:.4f},{r_unc:.4f},"
                    f"{loss_info.get('critic_loss',0):.6f},"
                    f"{loss_info.get('actor_loss',0):.6f},"
                    f"{loss_info.get('entropy',0):.4f},"
                    f"{loss_info.get('clip_frac',0):.4f},{elapsed:.1f}\n")

        if not self.writer: return
        w = self.writer

        # ── reward ───────────────────────────────────
        w.add_scalar("reward/mean",  r_mean, ep)
        w.add_scalar("reward/sum",   r_sum,  ep)

        # ── safety ───────────────────────────────────
        w.add_scalar("safety/drone_colls",   d_colls,  ep)
        w.add_scalar("safety/terrain_hits",  t_hits,   ep)
        w.add_scalar("safety/mean_battery",  mean_bat, ep)
        w.add_scalar("safety/rtb_count",     rtb_cnt,  ep)
        w.add_scalar("safety/active_at_end", active,   ep)

        # ── search ───────────────────────────────────
        w.add_scalar("search/uncertainty_pct", u_pct, ep)
        w.add_scalar("search/coverage_rate",   cov,   ep)
        w.add_scalar("search/r_unc_mean",      r_unc, ep)

        # ── target ───────────────────────────────────
        w.add_scalar("target/tracked_targets",      tracked,  ep)
        w.add_scalar("target/mean_trP",             mean_trP, ep)
        w.add_scalar("target/stale_count",          stale,    ep)
        w.add_scalar("target/committed_count",
                     info.get("committed_count",     0.0), ep)
        w.add_scalar("target/tracked_committed_ep",
                     info.get("tracked_committed_ep",0.0), ep)
        w.add_scalar("target/tracked_stale_ep",
                     info.get("tracked_stale_ep",   0.0), ep)
        w.add_scalar("target/candidate_count",
                     info.get("candidate_count",    0.0), ep)
        w.add_scalar("target/r_track_mean",    r_track, ep)
        w.add_scalar("target/r_stale_mean",    r_stale, ep)
        w.add_scalar("target/r_reacquire_mean",
                     info.get("r_reacquire_ep", 0.0), ep)

        # ── train ────────────────────────────────────
        w.add_scalar("train/critic_loss",      loss_info.get("critic_loss",      0), ep)
        w.add_scalar("train/actor_loss",       loss_info.get("actor_loss",       0), ep)
        w.add_scalar("train/entropy",          loss_info.get("entropy",          0), ep)
        w.add_scalar("train/clip_frac",        loss_info.get("clip_frac",        0), ep)
        w.add_scalar("train/critic_grad_norm", loss_info.get("critic_grad_norm", 0), ep)
        w.add_scalar("train/actor_grad_norm",  loss_info.get("actor_grad_norm",  0), ep)
        w.add_scalar("train/steps_per_ep",     float(steps),   ep)
        w.add_scalar("train/time_s",           float(elapsed), ep)

        # ── diag: advantage / return / value 분포 ───
        w.add_scalar("diag/adv_std", loss_info.get("diag_adv_std", 0), ep)
        w.add_scalar("diag/ret_std", loss_info.get("diag_ret_std", 0), ep)
        w.add_scalar("diag/val_std", loss_info.get("diag_val_std", 0), ep)

        # ── v12 신규: IMM 모드 확률 / 포병 relocation ──
        w.add_scalar("imm/tank0_cv_prob",
                     info.get("tank0_imm_cv_ep", 0.0), ep)
        w.add_scalar("imm/tank0_ct_prob",
                     info.get("tank0_imm_ct_ep", 0.0), ep)
        w.add_scalar("target/artil_relocating",
                     info.get("artil_relocating_ep", 0.0), ep)

    def log_tracking_by_type(self, ep, trP_infantry, trP_tank, trP_artillery):
        """타겟 타입별 trP — target/ 카테고리로 통합."""
        if not self.writer: return
        w = self.writer
        w.add_scalar("target/trP_infantry",  trP_infantry,  ep)
        w.add_scalar("target/trP_tank",      trP_tank,      ep)
        w.add_scalar("target/trP_artillery", trP_artillery, ep)

    def log_committed(self, ep, committed_count, trP_committed,
                      committed_stale, candidate_count=0, r_reacquire=0.0):
        """committed/candidate/stale 세부 지표 — target/ 카테고리로 통합."""
        if not self.writer: return
        w = self.writer
        w.add_scalar("target/trP_committed",   trP_committed,   ep)
        w.add_scalar("target/committed_stale", committed_stale, ep)
        w.add_scalar("target/r_reacquire",     r_reacquire,     ep)

    def log_battery_detail(self, ep, first_rtb_step):
        if not self.writer: return
        self.writer.add_scalar("battery/first_rtb_step", first_rtb_step, ep)

    def log_attn(self, ep, actor_attn, critic_attn,
                 zone_u_mean, drone_spread,
                 target_trP_mean, target_age_mean):
        """
        actor_attn / critic_attn: (NT, NT) or (H, NT, NT)
        """
        if not self.writer: return
        w = self.writer

        def bm(a, r0, r1, c0, c1):
            if a is None: return 0.0
            x = a[0] if a.dim() == 3 else a
            return float(x[r0:r1, c0:c1].mean())

        ND = N_DRONE
        NZ = N_ZONE
        NT = K_ACTIVE

        # token 통계
        w.add_scalar("token/zone_u_mean",      zone_u_mean,      ep)
        w.add_scalar("token/drone_spread",     drone_spread,     ep)
        w.add_scalar("token/target_trP_mean",  target_trP_mean,  ep)
        w.add_scalar("token/target_age_mean",  target_age_mean,  ep)

        # attention 패턴
        ND = N_DRONE_TRAIN; NZ = N_ZONE; NA = K_ACTIVE
        # actor
        w.add_scalar("attn/actor_drone_drone",
                     bm(actor_attn, 0,ND, 0,ND), ep)
        w.add_scalar("attn/actor_drone_zone",
                     bm(actor_attn, 0,ND, ND,ND+NZ), ep)
        w.add_scalar("attn/actor_drone_active",
                     bm(actor_attn, 0,ND, ND+NZ,ND+NZ+NA), ep)
        w.add_scalar("attn/actor_active_active",
                     bm(actor_attn, ND+NZ,ND+NZ+NA, ND+NZ,ND+NZ+NA), ep)
        # critic
        w.add_scalar("attn/critic_drone_drone",
                     bm(critic_attn, 0,ND, 0,ND), ep)
        w.add_scalar("attn/critic_drone_zone",
                     bm(critic_attn, 0,ND, ND,ND+NZ), ep)
        w.add_scalar("attn/critic_drone_active",
                     bm(critic_attn, 0,ND, ND+NZ,ND+NZ+NA), ep)

    def print_progress(self, ep, total_ep, ep_reward, info, loss_info, elapsed):
        tracked  = info.get("tracked_targets",      0)
        cm_ep    = info.get("tracked_committed_ep", 0.0)
        st_ep    = info.get("tracked_stale_ep",     0.0)
        t_hits   = info.get("terrain_hits_ep",      0)
        d_colls  = info.get("drone_colls_ep",       0)
        u_pct    = info.get("uncertainty_pct",      0.0)
        mean_bat = info.get("mean_battery_ep",      1.0)
        rtb_cnt  = info.get("rtb_count_ep",         0)
        bar = "█"*int(20*ep/total_ep) + "░"*(20-int(20*ep/total_ep))
        print(f"\r[{bar}] Ep {int(ep):4d}/{total_ep} | "
              f"R={ep_reward:7.1f} | "
              f"Cm={cm_ep:4.1f}/St={st_ep:4.1f} | "
              f"U={u_pct:5.1f}% | "
              f"Bat={mean_bat:.2f}(RTB:{rtb_cnt}) | "
              f"T={t_hits:3d}/C={d_colls:3d} | "
              f"CL={loss_info.get('critic_loss',0):.4f} | "
              f"AL={loss_info.get('actor_loss',0):.4f} | "
              f"Ent={loss_info.get('entropy',0):.4f} | "
              f"{elapsed:.0f}s")
        sys.stdout.flush()

    def close(self):
        if self.writer: self.writer.close()

# ============================================================
# 유틸
# ============================================================
def _patch_env_rewards(env_module, rd):
    """환경 모듈의 reward 상수를 직접 override. env_module은 import된 모듈 객체."""
    for k, v in rd.items():
        if hasattr(env_module, k): setattr(env_module, k, v)
        else: print(f"  [경고] '{k}' 없음")

def _get_latest_checkpoint(directory):
    if not os.path.isdir(directory): return None
    files = [f for f in os.listdir(directory)
             if f.startswith("graph_token_ep") and f.endswith(".pt")]
    if not files: return None
    return os.path.join(directory, sorted(files)[-1])

# ============================================================
# 에피소드 info 누적 헬퍼
# ============================================================
class EpisodeInfoAccumulator:
    """에피소드 동안 step info를 누적해서 요약."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.terrain_hits  = 0
        self.drone_colls   = 0
        self.r_track_sum   = 0.0
        self.r_stale_sum   = 0.0
        self.r_unc_sum     = 0.0
        self.rtb_count     = 0
        self.first_rtb_step= -1
        self.step_count    = 0
        self.battery_sum   = 0.0
        # 에피소드 평균 누적 (마지막 step 스냅샷 → 평균으로 변경)
        self.tracked_sum           = 0.0
        self.tracked_committed_sum = 0.0
        self.tracked_stale_sum     = 0.0
        self.candidate_sum         = 0.0
        self.mean_trp_sum          = 0.0
        self.stale_sum             = 0.0
        # v12 신규
        self.artil_relocating_sum  = 0.0
        self.tank0_imm_cv_sum      = 0.0
        self.tank0_imm_ct_sum      = 0.0

    def update(self, info, step):
        self.terrain_hits += len(info.get("terrain_hit",  []))
        self.drone_colls  += len(info.get("drone_collision", []))
        self.r_track_sum  += info.get("r_track", 0.0)
        self.r_stale_sum  += info.get("r_stale", 0.0)
        self.r_unc_sum    += info.get("r_unc",   0.0)
        new_rtb = len(info.get("returning_drones", []))
        if new_rtb > 0:
            self.rtb_count += new_rtb
            if self.first_rtb_step < 0:
                self.first_rtb_step = step
        self.battery_sum  += info.get("mean_battery", 1.0)
        # 에피소드 평균 누적
        self.tracked_sum           += info.get("tracked_targets_all",
                                               info.get("tracked_targets", 0))
        self.tracked_committed_sum += info.get("tracked_targets_committed", 0)
        self.tracked_stale_sum     += info.get("tracked_targets_stale", 0)
        self.candidate_sum         += info.get("candidate_count", 0)
        self.mean_trp_sum          += info.get("mean_tr_P", 1.0)
        self.stale_sum             += info.get("stale_count", 0)
        # v12 신규
        self.artil_relocating_sum  += info.get("artil_relocating", 0)
        self.tank0_imm_cv_sum      += info.get("tank0_imm_cv", 0.0)
        self.tank0_imm_ct_sum      += info.get("tank0_imm_ct", 0.0)
        self.step_count   += 1

    def summarize(self, last_info, nt_non_terrain):
        """에피소드 전체 평균 기반 요약 dict 반환."""
        raw_u = last_info.get("total_uncertainty", 0.0)
        u_pct = float(raw_u / max(nt_non_terrain, 1) * 100.0)
        n     = max(self.step_count, 1)
        return {
            "uncertainty_pct"    : u_pct,
            "terrain_hits_ep"    : self.terrain_hits,
            "drone_colls_ep"     : self.drone_colls,
            # 에피소드 평균 (마지막 step 스냅샷 → 평균)
            "tracked_targets"    : self.tracked_sum / n,
            "tracked_targets_ep" : self.tracked_sum / n,
            "tracked_committed_ep": self.tracked_committed_sum / n,
            "tracked_stale_ep"   : self.tracked_stale_sum / n,
            "candidate_count"    : self.candidate_sum / n,
            "stale_count"        : self.stale_sum / n,
            "stale_count_ep"     : self.stale_sum / n,
            "mean_tr_P"          : self.mean_trp_sum / n,
            "committed_count"    : self.tracked_committed_sum / n,
            "mean_battery_ep"    : self.battery_sum / n,
            "rtb_count_ep"       : self.rtb_count,
            "first_rtb_step"     : self.first_rtb_step,
            "active_at_end"      : last_info.get("active_drones", N_DRONE_TRAIN),
            "r_track_ep"         : self.r_track_sum / n,
            "r_stale_ep"         : self.r_stale_sum / n,
            "r_unc_ep"           : self.r_unc_sum   / n,
            # v12 신규
            "artil_relocating_ep" : self.artil_relocating_sum / n,
            "tank0_imm_cv_ep"     : self.tank0_imm_cv_sum / n,
            "tank0_imm_ct_ep"     : self.tank0_imm_ct_sum / n,
        }

# ============================================================
# train_stage
# ============================================================
def train_stage(stage, cur_cfg=None, train_cfg=None):
    if cur_cfg is None:   cur_cfg   = CurriculumConfig(stage=stage)
    else:                 cur_cfg.stage = stage
    if train_cfg is None: train_cfg = TrainConfig()
    train_cfg.save_dir       = cur_cfg.save_dir()
    train_cfg.total_episodes = cur_cfg.stage_episodes[stage]

    set_global_seed(train_cfg.seed)

    # seed별로 checkpoint/log가 섞이지 않게 분리
    train_cfg.save_dir = os.path.join(train_cfg.save_dir, f"seed{train_cfg.seed}")

    print("="*65)
    print(f"MAPPO v9 Entity Transformer  Stage {stage}/3")
    print(f"  seed      : {train_cfg.seed}")
    print(f"  에피소드  : {train_cfg.total_episodes}")
    print(f"  저장 경로 : {train_cfg.save_dir}")
    goal = cur_cfg.stage_goals[stage]
    if goal:
        print(f"  달성 조건 : {goal['metric']} "
              f"{'<' if goal['direction']=='below' else '>'} "
              f"{goal['threshold']} (최근 {goal['window']}ep)")
    print("="*65)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from isaac_env_v13 import DroneSwarmEnv
    print("\n[환경 초기화]")
    # K-slot sensitivity: env가 자체 default(K_ACTIVE=12)를 쓰지 않도록 명시적으로 전달
    base_env = DroneSwarmEnv(randomize_terrain=False, terrain_seed=42, k_active=K_ACTIVE)
    env      = GraphTokenObsWrapper(base_env,
                                    max_ep_steps=train_cfg.rollout_steps)

    rd = cur_cfg.stage_rewards.get(stage, {})
    if rd:
        import isaac_env_v13 as _env_mod
        print(f"\n[Stage {stage} 보상 오버라이드]")
        _patch_env_rewards(_env_mod, rd)
        for k, v in rd.items(): print(f"  {k:36s} = {v}")

    # 타겟 타입별 인덱스
    from isaac_env_v13 import TargetType
    tt = base_env.target_types
    TGT_INFANTRY_IDX  = [k for k, t in enumerate(tt) if t == TargetType.INFANTRY]
    TGT_TANK_IDX      = [k for k, t in enumerate(tt) if t == TargetType.TANK]
    TGT_ARTILLERY_IDX = [k for k, t in enumerate(tt) if t == TargetType.ARTILLERY]

    agent   = MAPPOAgent(train_cfg)
    logger  = TrainLogger(train_cfg.save_dir, stage=stage)
    acc     = EpisodeInfoAccumulator()
    start   = 0
    ck = _get_latest_checkpoint(train_cfg.save_dir)
    if ck:
        start = agent.load(ck)
        print(f"  [재개] ep {start}부터 재시작")
    else:
        print("  [처음부터 학습]")

    nt_non_terrain = int((~base_env.ds_map.terrain_mask).sum())

    ep_bar = tqdm(range(start+1, train_cfg.total_episodes+1),
                  desc=f"Graph-Token Stage{stage}", unit="ep", ncols=110)
    for ep in ep_bar:
        agent.episode  = ep
        agent.ep_prog  = (ep-1) / max(train_cfg.total_episodes-1, 1)
        t0 = time.time()
        obs484, gobs = env.reset()
        ep_r     = np.zeros(env.num_drones, dtype=np.float32)
        ep_steps = 0
        last_info= {}
        loss_info= {}
        acc.reset()

        for step in range(train_cfg.rollout_steps):
            ep_steps += 1; agent.total_steps += 1
            obs_t  = torch.FloatTensor(obs484).to(agent.dev)
            gobs_t = torch.FloatTensor(gobs).to(agent.dev)
            acts, lps, vals, _, _ = agent.get_action_and_value(obs484, gobs)
            (nobs484, ngobs), rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step)
            last_info = info
            agent.buffer.push(
                obs_t, gobs_t, acts, lps,
                torch.FloatTensor(rews).to(agent.dev),
                vals,
                torch.FloatTensor(dones.astype(np.float32)).to(agent.dev))
            ep_r += rews; obs484 = nobs484; gobs = ngobs
            if dones.all(): break

        # GAE + update
        with torch.no_grad():
            _, _, nv, _, _ = agent.get_action_and_value(obs484, gobs)
            agent.buffer.compute_gae(
                nv, torch.FloatTensor(dones).to(agent.dev),
                train_cfg.gamma, train_cfg.gae_lambda)
        loss_info = agent.update()

        # 에피소드 요약
        elapsed  = time.time() - t0
        ep_info  = acc.summarize(last_info, nt_non_terrain)
        agent.ep_rewards_history.append(float(ep_r.mean()))

        # ── eval-only: belief_tracker trP (action 결정에 미사용) ──────
        if getattr(base_env, "belief_tracker", None) is not None:
            bt = base_env.belief_tracker
            def _mtrP(idxs):
                return float(np.mean([bt.get_tr_P_norm(k) for k in idxs])) if idxs else 1.0
            ep_info["trP_infantry"]  = _mtrP(TGT_INFANTRY_IDX)
            ep_info["trP_tank"]      = _mtrP(TGT_TANK_IDX)
            ep_info["trP_artillery"] = _mtrP(TGT_ARTILLERY_IDX)

            from isaac_env_v13 import TargetState
            states = bt.target_states
            cm_ids = [k for k in range(base_env.num_targets)
                      if states[k] == TargetState.COMMITTED]
            ep_info["trP_committed"] = (
                float(np.mean([bt.get_tr_P_norm(k) for k in cm_ids]))
                if cm_ids else 1.0)

        logger.log_episode(ep, ep_steps, ep_r, ep_info, loss_info, elapsed)

        # attention 로그 (attn_log_interval마다)
        if ep % train_cfg.attn_log_interval == 0:
            with torch.no_grad():
                obs_t2  = torch.FloatTensor(obs484).to(agent.dev).unsqueeze(0)
                gobs_t2 = torch.FloatTensor(gobs).to(agent.dev).unsqueeze(0)
                _, aw  = agent.actor( obs_t2, gobs_t2, agent.ep_prog, return_attn=True)
                _, cw  = agent.critic(obs_t2, gobs_t2, agent.ep_prog, return_attn=True)
                df, zf, af, gf, pi_pos = agent.actor.tb(obs_t2, agent.ep_prog)
                zone_u_mean     = float(zf[:,:,4].mean().cpu())
                nd_actual       = pi_pos.shape[1] - N_ZONE
                ctr             = pi_pos[:,:nd_actual,:].mean(1, keepdim=True)
                drone_spread    = float(
                    torch.norm(pi_pos[:,:nd_actual,:]-ctr, dim=-1).mean().cpu())
                target_trP_mean = float(af[:,:,4].mean().cpu())
                target_age_mean = float(af[:,:,5].mean().cpu())
            logger.log_attn(ep, aw, cw, zone_u_mean, drone_spread,
                            target_trP_mean, target_age_mean)

        ep_bar.set_postfix({
            "R":   f"{float(ep_r.mean()):.1f}",
            "U":   f"{ep_info.get('uncertainty_pct', 0):.1f}%",
            "Trk": f"{ep_info.get('tracked_targets', 0):.1f}",
            "AL":  f"{loss_info.get('actor_loss', 0):.3f}",
            "t":   f"{elapsed:.0f}s",
        })
        if ep % train_cfg.save_interval == 0:
            agent.save(ep)

    agent.save(train_cfg.total_episodes)
    print(f"\nStage{stage} 완료! 총스텝:{agent.total_steps:,}  "
          f"로그:{logger.log_path}")
    logger.close()
    return agent

# ============================================================
# main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",     type=int, default=1)
    parser.add_argument("--save_dir",  type=str, default=None)
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    cur_cfg   = CurriculumConfig(stage=args.stage)
    train_cfg = TrainConfig()
    train_cfg.seed = args.seed
    if args.save_dir:
        cur_cfg.base_save_dir = args.save_dir

    train_stage(args.stage, cur_cfg=cur_cfg, train_cfg=train_cfg)

if __name__ == "__main__":
    main()
