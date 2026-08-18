"""
gat_mappo_paper_v13.py
======================
GAT 논문(Zhao et al. 2025) 방식 observation + MAPPO 학습 코드.

obs:  gat_obs_wrapper_v13.GATObsWrapper 가 생성한 graph flat tensor
      shape (num_drones, GAT_OBS_DIM=129)

backbone: PaperGATLayer — drone / target / obstacle 3타입 거리 기반 aggregation
          (논문 수식 12~21 직접 대응)

proposed 요소 완전 제외:
  - DS map / fused priority / IMM-EKF / active track 없음
  - 오직 FOV 내 직접 관측 정보만 사용

저장: 100ep 마다 checkpoint 저장 + 마지막 ep 저장.

실행:
    python gat_mappo_paper_v13.py --stage 1
    python gat_mappo_paper_v13.py --stage 1 --save_dir /path/to/repository/gat_paper
"""

from __future__ import annotations
import os, sys, time, importlib
from pathlib import Path
import numpy as np

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ── wrapper import ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gat_obs_wrapper_v13 import (
    GATObsWrapper, GAT_OBS_DIM,
    SELF_DIM, NEIGHBOR_DIM, TARGET_DIM, OBSTACLE_DIM,
    MAX_NEIGHBOR_DRONE, MAX_OBS_TARGET, MAX_OBSTACLE,
)

# ── 상수 ──────────────────────────────────────────────────────────────
ACTION_DIM    = 4
N_DRONE_TRAIN = 4
NUM_DRONES    = N_DRONE_TRAIN   # ← 추가
TOKEN_DIM     = 128

D_SELF     = SELF_DIM   # 5
D_NEIGHBOR = 4          # rel_x, rel_y, dist_norm, active
D_TARGET   = 4          # rel_x, rel_y, dist_norm, valid
D_OBSTACLE = 3          # rel_x, rel_y, dist_norm


# ═══════════════════════════════════════════════════════════════════════
# Seed 고정 (training-seed robustness 실험용)
# ═══════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    total_episodes  : int   = 3000
    rollout_steps   : int   = 3000
    token_dim       : int   = TOKEN_DIM
    num_layers      : int   = 3
    ffn_dim         : int   = 256
    lr_actor        : float = 3e-4
    lr_critic       : float = 1e-4
    gamma           : float = 0.99
    gae_lambda      : float = 0.95
    clip_param      : float = 0.2
    ppo_epochs      : int   = 4
    ppo_batch_size  : int   = 256
    entropy_coef    : float = 0.02
    vf_coef         : float = 0.5
    max_grad_norm   : float = 1.0
    save_dir        : str   = str(REPO_ROOT / "gat_paper" / "stage1")
    save_interval   : int   = 100
    attn_log_interval: int  = 50
    env_module      : str   = "isaac_env_v13"
    beta_agent      : float = 2.0
    beta_target     : float = 2.0
    beta_obstacle   : float = 2.0
    device : str = "cuda" if torch.cuda.is_available() else "cpu"
    seed : int = 0


@dataclass
class CurriculumConfig:
    stage          : int = 1
    base_save_dir  : str = str(REPO_ROOT / "gat_paper")
    stage_episodes : dict = field(default_factory=lambda: {1:1500, 2:2000, 3:2000})
    stage_rewards  : dict = field(default_factory=lambda: {
        1: {"LAMBDA_TRACK":0.3, "LAMBDA_STALE":0.01,
            "REWARD_UNCERTAINTY_REDUCE":1.5,
            "PENALTY_TERRAIN":-0.05,
            "PENALTY_DRONE_COLLISION":-1.0, "ANTIAIR_KILL_PENALTY":0.0},
        2: {"LAMBDA_TRACK":0.5, "LAMBDA_STALE":0.02,
            "REWARD_UNCERTAINTY_REDUCE":1.0,
            "PENALTY_TERRAIN":-0.3,
            "PENALTY_DRONE_COLLISION":-1.0, "ANTIAIR_KILL_PENALTY":0.0},
        3: {"LAMBDA_TRACK":0.5, "LAMBDA_STALE":0.02,
            "REWARD_UNCERTAINTY_REDUCE":1.0,
            "PENALTY_TERRAIN":-0.1,
            "PENALTY_DRONE_COLLISION":-1.0, "ANTIAIR_KILL_PENALTY":0.0},
    })

    def save_dir(self):
        return os.path.join(self.base_save_dir, f"stage{self.stage}")

    def prev_stage_dir(self):
        return None if self.stage <= 1 else \
               os.path.join(self.base_save_dir, f"stage{self.stage-1}")


# ═══════════════════════════════════════════════════════════════════════
# Graph obs 파싱
# ═══════════════════════════════════════════════════════════════════════

def parse_gat_obs(obs: torch.Tensor):
    """
    flat GAT obs → 구조화된 텐서.
    obs: (B, N, GAT_OBS_DIM)
    """
    B, N, _ = obs.shape
    ptr = 0

    self_f = obs[..., ptr:ptr+SELF_DIM]; ptr += SELF_DIM

    nbr_f = obs[..., ptr:ptr+NEIGHBOR_DIM].reshape(B, N, MAX_NEIGHBOR_DRONE, D_NEIGHBOR)
    ptr  += NEIGHBOR_DIM

    tgt_f = obs[..., ptr:ptr+TARGET_DIM].reshape(B, N, MAX_OBS_TARGET, D_TARGET)
    ptr  += TARGET_DIM

    obs_f = obs[..., ptr:ptr+OBSTACLE_DIM].reshape(B, N, MAX_OBSTACLE, D_OBSTACLE)

    # 상대 위치 (첫 2차원)
    nbr_pos = nbr_f[..., :2]
    tgt_pos = tgt_f[..., :2]
    obs_pos = obs_f[..., :2]

    # valid mask
    nbr_mask = nbr_f[..., -1]                             # (B,N,K_nbr)
    tgt_mask = tgt_f[..., -1]                             # (B,N,K_tgt)
    obs_mask = (obs_f.norm(dim=-1) > 1e-6).float()        # (B,N,K_obs)

    return self_f, nbr_f, tgt_f, obs_f, nbr_pos, tgt_pos, obs_pos, \
           nbr_mask, tgt_mask, obs_mask


# ═══════════════════════════════════════════════════════════════════════
# PaperGATLayer  — 논문 수식 (12)~(21)
# ═══════════════════════════════════════════════════════════════════════

class PaperGATLayer(nn.Module):
    """
    논문 수식 (12)~(21) 직접 구현.

    3타입 aggregation:
        h_agent   = Σ α_a · W_a · h_nbr       (논문 수식 18)
        h_target  = Σ α_t · W_t · h_tgt       (논문 수식 20)
        h_obstacle= Σ α_o · W_o · h_obs       (논문 수식 19)

    output (수식 21):
        h' = ReLU([h_self || h_agent || h_target || h_obstacle] W) + residual
    """

    def __init__(self, d: int, ffn_dim: int,
                 beta_agent: float, beta_target: float, beta_obstacle: float):
        super().__init__()
        self.beta_a = beta_agent
        self.beta_t = beta_target
        self.beta_o = beta_obstacle

        self.W_agent    = nn.Linear(d, d, bias=False)
        self.W_target   = nn.Linear(d, d, bias=False)
        self.W_obstacle = nn.Linear(d, d, bias=False)

        self.out_proj = nn.Sequential(
            nn.Linear(d * 4, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d),
        )
        self.norm = nn.LayerNorm(d)

    @staticmethod
    def _aggregate(rel_pos: torch.Tensor,
                   value:   torch.Tensor,
                   mask:    torch.Tensor,
                   beta:    float) -> torch.Tensor:
        """
        rel_pos: (B,N,K,2)  relative position (== distance direction)
        value:   (B,N,K,D)
        mask:    (B,N,K)    1=valid
        """
        dist  = rel_pos.norm(dim=-1).clamp(min=1e-6)     # (B,N,K)
        score = torch.exp(-beta * dist) * mask.float()
        denom = score.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha = score / denom                              # (B,N,K)
        return (alpha.unsqueeze(-1) * value).sum(dim=2)   # (B,N,D)

    def forward(self,
                drone_h:  torch.Tensor,    # (B,N,D)
                nbr_h:    torch.Tensor,    # (B,N,K_nbr,D)
                tgt_h:    torch.Tensor,    # (B,N,K_tgt,D)
                obs_h:    torch.Tensor,    # (B,N,K_obs,D)
                nbr_pos:  torch.Tensor,    # (B,N,K_nbr,2)
                tgt_pos:  torch.Tensor,    # (B,N,K_tgt,2)
                obs_pos:  torch.Tensor,    # (B,N,K_obs,2)
                nbr_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                obs_mask: torch.Tensor,
                ) -> torch.Tensor:

        h_agent  = self._aggregate(nbr_pos, self.W_agent(nbr_h),    nbr_mask, self.beta_a)
        h_target = self._aggregate(tgt_pos, self.W_target(tgt_h),   tgt_mask, self.beta_t)
        h_obs    = self._aggregate(obs_pos, self.W_obstacle(obs_h),  obs_mask, self.beta_o)

        cat   = torch.cat([drone_h, h_agent, h_target, h_obs], dim=-1)  # (B,N,4D)
        h_new = self.out_proj(cat)
        return self.norm(drone_h + h_new)


# ═══════════════════════════════════════════════════════════════════════
# PaperGATBackbone
# ═══════════════════════════════════════════════════════════════════════

class PaperGATBackbone(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        D = cfg.token_dim

        self.self_proj     = nn.Linear(D_SELF,     D)
        self.neighbor_proj = nn.Linear(D_NEIGHBOR, D)
        self.target_proj   = nn.Linear(D_TARGET,   D)
        self.obstacle_proj = nn.Linear(D_OBSTACLE, D)

        self.layers = nn.ModuleList([
            PaperGATLayer(D, cfg.ffn_dim,
                          cfg.beta_agent, cfg.beta_target, cfg.beta_obstacle)
            for _ in range(cfg.num_layers)
        ])

        # critic용 global feature
        self.global_proj = nn.Linear(D * 4, D)

    def forward(self, obs: torch.Tensor):
        """
        obs: (B, N, GAT_OBS_DIM)
        Returns:
            drone_h:  (B, N, D)   actor용
            global_h: (B, D)      critic용
        """
        (self_f, nbr_f, tgt_f, obs_f,
         nbr_pos, tgt_pos, obs_pos,
         nbr_mask, tgt_mask, obs_mask) = parse_gat_obs(obs)

        drone_h = self.self_proj(self_f)
        nbr_h   = self.neighbor_proj(nbr_f)
        tgt_h   = self.target_proj(tgt_f)
        obs_h   = self.obstacle_proj(obs_f)

        for layer in self.layers:
            drone_h = layer(drone_h, nbr_h, tgt_h, obs_h,
                            nbr_pos, tgt_pos, obs_pos,
                            nbr_mask, tgt_mask, obs_mask)

        # global pooling for critic
        drone_pool = drone_h.mean(1)

        def _masked_pool(h, mask):
            cnt = mask.sum(-1, keepdim=True).clamp(min=1)
            return (h * mask.unsqueeze(-1)).sum(2) / cnt   # (B,N,D)

        nbr_pool = _masked_pool(nbr_h, nbr_mask).mean(1)
        tgt_pool = _masked_pool(tgt_h, tgt_mask).mean(1)
        obs_pool = _masked_pool(obs_h, obs_mask).mean(1)

        global_h = self.global_proj(
            torch.cat([drone_pool, nbr_pool, tgt_pool, obs_pool], dim=-1))

        return drone_h, global_h


# ═══════════════════════════════════════════════════════════════════════
# Actor / Critic
# ═══════════════════════════════════════════════════════════════════════

class GATActor(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.backbone = PaperGATBackbone(cfg)
        D = cfg.token_dim
        self.head = nn.Sequential(nn.Linear(D,D), nn.ReLU(), nn.Linear(D, ACTION_DIM))

    def forward(self, obs: torch.Tensor):
        drone_h, _ = self.backbone(obs)
        return self.head(drone_h)   # (B, N, ACTION_DIM)


class GATCritic(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.backbone = PaperGATBackbone(cfg)
        D = cfg.token_dim
        self.head = nn.Sequential(nn.Linear(D,D), nn.ReLU(), nn.Linear(D, 1))

    def forward(self, obs: torch.Tensor):
        _, global_h = self.backbone(obs)            # (B, D)
        v_team = self.head(global_h).squeeze(-1)    # (B,)
        N = obs.shape[1]
        return v_team.unsqueeze(-1).expand(-1, N)   # (B, N)


# ═══════════════════════════════════════════════════════════════════════
# RolloutBuffer
# ═══════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self, capacity, num_drones, obs_dim, device="cpu"):
        self.capacity = capacity
        self.device   = device
        self.pos      = 0
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs        = z(capacity, num_drones, obs_dim)
        self.actions    = torch.zeros(capacity, num_drones, dtype=torch.int64, device=device)
        self.logprobs   = z(capacity, num_drones)
        self.rewards    = z(capacity, num_drones)
        self.values     = z(capacity, num_drones)
        self.dones      = z(capacity, num_drones)
        self.returns    = z(capacity, num_drones)
        self.advantages = z(capacity, num_drones)

    def push(self, obs, actions, logprobs, rewards, values, dones):
        p = self.pos
        self.obs[p]      = obs
        self.actions[p]  = actions
        self.logprobs[p] = logprobs
        self.rewards[p]  = rewards
        self.values[p]   = values
        self.dones[p]    = dones
        self.pos += 1

    def compute_gae(self, next_value, next_done, gamma, lam):
        gae = 0
        for t in reversed(range(self.pos)):
            nt    = 1.0 - (next_done if t == self.pos-1 else self.dones[t+1])
            nv    =        next_value if t == self.pos-1 else self.values[t+1]
            delta = self.rewards[t] + gamma * nv * nt - self.values[t]
            gae   = delta + gamma * lam * nt * gae
            self.advantages[t] = gae
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

    def get_generator(self, batch_size):
        idx = np.random.permutation(self.pos)
        for s in range(0, self.pos, batch_size):
            i = idx[s:s+batch_size]
            yield (self.obs[i], self.actions[i], self.logprobs[i],
                   self.returns[i], self.advantages[i])

    def clear(self): self.pos = 0


# ═══════════════════════════════════════════════════════════════════════
# GATMAPPOAgent
# ═══════════════════════════════════════════════════════════════════════

class GATMAPPOAgent:
    def __init__(self, cfg: TrainConfig, num_drones: int):
        self.cfg         = cfg
        self.dev         = torch.device(cfg.device)
        self.num_drones  = num_drones
        self.actor       = GATActor(cfg).to(self.dev)
        self.critic      = GATCritic(cfg).to(self.dev)
        self.a_opt       = optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.c_opt       = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.buffer      = RolloutBuffer(
            cfg.rollout_steps + 10, num_drones, GAT_OBS_DIM, device=self.dev)
        self.episode     = 0
        self.total_steps = 0

        ap = sum(p.numel() for p in self.actor.parameters())
        cp = sum(p.numel() for p in self.critic.parameters())
        print("GATMAPPOAgent [논문식 GAT / paper-faithful] 초기화 완료")
        print(f"  obs_dim={GAT_OBS_DIM}  Actor={ap:,}  Critic={cp:,}  device={self.dev}")
        print(f"  노드: self({D_SELF}) + neighbor({MAX_NEIGHBOR_DRONE}x{D_NEIGHBOR})"
              f" + target({MAX_OBS_TARGET}x{D_TARGET}) + obstacle({MAX_OBSTACLE}x{D_OBSTACLE})")

    @torch.no_grad()
    def get_action_and_value(self, obs: np.ndarray, deterministic=False):
        obs_t = torch.FloatTensor(obs).to(self.dev).unsqueeze(0)  # (1,N,D)
        N     = obs.shape[0]

        logits = self.actor(obs_t)    # (1,N,A)
        values = self.critic(obs_t)   # (1,N)

        actions  = torch.zeros(N, dtype=torch.int64,   device=self.dev)
        logprobs = torch.zeros(N, dtype=torch.float32, device=self.dev)
        for i in range(N):
            dist = Categorical(logits=logits[:, i, :])
            a    = logits[:, i, :].argmax(-1) if deterministic else dist.sample()
            actions[i]  = a.squeeze()
            logprobs[i] = dist.log_prob(a).squeeze()

        return actions, logprobs, values.squeeze(0)

    def update(self) -> dict:
        cfg = self.cfg
        buf = self.buffer
        ta = tc = te = tf = nu = 0.0
        total_cg = total_ag = 0.0
        diag_adv = float(buf.advantages[:buf.pos].std()) if buf.pos > 0 else 0.0

        for _ in range(cfg.ppo_epochs):
            for mb_obs, mb_act, mb_lp, mb_ret, mb_adv in \
                    buf.get_generator(cfg.ppo_batch_size):
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # critic
                V  = self.critic(mb_obs)
                cl = cfg.vf_coef * F.mse_loss(V, mb_ret)
                self.c_opt.zero_grad(); cl.backward()
                cg = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.c_opt.step(); total_cg += float(cg)

                # actor
                logits = self.actor(mb_obs)
                n_d    = mb_obs.shape[1]
                al     = torch.tensor(0.0, device=self.dev)
                ent = cf = 0.0
                for i in range(n_d):
                    d  = Categorical(logits=logits[:, i, :])
                    lp = d.log_prob(mb_act[:, i])
                    ent += d.entropy().mean().item()
                    r  = torch.exp(lp - mb_lp[:, i])
                    s1 = r * mb_adv[:, i]
                    s2 = torch.clamp(r, 1-cfg.clip_param, 1+cfg.clip_param) * mb_adv[:, i]
                    al = al + (-torch.min(s1,s2).mean() - cfg.entropy_coef*d.entropy().mean())
                    cf += ((r-1).abs() > cfg.clip_param).float().mean().item()
                al = al / max(n_d, 1)
                self.a_opt.zero_grad(); al.backward()
                ag = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.a_opt.step(); total_ag += float(ag)

                ta += al.item(); tc += cl.item()
                te += ent/max(n_d,1); tf += cf/max(n_d,1); nu += 1

        buf.clear()
        n = max(nu, 1)
        return dict(actor_loss=ta/n, critic_loss=tc/n, entropy=te/n,
                    clip_frac=tf/n, critic_grad_norm=total_cg/n,
                    actor_grad_norm=total_ag/n, diag_adv_std=diag_adv)

    def save(self, ep: int) -> str:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        p = os.path.join(self.cfg.save_dir, f"gat_paper_ep{ep:05d}.pt")
        torch.save(dict(episode=ep, actor=self.actor.state_dict(),
                        critic=self.critic.state_dict(),
                        total_steps=self.total_steps,
                        cfg=self.cfg.__dict__), p)
        print(f"  [저장] {p}")
        return p

    def load(self, path: str) -> int:
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ck["actor"],   strict=True)
        self.critic.load_state_dict(ck["critic"], strict=True)
        self.total_steps = ck.get("total_steps", 0)
        ep = ck.get("episode", 0)
        print(f"  [로드] {path}  (ep={ep})")
        return ep


# ═══════════════════════════════════════════════════════════════════════
# TrainLogger
# ═══════════════════════════════════════════════════════════════════════

class TrainLogger:
    def __init__(self, save_dir: str, stage: int = 1):
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(save_dir, "train_log_gat_paper.csv")
        with open(self.log_path, "w") as f:
            f.write("episode,steps,reward_mean,reward_sum,uncertainty_pct,"
                    "tracked_targets,mean_trP,stale_count,committed_count,"
                    "mean_battery,rtb_count,active_at_end,depleted_ep,"
                    "r_track,r_stale,r_unc,"
                    "critic_loss,actor_loss,entropy,clip_frac,"
                    "critic_grad_norm,actor_grad_norm,adv_std,time_s\n")
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = os.path.join(save_dir, "tb_logs", f"gat_paper_stage{stage}")
            self.writer = SummaryWriter(log_dir=tb)
            print(f"  [TensorBoard] tensorboard --logdir {tb}")
        except ImportError:
            self.writer = None

    def log_episode(self, ep, steps, rewards, ep_info, loss_info, elapsed):
        r_mean      = float(np.mean(rewards))
        r_sum       = float(np.sum(rewards))
        u_pct       = ep_info.get("uncertainty_pct",       0.0)
        cov         = max(0.0, 1.0 - u_pct / 100.0)
        t_hits      = ep_info.get("terrain_hits_ep",       0)
        d_colls     = ep_info.get("drone_colls_ep",        0)
        tracked     = ep_info.get("tracked_targets",       0.0)
        tracked_cm  = ep_info.get("tracked_committed_ep",  0.0)
        tracked_st  = ep_info.get("tracked_stale_ep",      0.0)
        candidate   = ep_info.get("candidate_count",       0.0)
        mean_trP    = ep_info.get("mean_tr_P",              1.0)
        trP_cm      = ep_info.get("trP_committed",          1.0)
        stale       = ep_info.get("stale_count",            0.0)
        committed   = ep_info.get("committed_count",        0.0)
        trP_inf     = ep_info.get("trP_infantry",           1.0)
        trP_tnk     = ep_info.get("trP_tank",               1.0)
        trP_art     = ep_info.get("trP_artillery",          1.0)
        bat         = ep_info.get("mean_battery_ep",        1.0)
        rtb         = ep_info.get("rtb_count_ep",           0)
        first_rtb   = ep_info.get("first_rtb_step",        -1)
        active      = ep_info.get("active_at_end",          NUM_DRONES)
        depleted    = ep_info.get("depleted_ep",             0)
        r_track     = ep_info.get("r_track_ep",             0.0)
        r_stale     = ep_info.get("r_stale_ep",             0.0)
        r_unc       = ep_info.get("r_unc_ep",               0.0)
        r_reacq     = ep_info.get("r_reacquire_ep",         0.0)
        artil_rel   = ep_info.get("artil_relocating_ep",    0.0)
        imm_cv      = ep_info.get("tank0_imm_cv_ep",        0.0)
        imm_ct      = ep_info.get("tank0_imm_ct_ep",        0.0)
        cm_stale    = ep_info.get("committed_stale",        0.0)

        with open(self.log_path, "a") as f:
            f.write(
                f"{ep},{steps},{r_mean:.4f},{r_sum:.4f},"
                f"{u_pct:.1f},{cov:.4f},{t_hits},{d_colls},"
                f"{tracked:.2f},{tracked_cm:.2f},{tracked_st:.2f},{candidate:.2f},"
                f"{mean_trP:.4f},{trP_cm:.4f},{stale:.2f},{committed:.2f},"
                f"{trP_inf:.4f},{trP_tnk:.4f},{trP_art:.4f},"
                f"{bat:.4f},{rtb},{first_rtb},{active},{depleted},"
                f"{r_track:.4f},{r_stale:.4f},{r_unc:.4f},{r_reacq:.4f},"
                f"{artil_rel:.4f},{imm_cv:.4f},{imm_ct:.4f},"
                f"{loss_info.get('critic_loss',0):.6f},"
                f"{loss_info.get('actor_loss',0):.6f},"
                f"{loss_info.get('entropy',0):.4f},"
                f"{loss_info.get('clip_frac',0):.4f},"
                f"{loss_info.get('critic_grad_norm',0):.4f},"
                f"{loss_info.get('actor_grad_norm',0):.4f},"
                f"{loss_info.get('diag_adv_std',0):.4f},"
                f"{loss_info.get('diag_ret_std',0):.4f},"
                f"{loss_info.get('diag_val_std',0):.4f},{elapsed:.1f}\n"
            )

        if not self.writer: return
        w = self.writer

        # ── reward ───────────────────────────────────
        w.add_scalar("reward/mean",  r_mean, ep)
        w.add_scalar("reward/sum",   r_sum,  ep)

        # ── safety ───────────────────────────────────
        w.add_scalar("safety/drone_colls",   d_colls,  ep)
        w.add_scalar("safety/terrain_hits",  t_hits,   ep)
        w.add_scalar("safety/mean_battery",  bat,      ep)
        w.add_scalar("safety/rtb_count",     rtb,      ep)
        w.add_scalar("safety/active_at_end", active,   ep)
        w.add_scalar("safety/depleted_ep",   depleted, ep)

        # ── search ───────────────────────────────────
        w.add_scalar("search/uncertainty_pct", u_pct, ep)
        w.add_scalar("search/coverage_rate",   cov,   ep)
        w.add_scalar("search/r_unc_mean",      r_unc, ep)

        # ── target ───────────────────────────────────
        w.add_scalar("target/tracked_targets",      tracked,   ep)
        w.add_scalar("target/tracked_committed_ep", tracked_cm,ep)
        w.add_scalar("target/tracked_stale_ep",     tracked_st,ep)
        w.add_scalar("target/candidate_count",      candidate, ep)
        w.add_scalar("target/mean_trP",             mean_trP,  ep)
        w.add_scalar("target/trP_committed",        trP_cm,    ep)
        w.add_scalar("target/stale_count",          stale,     ep)
        w.add_scalar("target/committed_count",      committed, ep)
        w.add_scalar("target/committed_stale",      cm_stale,  ep)
        w.add_scalar("target/trP_infantry",         trP_inf,   ep)
        w.add_scalar("target/trP_tank",             trP_tnk,   ep)
        w.add_scalar("target/trP_artillery",        trP_art,   ep)
        w.add_scalar("target/r_track_mean",         r_track,   ep)
        w.add_scalar("target/r_stale_mean",         r_stale,   ep)
        w.add_scalar("target/r_reacquire_mean",     r_reacq,   ep)
        w.add_scalar("target/artil_relocating",     artil_rel, ep)

        # ── IMM ──────────────────────────────────────
        w.add_scalar("imm/tank0_cv_prob", imm_cv, ep)
        w.add_scalar("imm/tank0_ct_prob", imm_ct, ep)

        # ── battery detail ───────────────────────────
        w.add_scalar("battery/first_rtb_step",
                     float(first_rtb) if first_rtb >= 0 else float(steps), ep)

        # ── train ────────────────────────────────────
        w.add_scalar("train/critic_loss",      loss_info.get("critic_loss",      0), ep)
        w.add_scalar("train/actor_loss",       loss_info.get("actor_loss",       0), ep)
        w.add_scalar("train/entropy",          loss_info.get("entropy",          0), ep)
        w.add_scalar("train/clip_frac",        loss_info.get("clip_frac",        0), ep)
        w.add_scalar("train/critic_grad_norm", loss_info.get("critic_grad_norm", 0), ep)
        w.add_scalar("train/actor_grad_norm",  loss_info.get("actor_grad_norm",  0), ep)
        w.add_scalar("train/steps_per_ep",     float(steps),   ep)
        w.add_scalar("train/time_s",           float(elapsed), ep)

        # ── diag ─────────────────────────────────────
        w.add_scalar("diag/adv_std", loss_info.get("diag_adv_std", 0), ep)
        w.add_scalar("diag/ret_std", loss_info.get("diag_ret_std", 0), ep)
        w.add_scalar("diag/val_std", loss_info.get("diag_val_std", 0), ep)

    def log_attn(self, ep, actor_attn, critic_attn,
                 zone_u_mean=0.0, drone_spread=0.0,
                 target_trP_mean=1.0, target_age_mean=0.0):
        """token 통계 + attention 패턴 (token_ppo_v3 동일 키)."""
        if not self.writer: return
        w = self.writer

        # ── token 통계 ────────────────────────────────
        w.add_scalar("token/zone_u_mean",     zone_u_mean,     ep)
        w.add_scalar("token/drone_spread",    drone_spread,    ep)
        w.add_scalar("token/target_trP_mean", target_trP_mean, ep)
        w.add_scalar("token/target_age_mean", target_age_mean, ep)

        # ── attention 패턴 ────────────────────────────
        if actor_attn is not None and isinstance(actor_attn, dict):
            for key, val in actor_attn.items():
                if val is not None:
                    w.add_scalar(f"attn/actor_{key}_mean", float(val.mean()), ep)
        if critic_attn is not None and isinstance(critic_attn, dict):
            for key, val in critic_attn.items():
                if val is not None:
                    w.add_scalar(f"attn/critic_{key}_mean", float(val.mean()), ep)

    def close(self):
        if self.writer: self.writer.close()



# ═══════════════════════════════════════════════════════════════════════
# EpisodeInfoAccumulator
# ═══════════════════════════════════════════════════════════════════════

class EpisodeInfoAccumulator:
    def __init__(self): self.reset()

    def reset(self):
        self.terrain_hits  = 0
        self.drone_colls   = 0
        self.battery_sum   = 0.0
        self.rtb_count     = 0
        self.first_rtb_step= -1
        self.depleted_sum  = 0
        self.tracked_sum             = 0.0
        self.tracked_committed_sum   = 0.0
        self.tracked_stale_sum       = 0.0
        self.candidate_sum           = 0.0
        self.committed_sum           = 0.0
        self.stale_sum               = 0.0
        self.mean_trp_sum            = 0.0
        self.r_track_sum   = 0.0
        self.r_stale_sum   = 0.0
        self.r_unc_sum     = 0.0
        self.r_reacq_sum   = 0.0
        self.artil_reloc_sum = 0.0
        self.tank0_imm_cv_sum= 0.0
        self.tank0_imm_ct_sum= 0.0
        self.step_count    = 0

    def update(self, info: dict, step: int):
        self.terrain_hits += len(info.get("terrain_hit",      []))
        self.drone_colls  += len(info.get("drone_collision",  []))
        self.battery_sum  += info.get("mean_battery",         1.0)
        new_rtb = len(info.get("returning_drones", []))
        if new_rtb > 0:
            self.rtb_count += new_rtb
            if self.first_rtb_step < 0:
                self.first_rtb_step = step
        self.depleted_sum += len(info.get("depleted_drones",  []))
        self.tracked_sum           += info.get("tracked_targets_all",
                                               info.get("tracked_targets", 0))
        self.tracked_committed_sum += info.get("tracked_targets_committed", 0)
        self.tracked_stale_sum     += info.get("tracked_targets_stale",    0)
        self.candidate_sum         += info.get("candidate_count",          0)
        self.committed_sum         += info.get("committed_count",          0)
        self.stale_sum             += info.get("stale_count",              0)
        self.mean_trp_sum          += info.get("mean_tr_P",                1.0)
        self.r_track_sum  += info.get("r_track",    0.0)
        self.r_stale_sum  += info.get("r_stale",    0.0)
        self.r_unc_sum    += info.get("r_unc",      0.0)
        self.r_reacq_sum  += info.get("r_reacquire",0.0)
        self.artil_reloc_sum  += info.get("artil_relocating", 0)
        self.tank0_imm_cv_sum += info.get("tank0_imm_cv",     0.0)
        self.tank0_imm_ct_sum += info.get("tank0_imm_ct",     0.0)
        self.step_count   += 1

    def summarize(self, last_info: dict, nt_non_terrain: int) -> dict:
        raw_u = last_info.get("total_uncertainty", 0.0)
        u_pct = float(raw_u / max(nt_non_terrain, 1) * 100.0)
        n     = max(self.step_count, 1)
        return {
            "uncertainty_pct"      : u_pct,
            "terrain_hits_ep"      : self.terrain_hits,
            "drone_colls_ep"       : self.drone_colls,
            "mean_battery_ep"      : self.battery_sum   / n,
            "rtb_count_ep"         : self.rtb_count,
            "first_rtb_step"       : self.first_rtb_step,
            "depleted_ep"          : self.depleted_sum,
            "tracked_targets"      : self.tracked_sum            / n,
            "tracked_committed_ep" : self.tracked_committed_sum  / n,
            "tracked_stale_ep"     : self.tracked_stale_sum      / n,
            "candidate_count"      : self.candidate_sum          / n,
            "committed_count"      : self.committed_sum          / n,
            "stale_count"          : self.stale_sum              / n,
            "mean_tr_P"            : self.mean_trp_sum           / n,
            "r_track_ep"           : self.r_track_sum            / n,
            "r_stale_ep"           : self.r_stale_sum            / n,
            "r_unc_ep"             : self.r_unc_sum               / n,
            "r_reacquire_ep"       : self.r_reacq_sum            / n,
            "artil_relocating_ep"  : self.artil_reloc_sum        / n,
            "tank0_imm_cv_ep"      : self.tank0_imm_cv_sum       / n,
            "tank0_imm_ct_ep"      : self.tank0_imm_ct_sum       / n,
            "active_at_end"        : last_info.get("active_drones", N_DRONE_TRAIN),
        }


# ═══════════════════════════════════════════════════════════════════════
# 유틸
# ═══════════════════════════════════════════════════════════════════════

def import_env_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        if name == "isaac_env_v13":
            return importlib.import_module("isaac_env_v12")
        raise

def _patch_env_rewards(env_module, rd: dict):
    for k, v in rd.items():
        if hasattr(env_module, k): setattr(env_module, k, v)
        else: print(f"  [경고] env module에 '{k}' 없음")

def _get_latest_checkpoint(save_dir: str):
    if not os.path.isdir(save_dir): return None
    pts = [f for f in os.listdir(save_dir)
           if f.startswith("gat_paper_ep") and f.endswith(".pt")]
    if not pts: return None
    def _ep(fname):
        try: return int(fname.replace("gat_paper_ep","").replace(".pt",""))
        except: return -1
    pts.sort(key=_ep)
    return os.path.join(save_dir, pts[-1])

def _target_type_indices(env, env_module):
    TT = getattr(env_module, "TargetType", None)
    if TT is None or not hasattr(env.env, "target_types"):
        return [], [], []
    inf_val = getattr(TT, "INFANTRY",  0)
    tnk_val = getattr(TT, "TANK",      2)
    art_val = getattr(TT, "ARTILLERY", 1)
    tt = env.env.target_types
    return (
        [k for k,t in enumerate(tt) if t == inf_val],
        [k for k,t in enumerate(tt) if t == tnk_val],
        [k for k,t in enumerate(tt) if t == art_val],
    )


# ═══════════════════════════════════════════════════════════════════════
# 학습 루프
# ═══════════════════════════════════════════════════════════════════════

def train_stage(stage: int, cur_cfg=None, train_cfg=None):
    if cur_cfg   is None: cur_cfg   = CurriculumConfig(stage=stage)
    else: cur_cfg.stage = stage
    if train_cfg is None: train_cfg = TrainConfig()
    train_cfg.save_dir       = cur_cfg.save_dir()
    train_cfg.total_episodes = cur_cfg.stage_episodes[stage]

    set_global_seed(train_cfg.seed)

    # seed별로 checkpoint/log가 섞이지 않게 분리
    train_cfg.save_dir = os.path.join(train_cfg.save_dir, f"seed{train_cfg.seed}")

    print("=" * 70)
    print(f"GAT-MAPPO Paper  Stage {stage}")
    print(f"  seed   : {train_cfg.seed}")
    print(f"  env    : {train_cfg.env_module}")
    print(f"  obs    : GAT graph obs ({GAT_OBS_DIM} dim) - no DS/EKF/fused")
    print(f"  저장   : {train_cfg.save_dir}  (매 {train_cfg.save_interval}ep)")
    print("=" * 70)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    env_module    = import_env_module(train_cfg.env_module)
    DroneSwarmEnv = getattr(env_module, "DroneSwarmEnv")

    base_env = DroneSwarmEnv(randomize_terrain=False, terrain_seed=42)
    env      = GATObsWrapper(base_env, max_ep_steps=train_cfg.rollout_steps)

    rd = cur_cfg.stage_rewards.get(stage, {})
    if rd:
        print(f"\n[Stage {stage} reward override]")
        _patch_env_rewards(env_module, rd)
        for k, v in rd.items(): print(f"  {k:36s} = {v}")

    TGT_INF, TGT_TNK, TGT_ART = _target_type_indices(env, env_module)
    agent  = GATMAPPOAgent(train_cfg, env.num_drones)
    logger = TrainLogger(train_cfg.save_dir, stage=stage)
    acc    = EpisodeInfoAccumulator()
    nt_non_terrain = int((~base_env.ds_map.terrain_mask).sum()) \
                     if hasattr(base_env, 'ds_map') else 1

    start = 0
    ck    = _get_latest_checkpoint(train_cfg.save_dir)
    if ck:
        start = agent.load(ck)
        print(f"  [resume] ep {start} 부터")
    else:
        print("  [new training]")

    ep_bar = tqdm(range(start+1, train_cfg.total_episodes+1),
                  desc=f"GAT-Paper Stage{stage}", unit="ep", ncols=120)

    for ep in ep_bar:
        agent.episode = ep
        t0    = time.time()
        obs   = env.reset()
        ep_r  = np.zeros(env.num_drones, dtype=np.float32)
        last_info = {}
        acc.reset()
        dones = np.zeros(env.num_drones, dtype=np.float32)

        for step in range(train_cfg.rollout_steps):
            agent.total_steps += 1
            obs_t = torch.FloatTensor(obs).to(agent.dev)

            acts, lps, vals = agent.get_action_and_value(obs)
            nobs, rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step)
            last_info = info
            agent.buffer.push(
                obs_t, acts, lps,
                torch.FloatTensor(rews).to(agent.dev),
                vals,
                torch.FloatTensor(dones).to(agent.dev),
            )
            ep_r += rews
            obs   = nobs
            if np.asarray(dones).all(): break

        with torch.no_grad():
            _, _, nv = agent.get_action_and_value(obs)
            agent.buffer.compute_gae(
                nv,
                torch.FloatTensor(dones).to(agent.dev),
                train_cfg.gamma, train_cfg.gae_lambda,
            )

        loss_info = agent.update()
        elapsed   = time.time() - t0
        ep_info   = acc.summarize(last_info, nt_non_terrain)

        # GAT-paper는 별도 attention dict를 반환하지 않으므로 dummy 처리
        actor_attn = None
        critic_attn = None

        # eval-only: belief_tracker로 trP 기록 (action 결정에 미사용)
        if getattr(base_env, "belief_tracker", None) is not None:
            def _mtrP(idxs):
                return float(np.mean(
                    [base_env.belief_tracker.get_tr_P_norm(k) for k in idxs]
                )) if idxs else 1.0
            ep_info["trP_infantry"]  = _mtrP(TGT_INF)
            ep_info["trP_tank"]      = _mtrP(TGT_TNK)

            # committed / stale trP (eval-only)
            from isaac_env_v13 import TargetState as _TS
            bt = base_env.belief_tracker
            cm_ids = [k for k in range(base_env.num_targets)
                      if bt.target_states[k] == _TS.COMMITTED]
            ep_info["trP_committed"] = (
                float(np.mean([bt.get_tr_P_norm(k) for k in cm_ids]))
                if cm_ids else 1.0)
            st_ids = [k for k in range(base_env.num_targets)
                      if bt.target_states[k] == _TS.STALE]
            ep_info["committed_stale"] = float(len(st_ids))

        logger.log_episode(ep, step+1, ep_r, ep_info, loss_info, elapsed)

        # ── 세부 attn 로그 (attn_log_interval 마다) ─────────────────
        # if ep % train_cfg.attn_log_interval == 0 and actor_attn is not None:
        #    zone_u_mean     = ep_info.get("uncertainty_pct", 0.0) / 100.0
        #    drone_spread    = float(np.std(
        #        base_env.drone_positions[:, :2], axis=0).mean())                 if hasattr(base_env, "drone_positions") else 0.0
        #    target_trP_mean = float(ep_info.get("mean_tr_P", 1.0))
        #    target_age_mean = float(ep_info.get("tracked_stale_ep", 0.0))
        #    logger.log_attn(ep, actor_attn, critic_attn,
        #                    zone_u_mean, drone_spread,
        #                    target_trP_mean, target_age_mean)

        ep_bar.set_postfix({
            "R":   f"{float(ep_r.mean()):.1f}",
            "U":   f"{ep_info.get('uncertainty_pct',0):.1f}%",
            "Trk": f"{ep_info.get('tracked_targets',0):.1f}",
            "AL":  f"{loss_info.get('actor_loss',0):.3f}",
        })

        # 100ep마다 저장
        if ep % train_cfg.save_interval == 0:
            agent.save(ep)

    # 마지막 ep 저장
    agent.save(train_cfg.total_episodes)
    logger.close()
    print(f"\nGAT-Paper Stage{stage} 완료.  total_steps={agent.total_steps:,}")
    return agent


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      type=int, default=1)
    parser.add_argument("--save_dir",   type=str, default=None)
    parser.add_argument("--env_module", type=str, default="isaac_env_v13")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    cur_cfg   = CurriculumConfig(stage=args.stage)
    if args.save_dir:
        cur_cfg.base_save_dir = args.save_dir
    train_cfg = TrainConfig(env_module=args.env_module)
    train_cfg.seed = args.seed
    train_stage(args.stage, cur_cfg=cur_cfg, train_cfg=train_cfg)


if __name__ == "__main__":
    main()
