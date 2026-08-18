"""
타겟 이동 모델 시뮬레이션 + GIF 생성
- 4분할 GIF: 타겟별 이동 궤적
- 전체 환경 GIF: 4종류 타겟 동시 표시
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
import io

# ── 시뮬레이션 파라미터 ────────────────────────────────────
N_STEPS      = 300
DT           = 1.0
SEED         = 42

# 맵 범위
X_MIN, X_MAX = -20.0,  80.0
Y_MIN, Y_MAX = -50.0,  50.0

# 전차 OU 파라미터
TANK_OU_THETA = 0.15
TANK_OU_SIGMA = 0.8
TANK_SPEED_MAX = 3.0
TANK_X_RANGE   = (20, 80)
TANK_Y_RANGE   = (-30, 30)

# 보병 파라미터
INF_P_STOP     = 0.05
INF_P_MOVE     = 0.08
INF_SPEED_MIN  = 0.3
INF_SPEED_MAX  = 0.8
INF_TURN_SIGMA = 0.3

# 포병 파라미터
ARTIL_RELO_INTERVAL = 400
ARTIL_RELO_JITTER   = 200
ARTIL_RELO_SPEED    = 1.5
ARTIL_RELO_PROB     = 0.7

# ── 색상 ─────────────────────────────────────────────────
COLOR_TANK    = '#185FA5'   # 파랑
COLOR_INF     = '#3B6D11'   # 초록
COLOR_ARTIL   = '#BA7517'   # 주황
COLOR_AA      = '#A32D2D'   # 빨강
COLOR_BG      = '#F8F8F6'
COLOR_TERRAIN = '#D3D1C7'


# ══════════════════════════════════════════════════════════
# 이동 모델 시뮬레이터
# ══════════════════════════════════════════════════════════

def sim_tank(rng):
    pos = np.array([
        float(rng.uniform(*TANK_X_RANGE)),
        float(rng.uniform(*TANK_Y_RANGE))
    ])
    vel = np.zeros(2)
    traj = [pos.copy()]

    for _ in range(N_STEPS):
        dW = rng.standard_normal(2) * np.sqrt(DT)
        vel += -TANK_OU_THETA * vel * DT + TANK_OU_SIGMA * dW

        spd = np.linalg.norm(vel)
        if spd > TANK_SPEED_MAX:
            vel = vel / spd * TANK_SPEED_MAX

        npos = pos + vel

        # 경계 반사
        if not (TANK_X_RANGE[0] <= npos[0] <= TANK_X_RANGE[1]):
            vel[0] *= -1
            npos[0] = np.clip(npos[0], *TANK_X_RANGE)

        if not (TANK_Y_RANGE[0] <= npos[1] <= TANK_Y_RANGE[1]):
            vel[1] *= -1
            npos[1] = np.clip(npos[1], *TANK_Y_RANGE)

        pos = npos
        traj.append(pos.copy())

    return np.array(traj)


def sim_infantry(rng):
    pos = np.array([
        float(rng.uniform(5, 20)),
        float(rng.uniform(-30, 30))
    ])
    moving  = False
    heading = 0.0
    speed   = 0.0
    traj    = [pos.copy()]
    states  = [moving]   # True=이동, False=정지

    for _ in range(N_STEPS):
        if moving:
            if rng.random() < INF_P_STOP:
                moving = False
                speed = 0.0
        else:
            if rng.random() < INF_P_MOVE:
                moving  = True
                heading = float(rng.uniform(0, 2*np.pi))
                speed   = float(rng.uniform(INF_SPEED_MIN, INF_SPEED_MAX))

        if moving:
            heading += float(rng.standard_normal() * INF_TURN_SIGMA)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            nx = pos[0] + vx
            ny = pos[1] + vy

            if not (5 <= nx <= 20) or not (-30 <= ny <= 30):
                heading += np.pi
                nx, ny = float(pos[0]), float(pos[1])
                moving = False
                speed = 0.0

            pos = np.array([nx, ny])

        traj.append(pos.copy())
        states.append(moving)

    return np.array(traj), states


def sim_artillery(rng):
    pos = np.array([
        float(rng.uniform(42, 60)),
        float(rng.uniform(-20, 20))
    ])
    traj = [pos.copy()]
    timer = int(rng.integers(200, 400))
    target_pos = None
    states = ['static']  # 'static' or 'moving'

    for _ in range(N_STEPS):
        if target_pos is not None:
            dx = target_pos[0] - pos[0]
            dy = target_pos[1] - pos[1]
            dist = np.sqrt(dx*dx + dy*dy)

            if dist < ARTIL_RELO_SPEED * 1.5:
                pos = target_pos.copy()
                target_pos = None
                timer = int(
                    ARTIL_RELO_INTERVAL
                    + rng.integers(-ARTIL_RELO_JITTER // 2,
                                   ARTIL_RELO_JITTER // 2)
                )
                states.append('static')
            else:
                pos = pos + np.array([dx, dy]) / dist * ARTIL_RELO_SPEED
                states.append('moving')
        else:
            timer -= 1
            if timer <= 0 and rng.random() < ARTIL_RELO_PROB:
                tx = float(rng.uniform(42, 60))
                ty = float(rng.uniform(-20, 20))
                target_pos = np.array([tx, ty])
                states.append('moving')
            else:
                states.append('static')

        traj.append(pos.copy())

    return np.array(traj), states


def sim_antiair(rng):
    pos = np.array([
        float(rng.uniform(45, 57)),
        float(rng.uniform(-15, 15))
    ])
    traj = np.tile(pos, (N_STEPS + 1, 1))
    return traj


# ══════════════════════════════════════════════════════════
# GIF 1: 4분할
# ══════════════════════════════════════════════════════════

def make_4panel_gif(tank, inf_t, inf_s, artil_t, artil_s, aa):
    frames = []
    TRAIL = 40
    STEP_SKIP = 3

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 8),
        facecolor=COLOR_BG,
        constrained_layout=True
    )
    fig.suptitle("Target Movement Models", fontsize=14, fontweight='bold')

    titles = [
        "Tank (OU Process)",
        "Infantry (Stop/Move)",
        "Artillery (Relocation)",
        "Anti-Air (Static)"
    ]
    colors = [COLOR_TANK, COLOR_INF, COLOR_ARTIL, COLOR_AA]
    xlims  = [(15, 85), (0, 25), (37, 65), (40, 62)]
    ylims  = [(-35, 35), (-35, 35), (-25, 25), (-20, 20)]

    for ax, title, color, xl, yl in zip(axes.flat, titles, colors, xlims, ylims):
        ax.set_facecolor(COLOR_BG)
        ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=6)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    for t in range(0, N_STEPS + 1, STEP_SKIP):
        for ax in axes.flat:
            ax.cla()

        data = [tank, inf_t, artil_t, aa]

        for idx, (ax, title, color, xl, yl, traj) in enumerate(
            zip(axes.flat, titles, colors, xlims, ylims, data)
        ):
            ax.set_facecolor(COLOR_BG)
            ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=6)
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_box_aspect(1)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            t0 = max(0, t - TRAIL)
            seg = traj[t0:t+1]

            # 전체 경로
            if t > 0:
                ax.plot(
                    traj[:t+1, 0], traj[:t+1, 1],
                    color=color, alpha=0.08, linewidth=0.8, zorder=1
                )

            # trailing 궤적
            if len(seg) > 1:
                for i in range(len(seg) - 1):
                    alpha = 0.15 + 0.7 * (i / max(len(seg) - 1, 1))
                    ax.plot(
                        seg[i:i+2, 0], seg[i:i+2, 1],
                        color=color, alpha=alpha, linewidth=1.8, zorder=2
                    )

            # 현재 위치
            marker = 's' if idx == 3 else ('D' if idx == 2 else 'o')
            ax.scatter(
                traj[t, 0], traj[t, 1],
                color=color, s=60, zorder=5, marker=marker,
                edgecolors='white', linewidths=0.8
            )

            # 보병 상태 표시
            if idx == 1:
                state_str = "MOVE" if inf_s[t] else "STOP"
                state_col = COLOR_INF if inf_s[t] else '#888780'
                ax.text(
                    xl[0] + 0.5, yl[1] - 1.5, state_str,
                    fontsize=9, color=state_col, fontweight='bold'
                )

            # 포병 상태 표시
            if idx == 2:
                state_str = "MOVING" if artil_s[t] == 'moving' else "STATIC"
                state_col = COLOR_ARTIL if artil_s[t] == 'moving' else '#888780'
                ax.text(
                    xl[0] + 0.5, yl[1] - 1.5, state_str,
                    fontsize=9, color=state_col, fontweight='bold'
                )

            # step 표시
            ax.text(
                xl[1] - 0.5, yl[0] + 0.8, f"step {t}",
                fontsize=8, color='#888780', ha='right'
            )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            dpi=90,
            facecolor=COLOR_BG
        )
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)
    return frames


# ══════════════════════════════════════════════════════════
# GIF 2: 전체 환경
# ══════════════════════════════════════════════════════════

def make_fullenv_gif(rng):
    n_inf   = 8
    n_tank  = 2
    n_artil = 4
    n_aa    = 2

    tanks  = [sim_tank(rng) for _ in range(n_tank)]
    infs   = [sim_infantry(rng) for _ in range(n_inf)]
    artils = [sim_artillery(rng) for _ in range(n_artil)]
    aas    = [sim_antiair(rng) for _ in range(n_aa)]

    frames = []
    TRAIL = 30
    STEP_SKIP = 3

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_TANK,
               markersize=8, label=f'Tank ×{n_tank}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INF,
               markersize=8, label=f'Infantry ×{n_inf}'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_ARTIL,
               markersize=8, label=f'Artillery ×{n_artil}'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_AA,
               markersize=8, label=f'Anti-Air ×{n_aa}'),
    ]

    for t in range(0, N_STEPS + 1, STEP_SKIP):
        ax.cla()
        ax.set_facecolor(COLOR_BG)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect('equal')
        ax.set_title("Reconnaissance Environment — Target Movement",
                     fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # 구역 배경
        ax.add_patch(mpatches.FancyBboxPatch(
            (5, -30), 15, 60, linewidth=0,
            facecolor=COLOR_INF, alpha=0.05, zorder=0
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (20, -30), 60, 60, linewidth=0,
            facecolor=COLOR_TANK, alpha=0.03, zorder=0
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (42, -20), 18, 40, linewidth=0,
            facecolor=COLOR_ARTIL, alpha=0.06, zorder=0
        ))

        # 전차
        for traj in tanks:
            t0 = max(0, t - TRAIL)
            seg = traj[t0:t+1]
            ax.plot(traj[:t+1, 0], traj[:t+1, 1],
                    color=COLOR_TANK, alpha=0.07, linewidth=0.7, zorder=1)
            if len(seg) > 1:
                for i in range(len(seg) - 1):
                    a = 0.2 + 0.7 * (i / max(len(seg) - 1, 1))
                    ax.plot(seg[i:i+2, 0], seg[i:i+2, 1],
                            color=COLOR_TANK, alpha=a, linewidth=1.8, zorder=2)
            ax.scatter(traj[t, 0], traj[t, 1], color=COLOR_TANK, s=55,
                       zorder=5, marker='o', edgecolors='white', linewidths=0.8)

        # 보병
        for traj, states in infs:
            t0 = max(0, t - TRAIL)
            seg = traj[t0:t+1]
            ax.plot(traj[:t+1, 0], traj[:t+1, 1],
                    color=COLOR_INF, alpha=0.07, linewidth=0.7, zorder=1)
            if len(seg) > 1:
                for i in range(len(seg) - 1):
                    a = 0.2 + 0.6 * (i / max(len(seg) - 1, 1))
                    lw = 1.5 if states[t0 + i] else 0.5
                    ax.plot(seg[i:i+2, 0], seg[i:i+2, 1],
                            color=COLOR_INF, alpha=a, linewidth=lw, zorder=2)
            mfc = COLOR_INF if states[t] else '#B4B2A9'
            ax.scatter(traj[t, 0], traj[t, 1], color=mfc, s=40,
                       zorder=5, marker='o', edgecolors='white', linewidths=0.7)

        # 포병
        for traj, states in artils:
            t0 = max(0, t - TRAIL)
            seg = traj[t0:t+1]
            if len(seg) > 1:
                for i in range(len(seg) - 1):
                    a = 0.15 + 0.7 * (i / max(len(seg) - 1, 1))
                    ax.plot(seg[i:i+2, 0], seg[i:i+2, 1],
                            color=COLOR_ARTIL, alpha=a, linewidth=1.5, zorder=2)
            mfc = COLOR_ARTIL if states[t] == 'moving' else '#FAC775'
            ax.scatter(traj[t, 0], traj[t, 1], color=mfc, s=55,
                       zorder=5, marker='D', edgecolors='white', linewidths=0.8)

        # 방공
        for traj in aas:
            ax.scatter(traj[0, 0], traj[0, 1], color=COLOR_AA, s=65,
                       zorder=5, marker='s', edgecolors='white', linewidths=0.9)

        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=9, framealpha=0.85, edgecolor='#D3D1C7',
                  fancybox=False)
        ax.text(X_MIN + 1, Y_MAX - 3, f"step {t:4d} / {N_STEPS}",
                fontsize=9, color='#5F5E5A')

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            dpi=90,
            facecolor=COLOR_BG
        )
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)
    return frames


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    OUT_DIR = os.path.join(os.getcwd(), "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

    gif_4panel_path = os.path.join(OUT_DIR, "target_4panel.gif")
    gif_fullenv_path = os.path.join(OUT_DIR, "target_fullenv.gif")

    print("시뮬레이션 중...")
    tank_traj = sim_tank(rng)
    inf_traj, inf_states = sim_infantry(rng)
    artil_traj, artil_states = sim_artillery(rng)
    aa_traj = sim_antiair(rng)

    print("4분할 GIF 생성 중...")
    frames4 = make_4panel_gif(
        tank_traj, inf_traj, inf_states,
        artil_traj, artil_states, aa_traj
    )
    frames4[0].save(
        gif_4panel_path,
        save_all=True,
        append_images=frames4[1:],
        optimize=False,
        duration=60,
        loop=0
    )
    print(f"  → {gif_4panel_path}  ({len(frames4)} frames)")

    print("전체 환경 GIF 생성 중...")
    rng2 = np.random.default_rng(SEED)
    frames_full = make_fullenv_gif(rng2)
    frames_full[0].save(
        gif_fullenv_path,
        save_all=True,
        append_images=frames_full[1:],
        optimize=False,
        duration=60,
        loop=0
    )
    print(f"  → {gif_fullenv_path}  ({len(frames_full)} frames)")

    print("완료!")
