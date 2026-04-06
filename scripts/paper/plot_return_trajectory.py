#!/usr/bin/env python3
"""
Return 미션 궤적 시각화 스크립트 (논문용)

사용법:
  # 1) 단일 에피소드 시각화
  python scripts/plot_return_trajectory.py \
      --mode single \
      --csv figures/paper_figures/traj_data/ft_return_seed42/episode_000.csv \
      --title "SAC FT(Return) - Success" \
      --save traj_return_single

  # 2) 여러 모델 비교
  python scripts/plot_return_trajectory.py \
      --mode compare \
      --csv figures/paper_figures/traj_data/ft_return_seed42/episode_000.csv \
           figures/paper_figures/traj_data/scratch_return_seed42/episode_000.csv \
      --labels "SAC FT(Return)" "SAC Scratch" \
      --save traj_return_compare

  # 3) summary.csv 에피소드 목록 출력
  python scripts/plot_return_trajectory.py \
      --mode list \
      --csv figures/paper_figures/traj_data/ft_return_seed42

출력:
  figures/paper_figures/traj_figures/
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers projection)
from matplotlib.gridspec import GridSpec

ROOT    = Path(__file__).resolve().parent.parent.parent
TRAJ_IN = ROOT / "figures" / "paper_figures" / "traj_data"
OUT     = ROOT / "paper" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "Times New Roman",
    "font.size":         13,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

MODEL_COLORS = {
    "SPOT": "#1565C0", # Deep Blue
    "SAC FT(Self-Play)":  "#1565C0",
    "SAC FT(Self)":       "#1565C0", 
    "SAC FT(Return)":     "#2E7D32", # Strong Green
    "SAC FT (Return)":    "#2E7D32", 
    "SAC Scratch":        "#E67E22", # Strong Orange
}
DEFAULT_COLORS = ["#1565C0", "#E67E22", "#2E7D32", "#8E44AD"]

# ── Geometry helpers ────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def latlon_to_ne(ref_lat, ref_lon, lat, lon):
    """Return (north_m, east_m) relative to reference point."""
    n = haversine_m(ref_lat, ref_lon, lat, ref_lon) * (1 if lat >= ref_lat else -1)
    e = haversine_m(ref_lat, ref_lon, ref_lat, lon) * (1 if lon >= ref_lon else -1)
    return n, e


def df_to_ne(df, ref_lat, ref_lon):
    ne = [latlon_to_ne(ref_lat, ref_lon, r.lat, r.lon) for r in df.itertuples()]
    n, e = zip(*ne)
    return np.array(n), np.array(e)


def extract_waypoints(df, ref_lat, ref_lon):
    """
    Return list of (wp_number, north_m, east_m, alt_m) for each unique waypoint.
    Waypoints 1..N are the distinct goal positions that appear in order.
    The last wp_idx is skipped if it duplicates the previous goal.
    """
    rows = df.drop_duplicates("wp_idx").copy()
    # Drop last duplicate if same coords as previous
    if len(rows) > 1 and (
        rows.iloc[-1].goal_lat == rows.iloc[-2].goal_lat
        and rows.iloc[-1].goal_lon == rows.iloc[-2].goal_lon
    ):
        rows = rows.iloc[:-1]

    waypoints = []
    for i, row in enumerate(rows.itertuples()):
        n, e = latlon_to_ne(ref_lat, ref_lon, row.goal_lat, row.goal_lon)
        waypoints.append((i + 1, n, e, float(row.goal_alt)))
    return waypoints


def add_circle_xy(ax, cx, cy, r, *, color, alpha=0.18, lw=1.5):
    theta = np.linspace(0, 2*math.pi, 180)
    ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), color=color, linewidth=lw, alpha=alpha)

def add_sphere_3d(ax, cx, cy, cz, r, *, color, alpha=0.12, lw=0.6, n_u=20, n_v=12):
    u = np.linspace(0, 2*math.pi, n_u)
    # Upper hemisphere only
    v = np.linspace(0, math.pi/2, n_v // 2)
    uu, vv = np.meshgrid(u, v)
    ax.plot_wireframe(cx + r*np.cos(uu)*np.sin(vv),
                      cy + r*np.sin(uu)*np.sin(vv),
                      cz + r*np.cos(vv),
                      color=color, linewidth=lw, alpha=alpha)

def add_cylinder_3d(ax, cx, cy, cz, r, half_h, *, color, alpha=0.24, lw=1.0, n_theta=40):
    theta = np.linspace(0, 2*math.pi, n_theta)
    x = cx + r*np.cos(theta)
    y = cy + r*np.sin(theta)
    z0, z1 = cz - half_h, cz + half_h
    ax.plot(x, y, np.full_like(x, z0), color=color, linewidth=lw, alpha=alpha)
    ax.plot(x, y, np.full_like(x, z1), color=color, linewidth=lw, alpha=alpha)
    for k in range(0, len(theta), max(len(theta)//8, 1)):
        ax.plot([x[k], x[k]], [y[k], y[k]], [z0, z1], color=color, linewidth=lw, alpha=alpha)


# ── Single episode plot (Reverted to training code style) ──────────────────
def plot_single(csv_path, title=None, save_name=None):
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    ref_lat, ref_lon = float(df.iloc[0].lat), float(df.iloc[0].lon)
    ref_alt          = float(df.iloc[0].alt_m)

    north, east = df_to_ne(df, ref_lat, ref_lon)
    alt_abs     = df["alt_m"].values
    spd_kts     = df["speed_fps"].values * 0.592484
    wp_idxs     = df["wp_idx"].values.astype(int)
    
    # Identify WP reach steps
    wp_reach = {}
    for i in range(1, len(wp_idxs)):
        if wp_idxs[i] > wp_idxs[i-1]:
            wp_reach[wp_idxs[i-1]] = i
    
    # Success gate info
    r_succ, h_succ = 1500.0, 1000.0
    in_gate = [
        (float(d) < r_succ) and (abs(float(a) - float(ga)) < h_succ)
        for d, a, ga in zip(df.dist_to_goal_m, df.alt_m, df.goal_alt)
    ]


    last = df.iloc[-1]
    is_success = bool(last.success)
    is_crashed = bool(last.crashed)
    status_str = "Success" if is_success else ("Crashed" if is_crashed else "Failure")
    status_col = "black" # Use black as requested earlier

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], hspace=0.45, wspace=0.45)
    
    ax3d   = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy  = fig.add_subplot(gs[0, 1])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_spd = fig.add_subplot(gs[2, 1])

    # 1. 3D View
    stage_colors = {0: "C1", 1: "C2", 2: "C3", 3: "C4"}
    pts = np.column_stack([east, north, df["alt_m"].values])
    if len(pts) >= 2:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_cols = [stage_colors.get(s, "C0") for s in wp_idxs[:-1]]
        lc = Line3DCollection(segs, colors=seg_cols, linewidths=2.6, alpha=1.0)
        ax3d.add_collection3d(lc)


    
    # Shadow
    z_floor = 0
    pts_g = pts.copy(); pts_g[:, 2] = z_floor
    segs_g = np.stack([pts_g[:-1], pts_g[1:]], axis=1)
    lc_g = Line3DCollection(segs_g, colors="k", linewidths=1.0, alpha=0.12)
    ax3d.add_collection3d(lc_g)


    # Waypoints & Gates
    wps = extract_waypoints(df, ref_lat, ref_lon)
    for wp_num, n_m, e_m, a_m in wps:
        c = stage_colors.get(wp_num-1, "C0")
        ax3d.scatter(e_m, n_m, a_m, s=120, color=c, edgecolors="black", label=f"WP{wp_num}")
        add_cylinder_3d(ax3d, e_m, n_m, a_m, r_succ, h_succ, color=c)
    
    # Green segments
    for i in range(len(east) - 1):
        if i < len(in_gate) and in_gate[i]:
            ax3d.plot(east[i:i+2], north[i:i+2], df["alt_m"].values[i:i+2], color="green", lw=3.2, alpha=0.95)


    # Star Markers
    for k, st in wp_reach.items():
        if 0 <= st < len(east):
            ax3d.scatter(east[st], north[st], df["alt_m"].values[st], s=160, marker="*", color=stage_colors.get(k, "C0"), edgecolors="black")
    if is_success:
        ax3d.scatter(east[-1], north[-1], df["alt_m"].values[-1], s=180, marker="*", color="green", edgecolors="black")


    ax3d.set_title("3D Trajectory View", color="black", fontweight="bold")
    ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Altitude (m)")
    ax3d.view_init(elev=22, azim=-55)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.set_box_aspect((1, 1, 0.5))

    e_mid, n_mid = (east.max()+east.min())/2, (north.max()+north.min())/2
    range_max = max(east.max()-east.min(), north.max()-north.min(), 15000.0)
    ax3d.set_xlim(e_mid - range_max/2, e_mid + range_max/2)
    ax3d.set_ylim(n_mid - range_max/2, n_mid + range_max/2)
    ax3d.set_zlim(0, 10000)


    # 2. XY View
    ax_xy.plot(east, north, color="C0", lw=2.4, alpha=0.95)

    for wp_num, n_m, e_m, a_m in wps:
        c = stage_colors.get(wp_num-1, "C0")
        ax_xy.scatter(e_m, n_m, color=c, edgecolors="black", s=60)
        add_circle_xy(ax_xy, e_m, n_m, r_succ, color=c)
    
    for i in range(len(east) - 1):
        if i < len(in_gate) and in_gate[i]:
            ax_xy.plot(east[i:i+2], north[i:i+2], color="green", lw=3.0, alpha=0.95)
    
    for k, st in wp_reach.items():
        if 0 <= st < len(east):
            ax_xy.scatter(east[st], north[st], color=stage_colors.get(k, "C0"), edgecolors="black", s=120, marker="*")
    if is_success:
        ax_xy.scatter(east[-1], north[-1], color="green", edgecolors="black", s=120, marker="*")

    ax_xy.set_title("Top-Down View (XY)", color="black", fontweight="bold")
    ax_xy.set_xlabel("East (m)"); ax_xy.set_ylabel("North (m)")

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, ls="--", alpha=0.3)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, ls="--", alpha=0.3)

    # 3. XZ View
    ax_xz.plot(east, df["alt_m"].values, color="C0", lw=2.4, alpha=0.95)
    for wp_num, n_m, e_m, a_m in wps:
        c = stage_colors.get(wp_num-1, "C0")
        ax_xz.scatter(e_m, a_m, color=c, edgecolors="black", s=60)
        ax_xz.axvspan(e_m - r_succ, e_m + r_succ, color=c, alpha=0.1, lw=0)
        ax_xz.axhspan(a_m - h_succ, a_m + h_succ, color=c, alpha=0.08, lw=0)

    for i in range(len(east) - 1):
        if i < len(in_gate) and in_gate[i]:
            ax_xz.plot(east[i:i+2], df["alt_m"].values[i:i+2], color="green", lw=3.2, alpha=1.0)

    ax_xz.set_title("Side View (Altitude Profile)", color="black", fontweight="bold")
    ax_xz.set_xlabel("East (m)"); ax_xz.set_ylabel("Altitude (m)")
    ax_xz.set_ylim(0, 10000)
    ax_xz.grid(True, ls="--", alpha=0.3)



    # 4. Speed View
    ax_spd.plot(spd_kts, color="C4", lw=2.4, alpha=0.95)
    ax_spd.set_title("Velocity Profile", color="black", fontweight="bold")
    ax_spd.set_xlabel("Timestep"); ax_spd.set_ylabel("Speed (kts)")
    ax_spd.grid(True, ls="--", alpha=0.3)


    # 5. Legend
    lm = {
        "Agent Trajectory": Line2D([0],[0], color="C0", lw=2),
        "In Success Gate": Line2D([0],[0], color="green", lw=3),
        "WP Reached (*)": Line2D([0],[0], color="k", marker="*", ls=""),
        "WP (1-4)": Line2D([0],[0], color="C1", marker="o", ls=""),
        "Success Gate": mpatches.Patch(color="gray", alpha=0.3),
    }
    ax_xy.legend(list(lm.values()), list(lm.keys()), loc="upper left", bbox_to_anchor=(1.05, 0.98), frameon=True)

    # Suptitle
    t_str = title or "Return Mission Trajectory"
    fig.suptitle(f"{t_str} — {status_str}", fontsize=20, fontweight="bold", y=0.97, x=0.48, ha="center", color="black")

    if save_name is None:
        save_name = f"return_traj_{csv_path.parent.name}_{csv_path.stem}"

    fig.savefig(OUT / f"{save_name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[OK] {save_name}")



# ── Multi-model comparison plot (Upgrade) ──────────────────────────────────
def plot_compare(csv_list, labels, save_name="return_traj_comparison", title=None):
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    n = len(csv_list)
    dfs = [pd.read_csv(p) for p in csv_list]
    ref_lat, ref_lon = float(dfs[0].iloc[0].lat), float(dfs[0].iloc[0].lon)
    ref_alt          = float(dfs[0].iloc[0].alt_m)

    wps = extract_waypoints(dfs[0], ref_lat, ref_lon)
    r_succ, h_succ = 1500.0, 1000.0

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], hspace=0.45, wspace=0.45)
    
    ax3d   = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy  = fig.add_subplot(gs[0, 1])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_spd = fig.add_subplot(gs[2, 1])

    # Draw Waypoints & Gates (Background) — plot_single()과 줄단위 동일
    stage_colors = {0: "C1", 1: "C2", 2: "C3", 3: "C4"}
    for wp_num, n_m, e_m, a_m in wps:
        c = stage_colors.get(wp_num-1, "C0")                                      # plot_single: fallback "C0"
        ax3d.scatter(e_m, n_m, a_m, s=120, color=c, edgecolors="black")           # plot_single: alpha 없음 = 완전불투명
        add_cylinder_3d(ax3d, e_m, n_m, a_m, r_succ, h_succ, color=c)            # plot_single: alpha 없음 = 기본 0.24
        ax_xy.scatter(e_m, n_m, color=c, edgecolors="black", s=60)                # plot_single: s=60, edgecolors="black"
        add_circle_xy(ax_xy, e_m, n_m, r_succ, color=c)                           # 기준 이미지와 동일하게 기본값(0.18, 1.5) 사용
        ax_xz.scatter(e_m, a_m, color=c, edgecolors="black", s=60)                # plot_single: edgecolors="black"
        ax_xz.axvspan(e_m - r_succ, e_m + r_succ, color=c, alpha=0.1, lw=0)      # plot_single: alpha=0.1
        ax_xz.axhspan(a_m - h_succ, a_m + h_succ, color=c, alpha=0.08, lw=0)     # plot_single: alpha=0.08


    # All model trajectories
    for i, (df, label) in enumerate(zip(dfs, labels)):
        n_m, e_m = df_to_ne(df, ref_lat, ref_lon)
        a_abs    = df["alt_m"].values
        spd      = df["speed_fps"].values * 0.592484
        color    = MODEL_COLORS.get(label, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        
        ax3d.plot(e_m, n_m, a_abs, color=color, lw=2.6, alpha=1.0, label=label)
        # 시작점 표시 (plot_single 스타일)
        ax3d.scatter(e_m[0], n_m[0], a_abs[0], s=80, color=color, edgecolors="k", lw=0.8, zorder=5)
        ax_xy.plot(e_m, n_m, color=color, lw=2.4, alpha=0.9, label=label)
        ax_xy.scatter(e_m[0], n_m[0], s=50, color=color, edgecolors="k", lw=0.8, alpha=1.0, zorder=5)
        ax_xz.plot(e_m, a_abs, color=color, lw=2.2, alpha=1.0)
        ax_spd.plot(spd, color=color, lw=2.2, alpha=1.0)



    ax3d.set_title("3D Trajectory Comparison", color="black", fontweight="bold")
    ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Altitude (m)")
    ax3d.view_init(elev=22, azim=-55)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax3d.set_box_aspect((1, 1, 0.5))
    all_n = np.concatenate([df_to_ne(df, ref_lat, ref_lon)[0] for df in dfs])
    all_e = np.concatenate([df_to_ne(df, ref_lat, ref_lon)[1] for df in dfs])
    e_mid, n_mid = (all_e.max()+all_e.min())/2, (all_n.max()+all_n.min())/2
    range_max = max(all_e.max()-all_e.min(), all_n.max()-all_n.min(), 15000.0)
    ax3d.set_xlim(e_mid - range_max/2, e_mid + range_max/2)
    ax3d.set_ylim(n_mid - range_max/2, n_mid + range_max/2)
    ax3d.set_zlim(0, 10000)


    ax_xy.set_title("Top-Down View (XY)", color="black", fontweight="bold")
    ax_xy.set_xlabel("East (m)"); ax_xy.set_ylabel("North (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, ls="--", alpha=0.3)


    ax_xz.set_title("Side View (Altitude Profile)", color="black", fontweight="bold")
    ax_xz.set_xlabel("East (m)"); ax_xz.set_ylabel("Altitude (m)")
    ax_xz.set_ylim(0, 10000)
    ax_xz.grid(True, ls="--", alpha=0.3)



    ax_spd.set_title("Velocity Comparison", color="black", fontweight="bold", fontsize=10)
    ax_spd.set_xlabel("Timestep"); ax_spd.set_ylabel("Speed (kts)")
    ax_spd.grid(True, ls="--", alpha=0.3)


    # Legend for Models
    handles = [Line2D([0], [0], color=MODEL_COLORS.get(l, DEFAULT_COLORS[i%len(DEFAULT_COLORS)]), lw=2, label=l) 
               for i, l in enumerate(labels)]
    ax_xy.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 0.98), frameon=True, title="Models")

    t_str = title or "Comparison of Return Mission Trajectories"
    fig.suptitle(t_str, fontsize=20, fontweight="bold", y=0.97, x=0.48, ha="center", color="black")

    fig.savefig(OUT / f"{save_name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[OK] {save_name}")



# ── List episodes ────────────────────────────────────────────────────────────
def list_episodes(traj_dir):
    traj_dir = Path(traj_dir)
    summary = traj_dir / "summary.csv"
    if not summary.exists():
        # Fall back: list CSV files
        csvs = sorted(traj_dir.glob("episode_*.csv"))
        if not csvs:
            print(f"No summary.csv or episode_*.csv in {traj_dir}")
            return
        print(f"\n{'ep':>4}  {'steps':>6}  {'result':>10}  {'final_reward':>14}")
        print("-" * 40)
        for csv in csvs:
            df = pd.read_csv(csv)
            last = df.iloc[-1]
            is_suc = bool(last.success)
            is_crash = bool(last.crashed)
            result = "SUCCESS" if is_suc else ("CRASHED" if is_crash else "TIMEOUT")
            ep_num = int(csv.stem.split("_")[-1])
            print(f"{ep_num:>4}  {len(df):>6}  {result:>10}  {float(last.reward):>14.3f}")
        return

    df = pd.read_csv(summary)
    print(f"\n{'ep':>4}  {'seed':>6}  {'result':>10}  {'steps':>6}  {'reward':>10}")
    print("-" * 45)
    for r in df.itertuples():
        is_suc = bool(r.success) if hasattr(r, "success") else False
        is_crash = bool(r.crashed) if hasattr(r, "crashed") else False
        result = "SUCCESS" if is_suc else ("CRASHED" if is_crash else "TIMEOUT")
        seed = r.seed if hasattr(r, "seed") else "-"
        steps = r.steps if hasattr(r, "steps") else "-"
        reward = r.total_reward if hasattr(r, "total_reward") else r.reward
        print(f"{r.episode:>4}  {seed!s:>6}  {result:>10}  {steps!s:>6}  {float(reward):>10.1f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Return mission trajectory visualizer (paper-quality figures)"
    )
    ap.add_argument("--mode", choices=["compare", "single", "list"], default="single",
                    help="Visualization mode (default: single)")
    ap.add_argument("--csv", nargs="+", default=[],
                    help="CSV path(s). For single: one file. For compare: multiple files. "
                         "For list: a directory path.")
    ap.add_argument("--labels", nargs="+", default=[],
                    help="Labels for each CSV in compare mode")
    ap.add_argument("--title", default=None,
                    help="Figure title (single mode only)")
    ap.add_argument("--save", default=None,
                    help="Output filename stem (no extension)")
    args = ap.parse_args()

    def resolve(p):
        path = Path(p)
        if path.is_absolute():
            return path
        candidate = TRAJ_IN / p
        if candidate.exists():
            return candidate
        return Path(p)

    if args.mode == "list":
        d = resolve(args.csv[0]) if args.csv else TRAJ_IN
        list_episodes(d)

    elif args.mode == "single":
        if not args.csv:
            ap.error("--mode single requires --csv <path>")
        path = resolve(args.csv[0])
        save = args.save or f"return_traj_{path.parent.name}_{path.stem}"
        plot_single(path, title=args.title, save_name=save)

    elif args.mode == "compare":
        if not args.csv:
            ap.error("--mode compare requires --csv <path1> [path2 ...]")
        paths = [resolve(p) for p in args.csv]
        labels = args.labels or [p.parent.name for p in paths]
        save = args.save or "return_traj_comparison"
        plot_compare(paths, labels, save_name=save)


if __name__ == "__main__":
    main()
