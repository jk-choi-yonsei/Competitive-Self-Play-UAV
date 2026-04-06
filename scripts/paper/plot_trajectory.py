#!/usr/bin/env python3
"""
SAM Evasion Trajectory Visualization Script (for Paper)
- Apply 3x3 layout and visualization logic from existing training code (3D / XY / XZ / Speed / Legend)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.gridspec import GridSpec

ROOT    = Path(__file__).resolve().parent.parent.parent
TRAJ_IN = ROOT / "figures" / "paper_figures" / "traj_data"
OUT     = ROOT / "paper" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "Times New Roman",
    "font.size":       13,
    "axes.titlesize":  14,
    "axes.labelsize":  13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_COLORS = {
    "SPOT": "#1565C0", # Deep Blue (Premium)
    "SAC (Scratch)":       "#E67E22", # Strong Orange
    "SAC (Nav)":    "#2E7D32", # Deep Green
}
RADAR_COLOR = "#B03A2E"
DEFAULT_COLORS = ["#1565C0", "#E67E22", "#2E7D32", "#7E57C2"]

# ── Geometry helpers ───────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))

def latlon_to_ne(ref_lat, ref_lon, lat, lon):
    n = haversine_m(ref_lat, ref_lon, lat, ref_lon) * (1 if lat >= ref_lat else -1)
    e = haversine_m(ref_lat, ref_lon, ref_lat, lon) * (1 if lon >= ref_lon else -1)
    return n, e

def df_to_ne(df, ref_lat, ref_lon):
    ne = [latlon_to_ne(ref_lat, ref_lon, r.lat, r.lon) for r in df.itertuples()]
    n, e = zip(*ne)
    return np.array(n), np.array(e)

def sam_to_ne(df, ref_lat, ref_lon):
    row = df.iloc[0]
    return latlon_to_ne(ref_lat, ref_lon, row.sam_lat, row.sam_lon)

def goal_to_ne(df, ref_lat, ref_lon):
    row = df.iloc[0]
    return latlon_to_ne(ref_lat, ref_lon, row.goal_lat, row.goal_lon)

# ── Visualization helpers from training code ───────────────────────────────
def add_circle_xy(ax, cx, cy, r, *, color, alpha=0.18, lw=1.5):
    theta = np.linspace(0, 2*math.pi, 180)
    ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), color=color, linewidth=lw, alpha=alpha)

def add_cylinder_3d(ax, cx, cy, cz, r, half_h, *, color, alpha=0.24, lw=1.0, n_theta=40):
    theta = np.linspace(0, 2*math.pi, n_theta)
    x = cx + r*np.cos(theta); y = cy + r*np.sin(theta)
    z0, z1 = cz - half_h, cz + half_h
    ax.plot(x, y, np.full_like(x, z0), color=color, linewidth=lw, alpha=alpha)
    ax.plot(x, y, np.full_like(x, z1), color=color, linewidth=lw, alpha=alpha)
    for k in range(0, len(theta), max(len(theta)//8, 1)):
        ax.plot([x[k], x[k]], [y[k], y[k]], [z0, z1], color=color, linewidth=lw, alpha=alpha)

def add_sphere_3d(ax, cx, cy, cz, r, *, color, alpha=0.15, lw=0.7, n_u=20, n_v=12):
    u = np.linspace(0, 2*math.pi, n_u)
    # n_v/2 to only draw the upper hemisphere
    v = np.linspace(0, math.pi/2, n_v // 2)
    uu, vv = np.meshgrid(u, v)
    ax.plot_wireframe(cx + r*np.cos(uu)*np.sin(vv),
                      cy + r*np.sin(uu)*np.sin(vv),
                      cz + r*np.cos(vv),
                      color=color, linewidth=lw, alpha=alpha)

def colored_segments_2d(ax, east, north, in_radar, base_color):
    """Plot trajectory segments, dashed if in radar range."""
    # Group contiguous states to ensure dashed patterns (ls='--') look correct
    i = 0
    while i < len(east) - 1:
        state = bool(in_radar[i])
        j = i
        while j < len(east) - 1 and bool(in_radar[j]) == state:
            j += 1
        ls = "--" if state else "-"
        ax.plot(east[i:j+1], north[i:j+1], color=base_color, 
                ls=ls, lw=2.4 if state else 2.2, alpha=1.0, solid_capstyle="round")
        i = j

def colored_segments_3d(ax, east, north, alt, in_radar, base_color):
    """Plot 3D trajectory segments, dashed if in radar range."""
    i = 0
    while i < len(east) - 1:
        state = bool(in_radar[i])
        j = i
        while j < len(east) - 1 and bool(in_radar[j]) == state:
            j += 1
        ls = "--" if state else "-"
        # Match width and style with 2D
        ax.plot(east[i:j+1], north[i:j+1], alt[i:j+1], color=base_color, 
                ls=ls, lw=2.4 if state else 2.2, alpha=1.0)
        i = j

def colored_segments_xz(ax, east, alt, in_radar, base_color):
    """Plot XZ trajectory segments, dashed if in radar range."""
    i = 0
    while i < len(east) - 1:
        state = bool(in_radar[i])
        j = i
        while j < len(east) - 1 and bool(in_radar[j]) == state:
            j += 1
        ls = "--" if state else "-"
        ax.plot(east[i:j+1], alt[i:j+1], color=base_color, 
                ls=ls, lw=2.4 if state else 2.2, alpha=1.0, solid_capstyle="round")
        i = j

# ── Single episode plot (Upgrade) ──────────────────────────────────────────
def plot_single(csv_path, title=None, save_name=None):
    df = pd.read_csv(csv_path)
    ref_lat, ref_lon = float(df.iloc[0].lat), float(df.iloc[0].lon)
    ref_alt          = float(df.iloc[0].alt_m)

    north, east = df_to_ne(df, ref_lat, ref_lon)
    alt_abs     = df["alt_m"].values
    spd_kts     = df["speed_fps"].values * 0.592484
    radar       = df["in_radar"].values.astype(bool)
    
    sam_n, sam_e = sam_to_ne(df, ref_lat, ref_lon)
    gol_n, gol_e = goal_to_ne(df, ref_lat, ref_lon)
    gol_alt_rel  = float(df.iloc[0].goal_alt) - ref_alt

    last  = df.iloc[-1]
    is_success = bool(last.success)
    is_killed  = bool(last.killed)
    status_str = "SUCCESS" if is_success else ("KILLED BY SAM" if is_killed else "FAILED")
    status_col = "#2E7D32" if is_success else "#C62828"

    # Mission constants
    r_horiz, r_vert = 10000.0, 5000.0
    r_succ, h_succ  = 1500.0, 1000.0

    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], hspace=0.45, wspace=0.45)
    
    ax3d   = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy  = fig.add_subplot(gs[0, 1])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_spd = fig.add_subplot(gs[2, 1])


    # 1. 3D View
    colored_segments_3d(ax3d, east, north, alt_abs, radar, "#1565C0")
    z_floor = 0
    ax3d.plot(east, north, np.full_like(alt_abs, z_floor), color="k", lw=0.8, alpha=0.1)
    
    ax3d.scatter(east[0], north[0], alt_abs[0], s=80, color="blue", edgecolors="k", label="Start")
    ax3d.scatter(gol_e, gol_n, float(df.iloc[0].goal_alt), s=150, color="green", marker="*", edgecolors="k", label="Goal")
    add_cylinder_3d(ax3d, gol_e, gol_n, float(df.iloc[0].goal_alt), r_succ, h_succ, color="green")
    
    ax3d.scatter(sam_e, sam_n, 0, s=100, color="red", edgecolors="k", marker="^", label="SAM")
    add_sphere_3d(ax3d, sam_e, sam_n, 0, r_horiz*2, color="red", alpha=0.06, lw=0.5)
    add_sphere_3d(ax3d, sam_e, sam_n, 0, r_horiz, color="darkred", alpha=0.15, lw=0.8)

    ax3d.set_title("3D Trajectory View", pad=10, color="black", fontweight="bold")
    ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Altitude (m)")
    ax3d.view_init(elev=22, azim=-55)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    
    # Square aspect for N-E, and absolute Z limits
    ax3d.set_box_aspect((1, 1, 0.5))
    all_n = np.concatenate([north, [sam_n, gol_n]])
    all_e = np.concatenate([east, [sam_e, gol_e]])
    e_mid, n_mid = (all_e.max()+all_e.min())/2, (all_n.max()+all_n.min())/2
    range_max = max(all_e.max()-all_e.min(), all_n.max()-all_n.min(), 25000.0)
    ax3d.set_xlim(e_mid - range_max/2, e_mid + range_max/2)
    ax3d.set_ylim(n_mid - range_max/2, n_mid + range_max/2)
    ax3d.set_zlim(0, 10000)

    # 2. XY View
    add_circle_xy(ax_xy, sam_e, sam_n, r_horiz, color="red", alpha=0.20, lw=1.2)
    colored_segments_2d(ax_xy, east, north, radar, "#1565C0")
    ax_xy.scatter(east[0], north[0], s=50, color="blue", edgecolors="k")
    ax_xy.scatter(gol_e, gol_n, s=150, color="green", marker="*", edgecolors="k")
    ax_xy.scatter(sam_e, sam_n, s=80, color="red", marker="^", edgecolors="k")
    ax_xy.set_title("Top-Down View (XY)", color="black", fontweight="bold")
    ax_xy.set_xlabel("East (m)")
    ax_xy.set_ylabel("North (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, ls="--", alpha=0.3)
    # Add padding to ensure goal is visible
    xlim = ax_xy.get_xlim(); margin_x = (xlim[1]-xlim[0])*0.15
    ax_xy.set_xlim(xlim[0]-margin_x, xlim[1]+margin_x)

    # 3. XZ View
    colored_segments_xz(ax_xz, east, alt_abs, radar, "#1565C0")

    ax_xz.scatter(gol_e, float(df.iloc[0].goal_alt), s=100, color="green", marker="*")
    ax_xz.scatter(sam_e, 0, s=80, color="red", marker="^")

    # SAM Dome in XZ
    theta = np.linspace(0, np.pi, 100)
    ax_xz.plot(sam_e + r_horiz*np.cos(theta), r_vert*np.sin(theta), color="red", lw=1.2, ls="--", alpha=0.6)
    ax_xz.fill(sam_e + r_horiz*np.cos(theta), r_vert*np.sin(theta), color="red", alpha=0.05)

    ax_xz.set_title("Side View (Altitude Profile)", color="black", fontweight="bold")
    ax_xz.set_xlabel("East (m)")
    ax_xz.set_ylabel("Altitude (m)")
    ax_xz.set_ylim(0, 10000)
    ax_xz.grid(True, ls="--", alpha=0.3)

    # 4. Speed View
    ax_spd.plot(spd_kts, color="#7E57C2", lw=2)
    ax_spd.set_title("Velocity Profile", color="black", fontweight="bold")
    ax_spd.set_ylabel("Speed (kts)")
    ax_spd.grid(True, ls="--", alpha=0.3)

    # 5. Legend
    lm = {
        "Agent Trajectory": plt.Line2D([0],[0], color="#1565C0", lw=2),
        "In Radar Range": plt.Line2D([0],[0], color="#1565C0", lw=2, ls="--"),
        "Goal Waypoint": plt.Line2D([0],[0], color="green", marker="*", ls=""),
        "SAM Threat (10km)": plt.Line2D([0],[0], color="darkred", alpha=0.5, lw=2),
    }
    ax_xy.legend(list(lm.values()), list(lm.keys()), loc="upper left", bbox_to_anchor=(1.05, 1.0),
                   frameon=True, borderpad=0.5, labelspacing=0.35,
                   handlelength=1.5, handletextpad=0.5, fontsize=11)

    # Suptitle
    model_name = title or "Model Evaluation"
    fig.suptitle(f"{model_name} — {status_str}", fontsize=20, fontweight="bold", y=0.97, x=0.48, ha="center", color="black")

    if save_name is None:
        save_name = f"traj_{csv_path.parent.name}_{csv_path.stem}"
    fig.savefig(OUT / f"{save_name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[OK] {save_name}")

# ── Multi-model comparison plot (Premium Overlaid Layout) ──────────────────
def plot_compare(csv_paths, labels, save_name="traj_comparison", title=None):
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    n = len(csv_paths)
    dfs = [pd.read_csv(p) for p in csv_paths]
    ref_lat, ref_lon = float(dfs[0].iloc[0].lat), float(dfs[0].iloc[0].lon)
    ref_alt          = float(dfs[0].iloc[0].alt_m)
    r_horiz, r_vert = 10000.0, 5000.0

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], hspace=0.45, wspace=0.45)
    
    ax3d   = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy  = fig.add_subplot(gs[0, 1])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_spd = fig.add_subplot(gs[2, 1])

    # SAM Position (assume same for all in comparison)
    sam_n, sam_e = sam_to_ne(dfs[0], ref_lat, ref_lon)
    gol_n, gol_e = goal_to_ne(dfs[0], ref_lat, ref_lon)

    # -- Background: SAM Domes/Circles — Exactly same as plot_single --
    # XY: SAM circle (alpha=0.20, lw=1.2 — same as plot_single)
    add_circle_xy(ax_xy, sam_e, sam_n, r_horiz, color="red", alpha=0.20, lw=1.2)
    ax_xy.scatter(sam_e, sam_n, s=80,  color="red",   marker="^", edgecolors="k")   # Fully opaque
    ax_xy.scatter(gol_e, gol_n, s=150, color="green", marker="*", edgecolors="k")   # Fully opaque

    # XZ: SAM dome (alpha=0.6, fill=0.05 — same as plot_single)
    ax_xz.scatter(sam_e, 0, s=80, color="red", marker="^", edgecolors="k")
    ax_xz.scatter(gol_e, float(dfs[0].iloc[0].goal_alt), s=100, color="green", marker="*")
    theta = np.linspace(0, np.pi, 100)
    ax_xz.plot(sam_e + r_horiz*np.cos(theta), r_vert*np.sin(theta), color="red", lw=1.2, ls="--", alpha=0.6)
    ax_xz.fill(sam_e + r_horiz*np.cos(theta), r_vert*np.sin(theta), color="red", alpha=0.05)

    # 3D: 2 Spheres (same as plot_single: Outer r*2 + Inner r)
    add_sphere_3d(ax3d, sam_e, sam_n, 0, r_horiz*2, color="red",    alpha=0.06, lw=0.5)
    add_sphere_3d(ax3d, sam_e, sam_n, 0, r_horiz,   color="darkred", alpha=0.15, lw=0.8)
    # 3D SAM/Goal Markers (same as plot_single)
    ax3d.scatter(sam_e, sam_n, 0,    s=100, color="red",   edgecolors="k", marker="^")
    ax3d.scatter(gol_e, gol_n, float(dfs[0].iloc[0].goal_alt), s=150, color="green", edgecolors="k", marker="*")

    # ── Plot each model ──
    for i, (df, label) in enumerate(zip(dfs, labels)):
        north, east = df_to_ne(df, ref_lat, ref_lon)
        alt_abs     = df["alt_m"].values
        radar       = df["in_radar"].values.astype(bool)
        spd         = df["speed_fps"].values * 0.592484
        color       = MODEL_COLORS.get(label, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

        # Check for Popup
        popup_idx = -1
        if "popup_triggered" in df.columns:
            idx = df[df["popup_triggered"] == 1].index
            if not idx.empty: popup_idx = idx[0]

        # 1. 3D: Trajectory + Start point (plot_single: s=80, edgecolors="k")
        colored_segments_3d(ax3d, east, north, alt_abs, radar, color)
        ax3d.plot(east, north, np.full_like(alt_abs, 0), color="k", lw=0.8, alpha=0.08)  # shadow
        ax3d.scatter(east[0], north[0], alt_abs[0], s=80, color=color, edgecolors="k", zorder=5)

        # 2. XY: Trajectory + Start point (plot_single: s=50, edgecolors="k")
        colored_segments_2d(ax_xy, east, north, radar, color)
        ax_xy.scatter(east[0], north[0], s=50, color=color, edgecolors="k", lw=0.8, zorder=5)

        # 3. XZ
        colored_segments_xz(ax_xz, east, alt_abs, radar, color)

        # 4. Speed
        ax_spd.plot(spd, color=color, lw=2.2, alpha=1.0)

        # Popup Markings
        if popup_idx >= 0:
            ax3d.scatter(east[popup_idx], north[popup_idx], alt_abs[popup_idx], s=120, edgecolors="orange", facecolors="none", lw=2, zorder=10)
            ax_xy.scatter(east[popup_idx], north[popup_idx], s=120, edgecolors="orange", facecolors="none", lw=2, zorder=10)
            ax_xz.scatter(east[popup_idx], alt_abs[popup_idx], s=120, edgecolors="orange", facecolors="none", lw=2, zorder=10)

    # ── Finalize Appearance ──
    ax3d.set_title("3D Trajectory Comparison", color="black", fontweight="bold")
    ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Altitude (m)")
    ax3d.view_init(elev=22, azim=-55)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    
    # Expand 3D limits for square aspect
    ax3d.set_box_aspect((1, 1, 0.5))
    all_n = np.concatenate([df_to_ne(df, ref_lat, ref_lon)[0] for df in dfs] + [[sam_n, gol_n]])
    all_e = np.concatenate([df_to_ne(df, ref_lat, ref_lon)[1] for df in dfs] + [[sam_e, gol_e]])
    e_mid, n_mid = (all_e.max()+all_e.min())/2, (all_n.max()+all_n.min())/2
    range_max = max(all_e.max()-all_e.min(), all_n.max()-all_n.min(), 25000.0)
    ax3d.set_xlim(e_mid - range_max/2, e_mid + range_max/2)
    ax3d.set_ylim(n_mid - range_max/2, n_mid + range_max/2)
    ax3d.set_zlim(0, 10000)

    ax_xy.set_title("Top-Down View (XY)", color="black", fontweight="bold")
    ax_xy.set_xlabel("East (m)")
    ax_xy.set_ylabel("North (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, ls="--", alpha=0.3)

    ax_xz.set_title("Side View (Altitude Profile)", color="black", fontweight="bold")
    ax_xz.set_xlabel("East (m)")
    ax_xz.set_ylabel("Altitude (m)")
    ax_xz.grid(True, ls="--", alpha=0.3)
    ax_xz.set_ylim(0, 10000)

    ax_spd.set_title("Velocity Comparison", color="black", fontweight="bold", fontsize=10)
    ax_spd.set_xlabel("Timestep")
    ax_spd.set_ylabel("Speed (kts)")
    ax_spd.grid(True, ls="--", alpha=0.3)

    # Legend
    model_handles = [Line2D([0],[0], color=MODEL_COLORS.get(l, DEFAULT_COLORS[i%len(DEFAULT_COLORS)]), lw=2, label=l) 
                     for i, l in enumerate(labels)]
    
    any_popup = any("popup_triggered" in df.columns and (df["popup_triggered"] == 1).any() for df in dfs)
    
    style_handles = [
        Line2D([0],[0], color="black", lw=2, ls="--", label="In Radar Range"),
    ]
    if any_popup:
        style_handles.append(Line2D([0],[0], color="orange", marker="o", mfc="none", mew=2, ls="", label="Pop-up Event"))
    
    style_handles.append(Line2D([0],[0], color="red", lw=1.2, alpha=0.7, label="SAM Coverage (10km)"))

    ax_xy.legend(handles=model_handles + style_handles, loc="upper left", bbox_to_anchor=(1.05, 1.0),
                  frameon=True, borderpad=0.5, labelspacing=0.35,
                  handlelength=1.5, handletextpad=0.5, fontsize=11)

    fig.suptitle(title or "Trajectory Comparison", fontsize=20, fontweight="bold", y=0.97, x=0.48, ha="center", color="black")
    fig.savefig(OUT / f"{save_name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[OK] {save_name}")

def list_episodes(traj_dir):
    p = Path(traj_dir) / "summary.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    print(f"{'ep':>4} | {'seed':>6} | {'status':>8} | {'steps':>5} | {'reward':>10}")
    for r in df.itertuples():
        st = "SUCCESS" if r.success else ("KILLED" if r.killed else "FAIL")
        print(f"{r.episode:4d} | {r.seed:6d} | {st:8s} | {r.steps:5d} | {r.total_reward:10.1f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--csvs", nargs="+", default=[])
    ap.add_argument("--labels", nargs="+", default=[])
    ap.add_argument("--save", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    if args.list:
        list_episodes(args.dir or TRAJ_IN)
        return

    def resolve(p):
        path = Path(p)
        if path.is_absolute(): return path
        candidate = TRAJ_IN / p
        return candidate if candidate.exists() else path

    if args.compare:
        paths = [resolve(p) for p in args.csvs]
        labels = args.labels or [p.parent.name for p in paths]
        plot_compare(paths, labels, save_name=args.save or "traj_comparison", title=args.title)
    elif args.csv:
        plot_single(resolve(args.csv), title=args.title, save_name=args.save)

if __name__ == "__main__":
    main()
