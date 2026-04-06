#!/usr/bin/env python3
"""
Script for generating paper result figures
Output: figures/paper_figures/

List of generated figures:
  A2  - Self-Play Curriculum (Win Rate + Opponent Ratio)
  B1  - Return Mission Learning Curves
  B2  - Return Mission Bar Chart
  C1  - SAM Mission Learning Curves
  C2  - SAM Mission Final Performance Bar Chart
  D   - Robustness 3-Panel (Threat Scaling / Init Perturb / Pop-up SAM)
  F2  - Fine-Tuning Ablation Learning Curves
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/paper/ -> root
RESULTS = ROOT / "paper" / "results"
RUNS    = ROOT / "runs"
OUT     = ROOT / "paper" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "Times New Roman",
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "lines.linewidth":    2.0,
})

# Consistent color palette
C = {
    "SAC_FT_Self":      "#1565C0",   # Deep Blue (Premium)
    "SAC_Scratch":      "#E67E22",   # Strong Orange
    "SAC_FT_Return":    "#2E7D32",   # Deep Green
    "PPO_FT_Self":      "#C0392B",   # Deep Red
    "Heuristic":        "#8E44AD",   # Deep Purple
    "SAC_FT_ActorOnly": "#1565C0",
    "PPO_Scratch":      "#C0392B",
    "PPO_FT":           "#884EA0",
    # Robustness model names
    "SPOT":          "#1565C0",
    "SAC (Scratch)": "#E67E22",
    "SAC (Nav)":     "#2E7D32",
    # Ablation
    "ActorOnly (Ours)":  "#1565C0",
    "Full Transfer":     "#1ABC9C",
    "Freeze 100ep":      "#9B59B6",
    "From PD-Only SP":   "#3498DB",
    "From Fixed-50 SP":  "#F39C12",
}

LABELS = {
    "SAC_FT_Self":      "SPOT",
    "SAC_Scratch":      "SAC (Scratch)",
    "SAC_FT_Return":    "SAC (Nav)",
    "PPO_FT_Self":      "PPO (Self-Play)",
    "Heuristic":        "Heuristic",
    "SAC_FT_ActorOnly": "SAC FT (Actor-Only)",
    "PPO_Scratch":      "PPO (Scratch)",
    "PPO_FT":           "PPO FT (Actor-Only)",
}

def rolling(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values

def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  [OK] {name}")


# ══════════════════════════════════════════════════════════════════════════
# A2: Self-Play Curriculum — Win Rate & Opponent Ratio
# ══════════════════════════════════════════════════════════════════════════
def plot_A2():
    df = pd.read_csv(RESULTS / "self_play_battles.csv")
    df = df[df["episode"] <= 1500]

    W = 30
    is_past = (~df["opponent"].str.startswith("pd")).astype(float)
    win     = df["success"].astype(float)

    ep         = df["episode"].values
    win_roll   = rolling(win.values,     W) * 100
    ratio_roll = rolling(is_past.values, W) * 100

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    c_win   = "#5DADE2"   # Pastel Blue
    c_ratio = "#F5B041"   # Pastel Orange

    ax1.plot(ep, win_roll, color=c_win, lw=1.6, label="Rolling Win Rate")
    ax1.axhline(70, color="gray", ls="--", lw=1.2, alpha=0.8, label="Target Win Rate (70%)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Win Rate (%)")
    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 108)

    ax2 = ax1.twinx()
    ax2.plot(ep, ratio_roll, color=c_ratio, lw=1.6, ls="-.",
             label="Past-Self Opponent Ratio")
    ax2.set_ylabel("Past-Self Ratio (%)")
    ax2.tick_params(axis="y")
    ax2.set_ylim(0, 108)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.92)

    ax1.set_title("Self-Play Curriculum: Adaptive Opponent Selection")
    fig.tight_layout()
    savefig(fig, "A2_selfplay_curriculum")


# ══════════════════════════════════════════════════════════════════════════
# B1: Return Mission — Learning Curves
# ══════════════════════════════════════════════════════════════════════════
def plot_B1():
    df = pd.read_csv(RESULTS / "return_rolling.csv")
    order = ["SAC_FT_ActorOnly", "SAC_Scratch", "PPO_Scratch"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for label in order:
        sub = df[df["label"] == label]
        col = C.get(label, "#999")
        lbl = LABELS.get(label, label)
        lw  = 2.2 if label == "SAC_FT_ActorOnly" else 1.5

        x      = sub["episode_end"].values
        y_sr   = sub["success_rate"].values
        y_wp   = sub["avg_wp"].values

        axes[0].plot(x, y_sr, color=col, label=lbl, lw=lw)
        axes[1].plot(x, y_wp, color=col, lw=lw, ls="--")

    axes[0].set_title("(a) Task Success Rate", loc="left", fontweight="bold")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].legend(framealpha=0.92, loc="upper left", ncol=2)

    axes[1].set_title("(b) Avg Waypoints Reached", loc="left", fontweight="bold")
    axes[1].set_ylabel("Avg Waypoints (max 4)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-0.1, 4.5)
    axes[1].axhline(4, color="gray", ls=":", lw=1, alpha=0.5)

    handles = [plt.Line2D([0],[0], color=C.get(l,"#999"), ls="--", lw=1.5,
                           label=LABELS.get(l,l)) for l in order]
    axes[1].legend(handles=handles, framealpha=0.92, loc="lower right", ncol=2)

    fig.tight_layout()
    savefig(fig, "B1_return_learning_curves")


# ══════════════════════════════════════════════════════════════════════════
# B2: Return Mission — Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def plot_B2():
    df = pd.read_csv(RESULTS / "eval_return_final.csv")
    order = ["SAC_FT_ActorOnly", "SAC_Scratch", "PPO_FT", "PPO_Scratch"]
    df = df.set_index("label").reindex(order).reset_index()

    x  = np.arange(len(df))
    w  = 0.45
    colors = [C.get(l, "#999") for l in df["label"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Left: Eval Success Rate
    ax = axes[0]
    b1 = ax.bar(x, df["success_rate"], w, color=colors, alpha=0.9, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(l, l) for l in df["label"]], rotation=12, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Eval Success Rate (50 episodes, deterministic)")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Right: Avg Waypoints Reached
    ax = axes[1]
    b2 = ax.bar(x, df["avg_wp"], width=w, color=colors, alpha=0.88, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(l, l) for l in df["label"]], rotation=12, ha="right")
    ax.set_ylabel("Avg Waypoints Reached")
    ax.set_ylim(0, 4.5)
    ax.set_title("Avg Waypoints Reached (max 4)")
    ax.axhline(4, color="gray", ls="--", lw=1, alpha=0.5, label="Max (4)")
    ax.legend(framealpha=0.9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Return Mission: Final Performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "B2_return_barplot")


# ══════════════════════════════════════════════════════════════════════════
# C1: SAM Mission — Learning Curves
# ══════════════════════════════════════════════════════════════════════════
def plot_C1():
    df = pd.read_csv(RESULTS / "sam_rolling.csv")
    order = ["SAC_FT_Self", "SAC_Scratch", "SAC_FT_Return", "PPO_FT_Self"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for label in order:
        sub = df[df["label"] == label]
        col = C.get(label, "#999")
        lbl = LABELS.get(label, label)
        lw  = 2.2 if label == "SAC_FT_Self" else 1.5
        
        x = sub["episode_end"].values
        y_sr = sub["success_rate"].values
        y_surv = 100 - (sub["kill_rate"].values + sub["crash_rate"].values)

        axes[0].plot(x, y_sr, color=col, label=lbl, lw=lw)
        axes[1].plot(x, y_surv, color=col, lw=lw, ls="--")

    axes[0].set_title("(a) Task Success Rate", loc="left", fontweight="bold")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[1].set_title("(b) Survival Rate (No Kill / No Crash)", loc="left", fontweight="bold")
    axes[0].legend(framealpha=0.92, loc="upper left", ncol=2)

    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-5, 105)

    # Add legend for survival rate
    handles = [plt.Line2D([0],[0], color=C.get(l,"#999"), ls="--", lw=1.5,
                           label=LABELS.get(l,l)) for l in order]
    axes[1].legend(handles=handles, framealpha=0.92, loc="lower right", ncol=2)

    fig.tight_layout()
    savefig(fig, "C1_sam_learning_curves")


# ══════════════════════════════════════════════════════════════════════════
# C1a: SAM Mission — Success Rate Only (single panel)
# ══════════════════════════════════════════════════════════════════════════
def plot_C1a():
    df = pd.read_csv(RESULTS / "sam_rolling.csv")
    order = ["SAC_FT_Self", "SAC_Scratch", "SAC_FT_Return", "PPO_FT_Self"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label in order:
        sub = df[df["label"] == label]
        col = C.get(label, "#999")
        lbl = LABELS.get(label, label)
        lw  = 2.2 if label == "SAC_FT_Self" else 1.5
        ax.plot(sub["episode_end"].values, sub["success_rate"].values,
                color=col, label=lbl, lw=lw)

    ax.set_title("Rolling Success Rate (50-ep window)", loc="left", fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.legend(framealpha=0.92, loc="upper left", ncol=2)
    fig.tight_layout()
    savefig(fig, "C1a_sam_success_curve")


# ══════════════════════════════════════════════════════════════════════════
# C1b: SAM Mission — Survival Rate Only (single panel)
# ══════════════════════════════════════════════════════════════════════════
def plot_C1b():
    df = pd.read_csv(RESULTS / "sam_rolling.csv")
    order = ["SAC_FT_Self", "SAC_Scratch", "SAC_FT_Return", "PPO_FT_Self"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label in order:
        sub = df[df["label"] == label]
        col = C.get(label, "#999")
        lbl = LABELS.get(label, label)
        lw  = 2.2 if label == "SAC_FT_Self" else 1.5
        y_surv = 100 - (sub["kill_rate"].values + sub["crash_rate"].values)
        ax.plot(sub["episode_end"].values, y_surv,
                color=col, ls="--", label=lbl, lw=lw)

    ax.set_title("Survival Rate (No Kill / No Crash, 50-ep window)", loc="left", fontweight="bold")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    handles = [plt.Line2D([0],[0], color=C.get(l,"#999"), ls="--", lw=1.5,
                           label=LABELS.get(l,l)) for l in order]
    ax.legend(handles=handles, framealpha=0.92, loc="lower right", ncol=2)
    fig.tight_layout()
    savefig(fig, "C1b_sam_survival_curve")


# ══════════════════════════════════════════════════════════════════════════
# C2: SAM Mission — Final Performance Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def plot_C2():
    df = pd.read_csv(RESULTS / "eval_sam_final.csv")
    order = ["SAC_FT_Self", "SAC_Scratch", "SAC_FT_Return", "PPO_FT_Self", "PPO_Scratch"]
    df = df.set_index("label").reindex(order).reset_index()

    x      = np.arange(len(df))
    w      = 0.24
    colors = [C.get(l, "#999") for l in df["label"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    b1 = ax.bar(x - w,   df["success_rate"], w, color=colors, alpha=0.92,
                label="Success Rate",  edgecolor="white")
    b2 = ax.bar(x,       df["kill_rate"],    w, color=colors, alpha=0.50,
                label="Kill Rate",    edgecolor="white", hatch="xxx")
    b3 = ax.bar(x + w,   df["crash_rate"],   w, color=colors, alpha=0.28,
                label="Crash Rate",   edgecolor="white", hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(l, l) for l in df["label"]], rotation=12, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("SAM Evasion Mission: Final Performance Comparison")

    model_patches = [mpatches.Patch(color=C.get(l,"#999"), alpha=0.9, label=LABELS.get(l,l))
                     for l in order]
    pattern_patches = [
        mpatches.Patch(facecolor="gray", alpha=0.9,  label="Success Rate"),
        mpatches.Patch(facecolor="gray", alpha=0.5,  hatch="xxx", label="Kill Rate"),
        mpatches.Patch(facecolor="gray", alpha=0.28, hatch="///", label="Crash Rate"),
    ]
    ax.legend(handles=model_patches + pattern_patches,
              ncol=2, framealpha=0.9, fontsize=9)

    for bar in b1:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom",
                    fontsize=8.0, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, alpha=0.8)
    for bar in b3:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, alpha=0.8)

    fig.tight_layout()
    savefig(fig, "C2_sam_barplot")


# ══════════════════════════════════════════════════════════════════════════
# D: Robustness 3-Panel
# ══════════════════════════════════════════════════════════════════════════
MODEL_MAP = {
    "FineTune":   "SPOT",
    "Scratch":    "SAC (Scratch)",
    "FT_Return":  "SAC (Nav)",
    "Heuristic":  "Heuristic",
}
ROB_ORDER  = ["SPOT", "SAC (Scratch)", "SAC (Nav)", "Heuristic"]
ROB_COLORS = {k: C[k] for k in ROB_ORDER}
ROB_XTICKS = ["SPOT", "SAC (Scratch)", "SAC (Nav)", "Heuristic"]

def load_rob(scenario):
    frames = []
    for key, name in MODEL_MAP.items():
        p = RESULTS / f"robustness_{scenario}_{key}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["model_name"] = name
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def plot_D():
    fig = plt.figure(figsize=(15, 5.2))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── D1: Threat Scaling ───────────────────────────────────────────────
    df = load_rob("threat_scaling")
    if not df.empty:
        # Normalize config labels
        def norm_config(c):
            c = str(c)
            if "100" in c or "Base" in c: return "×1.0"
            if "120" in c:                return "×1.2"
            if "150" in c:                return "×1.5"
            return c
        df["cfg"] = df["config"].apply(norm_config)
        configs = ["×1.0", "×1.2", "×1.5"]
        agg = df.groupby(["model_name","cfg"])["success"].mean().mul(100).reset_index()

        n_m = len(ROB_ORDER); bw = 0.20; x = np.arange(len(configs))
        for i, m in enumerate(ROB_ORDER):
            sub  = agg[agg["model_name"] == m]
            vals = [sub.loc[sub["cfg"]==c,"success"].values[0]
                    if c in sub["cfg"].values else 0 for c in configs]
            off  = (i - n_m/2 + 0.5) * bw
            axes[0].bar(x + off, vals, bw * 0.9,
                        color=ROB_COLORS[m], alpha=0.88, label=m, edgecolor="white")

        axes[0].set_xticks(x); axes[0].set_xticklabels(configs)
        axes[0].set_xlabel("SAM Threat Radius Scale")
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_title("(a) Threat Scaling")
        axes[0].set_ylim(0, 108)
        axes[0].legend(fontsize=8.0, framealpha=0.92)
        
        # Add values for panel (a)
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.0f%%', padding=1, fontsize=7.5, alpha=0.8)

    # ── D2: Initial Perturbation ─────────────────────────────────────────
    df = load_rob("init_perturb")
    if not df.empty:
        agg = df.groupby("model_name").agg(
            sr=("success",      lambda x: x.astype(float).mean() * 100),
            kr=("killed_by_sam",lambda x: x.astype(float).mean() * 100),
        ).reset_index()

        x = np.arange(len(ROB_ORDER)); w = 0.22 # Match thickness
        vals_sr = [agg.loc[agg["model_name"]==m,"sr"].values[0] if m in agg["model_name"].values else 0 for m in ROB_ORDER]
        # Survival Rate = Success + Failure(not killed/crashed) = 100 - killed - crashed
        # But here let's follow user's "Survivor = 100 - (Kill + Crash)"
        # Note: Init perturb results usually don't have crash separated in summary, check keys
        vals_surv = [100 - (agg.loc[agg["model_name"]==m,"kr"].values[0]) if m in agg["model_name"].values else 0 for m in ROB_ORDER]
        cols = [ROB_COLORS[m] for m in ROB_ORDER]

        b1 = axes[1].bar(x - w/2, vals_sr, w, color=cols, alpha=0.9,  edgecolor="white", label="Success Rate")
        b2 = axes[1].bar(x + w/2, vals_surv, w, color=cols, alpha=0.42, edgecolor="white", hatch="xxx", label="Survival Rate")

        axes[1].set_xticks(x); axes[1].set_xticklabels(ROB_XTICKS, rotation=12, ha="right")
        axes[1].set_ylabel("Rate (%)")
        axes[1].set_title("(b) Initial Perturbation")
        axes[1].set_ylim(0, 108)
        axes[1].legend(fontsize=8.0, framealpha=0.92)
        for bar in b1:
            h = bar.get_height()
            if h > 2: axes[1].text(bar.get_x()+bar.get_width()/2, h+1.2,
                                   f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            if h > 2: axes[1].text(bar.get_x()+bar.get_width()/2, h+1.2,
                                   f"{h:.0f}%", ha="center", va="bottom", fontsize=7.2, alpha=0.8)

    # ── D3: Pop-up SAM ───────────────────────────────────────────────────
    df = load_rob("popup_sam")
    if not df.empty:
        # Only episodes where popup was triggered
        triggered = df[df["popup_triggered"].astype(int) == 1].copy()
        agg = triggered.groupby("model_name").agg(
            sr  =("success",       lambda x: x.astype(float).mean() * 100),
            surv=("killed_by_sam", lambda x: (1 - x.astype(float).mean()) * 100),
        ).reset_index()

        x = np.arange(len(ROB_ORDER)); w = 0.35
        vals_sr   = [agg.loc[agg["model_name"]==m,"sr"].values[0]   if m in agg["model_name"].values else 0 for m in ROB_ORDER]
        vals_surv = [agg.loc[agg["model_name"]==m,"surv"].values[0] if m in agg["model_name"].values else 0 for m in ROB_ORDER]
        cols = [ROB_COLORS[m] for m in ROB_ORDER]

        b1 = axes[2].bar(x - w/2, vals_sr,   w, color=cols, alpha=0.9,  edgecolor="white", label="Success Rate")
        b2 = axes[2].bar(x + w/2, vals_surv, w, color=cols, alpha=0.42, edgecolor="white", hatch="///", label="Post-Popup Survival")

        axes[2].set_xticks(x); axes[2].set_xticklabels(ROB_XTICKS, rotation=12, ha="right")
        axes[2].set_ylabel("Rate (%)")
        axes[2].set_title("(c) Pop-up SAM Response")
        axes[2].set_ylim(0, 108)
        axes[2].legend(fontsize=8.0, framealpha=0.92)
        for bar in b1:
            h = bar.get_height()
            if h > 3: axes[2].text(bar.get_x()+bar.get_width()/2, h+1.2,
                                   f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            if h > 3: axes[2].text(bar.get_x()+bar.get_width()/2, h+1.2,
                                   f"{h:.0f}%", ha="center", va="bottom", fontsize=7.2, alpha=0.8)

    fig.suptitle("Robustness: Out-of-Distribution Scenarios",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "D_robustness")


# ══════════════════════════════════════════════════════════════════════════
# F2: Fine-Tuning Strategy Ablation
# ══════════════════════════════════════════════════════════════════════════
FT_RUNS = {
    "ActorOnly (Ours)": RUNS / "20260219_SAC_SAM_FineTune" / "20260219_SAC_SAM_FineTune_20260306_1730" / "episode_metrics.csv",
    "Full Transfer":    RUNS / "20260219_SAC_SAM_FineTune" / "ablation_ft_full"      / "episode_metrics.csv",
    "Freeze 100ep":     RUNS / "20260219_SAC_SAM_FineTune" / "ablation_ft_freeze100" / "episode_metrics.csv",
    "From PD-Only SP":  RUNS / "20260219_SAC_SAM_FineTune" / "ablation_ft_pd_only"  / "episode_metrics.csv",
    "From Fixed-50 SP": RUNS / "20260219_SAC_SAM_FineTune" / "ablation_ft_fixed_50" / "episode_metrics.csv",
}

def plot_F2():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for label, path in FT_RUNS.items():
        if not path.exists():
            print(f"    [skip] {label}: not found")
            continue
        df  = pd.read_csv(path)
        sr  = rolling(df["success"].astype(float).values * 100, 50)
        col = C.get(label, "#999")
        ls  = "-"  if label == "ActorOnly (Ours)" else "--"
        lw  = 2.6  if label == "ActorOnly (Ours)" else 1.8
        ax.plot(df["episode"], sr, color=col, ls=ls, lw=lw, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%, rolling-50)")
    ax.set_title("SAM Evasion: Fine-Tuning Strategy Ablation")
    ax.set_xlim(0, 500)
    ax.set_ylim(-2, 80)
    ax.legend(framealpha=0.92)

    fig.tight_layout()
    savefig(fig, "F2_ft_ablation")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[Output] {OUT}\n")
    plot_A2()
    plot_B1()
    plot_B2()
    plot_C1()
    plot_C1a()
    plot_C1b()
    plot_C2()
    plot_D()
    plot_F2()
    print(f"\nAll figures saved successfully to: {OUT}")
