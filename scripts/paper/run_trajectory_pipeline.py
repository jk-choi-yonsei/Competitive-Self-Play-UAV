#!/usr/bin/env python3
"""
Trajectory collection + visualization full pipeline
Output: paper/figures/traj_*.pdf, traj_*.png
"""

import subprocess
import sys
import csv
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable
PAPER  = ROOT / "paper" / "paper_figures"
TDATA  = ROOT / "figures" / "paper_figures" / "traj_data"
PAPER.mkdir(parents=True, exist_ok=True)

def run(args, **kwargs):
    cmd = [PYTHON] + args
    print(f"\n$ {' '.join(str(a) for a in args)}")
    r = subprocess.run(cmd, cwd=str(ROOT), **kwargs)
    return r

def read_summary(traj_dir):
    """Return list of {episode, seed, success, killed, crashed, ...}"""
    p = Path(traj_dir) / "summary.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))

def find_good_seed(traj_dir, prefer="success"):
    rows = read_summary(traj_dir)
    if not rows:
        return None, None
    for r in rows:
        if int(r["success"]):
            return int(r["episode"]), int(r["seed"])
    # fallback: killed (still reached SAM zone at least)
    for r in rows:
        if int(r["killed"]):
            return int(r["episode"]), int(r["seed"])
    return int(rows[0]["episode"]), int(rows[0]["seed"])


# ═══════════════════════════════════════════════════════════════════════════
# 1. ft_self standard scenario: Run 50 episodes to ensure success cases
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: ft_self standard (50 eps) - finding success episodes")
print("="*60)

run(["scripts/paper/eval_trajectory.py",
     "--model", "ft_self", "--n", "50", "--seed", "100"])

rows_ft = read_summary(TDATA / "ft_self_standard_seed100")
success_rows = [r for r in rows_ft if int(r["success"])]
print(f"\nSuccess episodes: {[r['episode'] for r in success_rows]}")

# Choose a median success episode (avoid too early or extreme cases)
if len(success_rows) >= 3:
    chosen = success_rows[len(success_rows) // 2]
elif success_rows:
    chosen = success_rows[0]
else:
    # If no success, choose the one with most progress (min dist_to_goal)
    killed_rows = [r for r in rows_ft if int(r["killed"])]
    chosen = killed_rows[0] if killed_rows else rows_ft[0]
    print("[WARN] No success found, using best killed episode")

ep_num = int(chosen["episode"])
seed   = int(chosen["seed"])
print(f"\n=> Chosen episode {ep_num} (seed={seed}, "
      f"success={chosen['success']}, killed={chosen['killed']})")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Run scratch and ft_return with the same seed
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"STEP 2: scratch & ft_return with seed={seed}")
print("="*60)

run(["scripts/paper/eval_trajectory.py",
     "--model", "scratch",   "--n", "1", "--seed", str(seed)])
run(["scripts/paper/eval_trajectory.py",
     "--model", "ft_return", "--n", "1", "--seed", str(seed)])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pop-up SAM: Find success cases in 30 episodes of ft_self
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: popup SAM scenarios")
print("="*60)

run(["scripts/paper/eval_trajectory.py",
     "--model", "ft_self", "--mode", "popup", "--n", "30", "--seed", "200"])

rows_popup = read_summary(TDATA / "ft_self_popup_seed200")
suc_popup  = [r for r in rows_popup if int(r["success"])]
popup_chosen = (suc_popup[len(suc_popup)//2] if len(suc_popup) >= 2
                else suc_popup[0] if suc_popup
                else rows_popup[0])
popup_seed = int(popup_chosen["seed"])
print(f"\n=> Popup chosen: ep={popup_chosen['episode']}, seed={popup_seed}")

run(["scripts/paper/eval_trajectory.py",
     "--model", "scratch", "--mode", "popup", "--n", "1", "--seed", str(popup_seed)])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Visualization: Call plot_trajectory.py (save to paper/figures/)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: Generating trajectory figures -> paper/figures/")
print("="*60)

# Override OUT in plot_trajectory.py to paper/figures
import importlib.util, types, matplotlib
matplotlib.use("Agg")

spec = importlib.util.spec_from_file_location(
    "plot_traj", ROOT / "scripts" / "paper" / "plot_trajectory.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)
pt.OUT = PAPER   # redirect output

def save(name):
    return name  # already using PAPER as OUT

# ── 4a. 3-model comparison (main result) ─────────────────────────────────
csv_ft  = TDATA / "ft_self_standard_seed100"          / f"episode_{ep_num:03d}.csv"
csv_sc  = TDATA / f"scratch_standard_seed{seed}"   / "episode_000.csv"
csv_ret = TDATA / f"ft_return_standard_seed{seed}"  / "episode_000.csv"

if csv_ft.exists() and csv_sc.exists() and csv_ret.exists():
    pt.plot_compare(
        [csv_ft, csv_sc, csv_ret],
        ["SPOT", "SAC Scratch", "SAC FT (Return)"],
        save_name="traj_sam_comparison",
    )
else:
    print(f"[WARN] Missing CSVs for comparison: ft={csv_ft.exists()} sc={csv_sc.exists()} ret={csv_ret.exists()}")

# ── 4b. ft_self 성공 단독 (3D + top-down) ────────────────────────────────
if csv_ft.exists():
    pt.plot_single(csv_ft,
                   title="SPOT — SAM Evasion",
                   save_name="traj_ft_self_success")

# ── 4c. scratch 실패 단독 ─────────────────────────────────────────────────
if csv_sc.exists():
    pt.plot_single(csv_sc,
                   title="SAC Scratch — SAM Evasion",
                   save_name="traj_scratch_failure")

# ── 4d. Pop-up SAM comparison ─────────────────────────────────────────────
csv_pu_ft = TDATA / f"ft_self_popup_seed{popup_seed}"   / "episode_000.csv"
csv_pu_sc = TDATA / f"scratch_popup_seed{popup_seed}"   / "episode_000.csv"

if csv_pu_ft.exists() and csv_pu_sc.exists():
    pt.plot_compare(
        [csv_pu_ft, csv_pu_sc],
        ["SPOT", "SAC Scratch"],
        save_name="traj_popup_comparison",
    )
elif csv_pu_ft.exists():
    pt.plot_single(csv_pu_ft,
                   title="SPOT — Pop-up SAM",
                   save_name="traj_popup_ft_self")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Print final file list
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DONE. Files in paper/paper_figures/:")
print("="*60)
for f in sorted(PAPER.iterdir()):
    print(f"  {f.name}")
