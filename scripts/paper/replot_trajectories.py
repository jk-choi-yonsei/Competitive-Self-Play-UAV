"""STEP 4만 다시 실행: 기존 traj_data로 그림 재생성"""
import argparse
import importlib.util, matplotlib, csv
from pathlib import Path
import sys

ROOT  = Path(__file__).resolve().parent.parent.parent
PAPER = ROOT / "paper" / "paper_figures"
TDATA = ROOT / "figures" / "paper_figures" / "traj_data"
PAPER.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, default=None, help="Episode number for Standard SAM comparison")
ap.add_argument("--popup_ep", type=int, default=None, help="Episode number for Pop-up SAM comparison")
args = ap.parse_args()

matplotlib.use("Agg")
spec = importlib.util.spec_from_file_location("plot_traj", ROOT / "scripts" / "paper" / "plot_trajectory.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)
pt.OUT = PAPER

# ── 1. ft_self 성공 에피소드 찾기 ───────────────────────────────────────────
summary_path = TDATA / "ft_self_standard_seed100" / "summary.csv"
with open(summary_path) as f:
    rows = list(csv.DictReader(f))

success_rows = [r for r in rows if int(r["success"])]
print(f"ft_self 성공 에피소드: {[r['episode'] for r in success_rows]}")

if args.ep is not None:
    # User specified episode
    ep_num = args.ep
    # Find seed from summary for the chosen episode
    chosen = next((r for r in rows if int(r["episode"]) == ep_num), None)
    if chosen:
        seed = int(chosen["seed"])
        print(f"사용자 선택 (Standard): ep={ep_num}, seed={seed}")
    else:
        print(f"[ERR] Episode {ep_num} not found in summary. Standard plot may fail.")
        seed = 100 # fallback
else:
    chosen = success_rows[len(success_rows) // 2]
    ep_num = int(chosen["episode"])
    seed   = int(chosen["seed"])
    print(f"자동 선택 (Standard): ep={ep_num}, seed={seed} (중간값)")

# CSV 경로
csv_ft  = TDATA / "ft_self_standard_seed100"       / f"episode_{ep_num:03d}.csv"
csv_sc  = TDATA / f"scratch_standard_seed{seed}"   / "episode_000.csv"
csv_ret = TDATA / f"ft_return_standard_seed{seed}" / "episode_000.csv"

# 4a. 3-model 비교
if csv_ft.exists() and csv_sc.exists() and csv_ret.exists():
    pt.plot_compare(
        [csv_ft, csv_sc, csv_ret],
        ["SPOT", "SAC (Scratch)", "SAC (Nav)"],
        save_name="traj_sam_comparison",
    )
    print("[OK] traj_sam_comparison")
else:
    print(f"[WARN] 비교 CSV 누락: ft={csv_ft.exists()} sc={csv_sc.exists()} ret={csv_ret.exists()}")

# 4b. ft_self 성공 단독
if csv_ft.exists():
    pt.plot_single(csv_ft, title="SPOT — SAM Evasion",
                   save_name="traj_ft_self_success")
    print("[OK] traj_ft_self_success")

# 4c. scratch 실패 단독
if csv_sc.exists():
    pt.plot_single(csv_sc, title="SAC (Scratch) — SAM Evasion",
                   save_name="traj_scratch_failure")
    print("[OK] traj_scratch_failure")


# ── 2. Popup SAM 비교 ───────────────────────────────────────────────────────
popup_summary = TDATA / "ft_self_popup_seed200" / "summary.csv"
with open(popup_summary) as f:
    popup_rows = list(csv.DictReader(f))
suc_popup = [r for r in popup_rows if int(r["success"])]

if args.popup_ep is not None:
    popup_ep = args.popup_ep
    popup_chosen = next((r for r in popup_rows if int(r["episode"]) == popup_ep), None)
    if popup_chosen:
        popup_seed = int(popup_chosen["seed"])
        print(f"사용자 선택 (Popup): ep={popup_ep}, seed={popup_seed}")
    else:
        print(f"[ERR] Episode {popup_ep} not found in popup summary.")
        popup_seed = 200
else:
    popup_chosen = suc_popup[len(suc_popup)//2] if len(suc_popup) >= 2 else (suc_popup[0] if suc_popup else popup_rows[0])
    popup_ep   = int(popup_chosen["episode"])
    popup_seed = int(popup_chosen["seed"])
    print(f"자동 선택 (Popup): ep={popup_ep}, seed={popup_seed}")

csv_pu_ft = TDATA / "ft_self_popup_seed200"          / f"episode_{popup_ep:03d}.csv"
csv_pu_sc = TDATA / f"scratch_popup_seed{popup_seed}" / "episode_000.csv"

if csv_pu_ft.exists() and csv_pu_sc.exists():
    pt.plot_compare([csv_pu_ft, csv_pu_sc],
                    ["SPOT", "SAC (Scratch)"],
                    save_name="traj_popup_comparison")
    print("[OK] traj_popup_comparison")
elif csv_pu_ft.exists():
    pt.plot_single(csv_pu_ft, title="SPOT — Pop-up SAM",
                   save_name="traj_popup_ft_self")
    print("[OK] traj_popup_ft_self")

print("\nDONE. Files in paper/paper_figures/:")
for f in sorted(PAPER.iterdir()):
    print(f"  {f.name}")
