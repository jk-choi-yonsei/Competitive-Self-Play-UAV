"""Return 미션 궤적 그림 생성 → paper/paper_figures/"""
import argparse
import csv, importlib.util, matplotlib, sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent.parent
PAPER = ROOT / "paper" / "paper_figures"
TDATA = ROOT / "figures" / "paper_figures" / "traj_data"
PAPER.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--contrast_ep", type=int, default=None, help="Episode for Contrast (Success vs Fail) plot")
ap.add_argument("--compare_ep", type=int, default=None, help="Episode for Efficiency comparison (Both Success) plot")
args = ap.parse_args()

matplotlib.use("Agg")
spec = importlib.util.spec_from_file_location("plot_ret", ROOT / "scripts" / "paper" / "plot_return_trajectory.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)
pr.OUT = PAPER

def read_summary(path):
    with open(path) as f:
        return list(csv.DictReader(f))

ft_rows = read_summary(TDATA / "ft_return_seed42" / "summary.csv")
sc_rows = read_summary(TDATA / "scratch_seed42"   / "summary.csv")

# ── 1. Contrast 비교 (ft_return 성공 vs scratch 실패) ────────────────────────
if args.contrast_ep is not None:
    contrast_ep = args.contrast_ep
    print(f"사용자 선택 (Contrast): ep={contrast_ep}")
else:
    contrast_ep = None
    for r in ft_rows:
        ep = int(r["episode"])
        ft_ok = int(r["success"])
        # Find if scratch failed in this episode
        sc_r = next((row for row in sc_rows if int(row["episode"]) == ep), None)
        if ft_ok and sc_r and not int(sc_r["success"]):
            contrast_ep = ep
            print(f"자동 선택 (Contrast): ep={ep} (ft=SUCCESS, sc=FAIL)")
            break

if contrast_ep is not None:
    csv_ft = TDATA / "ft_return_seed42" / f"episode_{contrast_ep:03d}.csv"
    csv_sc = TDATA / "scratch_seed42"   / f"episode_{contrast_ep:03d}.csv"
    if csv_ft.exists() and csv_sc.exists():
        pr.plot_compare([csv_ft, csv_sc],
                        ["SAC FT (Return)", "SAC Scratch"],
                        save_name="traj_return_contrast")
        print("[OK] traj_return_contrast")

# ── 2. 둘 다 성공: 효율 비교 ────────────────────────────────────────────────
if args.compare_ep is not None:
    compare_ep = args.compare_ep
    print(f"사용자 선택 (Compare): ep={compare_ep}")
else:
    both_ok = []
    for r in ft_rows:
        ep = int(r["episode"])
        ft_ok = int(r["success"])
        sc_r = next((row for row in sc_rows if int(row["episode"]) == ep), None)
        if ft_ok and sc_r and int(sc_r["success"]):
            both_ok.append(ep)
    compare_ep = both_ok[len(both_ok)//2] if both_ok else 0
    print(f"자동 선택 (Compare): ep={compare_ep} (둘 다 성공)")

csv_ft_mid = TDATA / "ft_return_seed42" / f"episode_{compare_ep:03d}.csv"
csv_sc_mid = TDATA / "scratch_seed42"   / f"episode_{compare_ep:03d}.csv"
if csv_ft_mid.exists() and csv_sc_mid.exists():
    pr.plot_compare([csv_ft_mid, csv_sc_mid],
                    ["SAC FT (Return)", "SAC Scratch"],
                    save_name="traj_return_compare")
    print("[OK] traj_return_compare")

# ── 3. ft_return 성공 단독 ───────────────────────────────────────────────────
csv_ft_0 = TDATA / "ft_return_seed42" / "episode_000.csv"
if csv_ft_0.exists():
    pr.plot_single(csv_ft_0,
                   title="SAC FT (Return) — 4-Waypoint Navigation",
                   save_name="traj_return_ft_success")
    print("[OK] traj_return_ft_success")

print("\nDone. Files in paper/paper_figures/:")
for f in sorted(PAPER.iterdir()):
    if "return" in f.name:
        print(f"  {f.name}")
