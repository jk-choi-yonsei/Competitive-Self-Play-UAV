"""
20260320_SAC_Ablation_CkptSelection
  Self-play checkpoint selection criteria ablation
  ep 300 / 600 / 1200 / 1500 -> SAM fine-tune 500 ep each (ep 900 = ours, baseline)
  Result aggregation: results/ablation_ckpt_selection.csv

Usage:
  # Execute 5 runs sequentially and aggregate automatically
  python scripts/20260320_SAC_Ablation_CkptSelection.py

  # Aggregate only without execution (if runs are already complete)
  python scripts/20260320_SAC_Ablation_CkptSelection.py --aggregate_only
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    {"ckpt_ep": 300,  "run_name": "ablation_ep300",  "ckpt": "models/20260109_SAC_Self_Play/selfplay_epi_00300.pth"},
    {"ckpt_ep": 600,  "run_name": "ablation_ep600",  "ckpt": "models/20260109_SAC_Self_Play/selfplay_epi_00600.pth"},
    {"ckpt_ep": 1200, "run_name": "ablation_ep1200", "ckpt": "models/20260109_SAC_Self_Play/selfplay_epi_01200.pth"},
    {"ckpt_ep": 1500, "run_name": "ablation_ep1500", "ckpt": "models/20260109_SAC_Self_Play/selfplay_epi_01500.pth"},
]

FINETUNE_SCRIPT = PROJECT_ROOT / "scripts" / "20260219_SAC_SAM_FineTune.py"
METRICS_BASE = PROJECT_ROOT / "runs" / "20260219_SAC_SAM_FineTune"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "ablation_ckpt_selection.csv"
LAST_N = 50


def run_finetune(run_cfg: dict) -> None:
    ckpt_path = PROJECT_ROOT / run_cfg["ckpt"]
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint not found: {ckpt_path}  -- skipping run {run_cfg['run_name']}")
        return

    cmd = [
        sys.executable, str(FINETUNE_SCRIPT),
        "--episodes", "500",
        "--transfer_mode", "actor_only",
        "--init_from", str(ckpt_path),
        "--run_name", run_cfg["run_name"],
    ]
    print(f"\n{'='*60}")
    print(f"[RUN] {run_cfg['run_name']}  (ckpt ep {run_cfg['ckpt_ep']})")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] run {run_cfg['run_name']} exited with code {result.returncode}")


def aggregate() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_cfg in RUNS:
        run_name = run_cfg["run_name"]
        ckpt_ep  = run_cfg["ckpt_ep"]

        # episode_metrics.csv path
        metrics_csv = METRICS_BASE / run_name / "episode_metrics.csv"
        if not metrics_csv.exists():
            print(f"[WARN] Metrics not found for {run_name}: {metrics_csv}")
            rows.append({
                "transfer_ckpt": f"ep {ckpt_ep}",
                "run_name": run_name,
                "last50_success_rate": "",
                "last50_kill_rate": "",
                "last50_crash_rate": "",
                "last50_avg_reward": "",
                "n_episodes": "",
            })
            continue

        with metrics_csv.open(newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        if not reader:
            print(f"[WARN] Empty metrics CSV for {run_name}")
            rows.append({
                "transfer_ckpt": f"ep {ckpt_ep}",
                "run_name": run_name,
                "last50_success_rate": "",
                "last50_kill_rate": "",
                "last50_crash_rate": "",
                "last50_avg_reward": "",
                "n_episodes": 0,
            })
            continue

        last = reader[-LAST_N:]
        n = len(last)
        success_rate = sum(int(r["success"]) for r in last) / n * 100
        kill_rate    = sum(int(r["killed_by_sam"]) for r in last) / n * 100
        crash_rate   = sum(int(r["crashed"]) for r in last) / n * 100
        avg_reward   = sum(float(r["score"]) for r in last) / n

        label = f"ep {ckpt_ep}"
        rows.append({
            "transfer_ckpt": label,
            "run_name": run_name,
            "last50_success_rate": f"{success_rate:.1f}",
            "last50_kill_rate": f"{kill_rate:.1f}",
            "last50_crash_rate": f"{crash_rate:.1f}",
            "last50_avg_reward": f"{avg_reward:.2f}",
            "n_episodes": len(reader),
        })
        print(f"  {label:15s} | SR={success_rate:5.1f}% | Kill={kill_rate:5.1f}% | Crash={crash_rate:5.1f}% | AvgR={avg_reward:7.2f}  (last {n} / total {len(reader)} ep)")

    # Write summary CSV
    fieldnames = ["transfer_ckpt", "run_name", "last50_success_rate",
                  "last50_kill_rate", "last50_crash_rate", "last50_avg_reward", "n_episodes"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[Done] Summary saved to {OUTPUT_CSV}")

    # Pretty table
    print()
    print(f"{'Transfer ckpt':<16} {'Last-50 SR%':>12} {'Kill%':>8} {'Crash%':>8} {'AvgReward':>11}")
    print("-" * 60)
    for r in rows:
        print(f"{r['transfer_ckpt']:<16} {r['last50_success_rate']:>12} {r['last50_kill_rate']:>8} {r['last50_crash_rate']:>8} {r['last50_avg_reward']:>11}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip training; only aggregate existing results.")
    args = parser.parse_args()

    if not args.aggregate_only:
        for run_cfg in RUNS:
            run_finetune(run_cfg)

    print("\n[Aggregate] Computing last-50 statistics...\n")
    aggregate()
