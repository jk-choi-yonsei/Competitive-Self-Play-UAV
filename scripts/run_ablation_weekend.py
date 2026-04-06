"""
Ablation Study Weekend Runner
==============================
Sequential execution of 6 experiments:

[Ablation 1 - Self-Play Ratio Strategy]
  1. SP_PD_Only   : Self-Play 1000ep, ratio_mode=pd_only  -> Best ELO checkpoint -> SAM FT
  2. SP_Fixed_50  : Self-Play 1000ep, ratio_mode=fixed 0.5 -> Best ELO checkpoint -> SAM FT

[Ablation 2 - Transfer Scope]
  3. FT_Full      : SAM FT (Actor+Critic transfer) - using existing selfplay_epi_00900.pth

[Ablation 3 - Actor Freeze Warmup]
  4. FT_Freeze100 : SAM FT (Actor 100ep freeze) - using existing selfplay_epi_00900.pth

Comparison Baseline (already exists):
  - SAC_FT_Self  : actor_only, no freeze, dynamic ratio SP, ep900  (ELO 1314)
  - SAC_Scratch  : no transfer

Checkpoint criteria:
  - Existing SAC experiment: selfplay_epi_00900.pth (ELO 1314, almost identical to overall peak 1319)
  - Ablation 2,3: same ep900 used
  - Ablation 1 SP variation: automatically select episode checkpoint with highest ELO from CSV after each training
    (Checkpoints saved every 20ep -> select highest ELO episode that is a multiple of 20)

Run: python scripts/run_ablation_weekend.py
"""

import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# -- Path Settings -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
MODELS_DIR   = PROJECT_ROOT / "models"
RUNS_DIR     = PROJECT_ROOT / "runs" / "SAC_Self_Play"   # self-play log storage location
LOGS_DIR     = PROJECT_ROOT / "ablation_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# Existing Self-Play checkpoint (based on previous SAC experiment = ep900, ELO 1314)
EXISTING_SP_CKPT = str(MODELS_DIR / "20260109_SAC_Self_Play" / "selfplay_epi_00900.pth")

SP_SCRIPT = str(SCRIPTS_DIR / "20260109_SAC_Self_Play.py")
FT_SCRIPT = str(SCRIPTS_DIR / "20260219_SAC_SAM_FineTune.py")

CKPT_SAVE_INTERVAL = 20   # self-play script's episode % 20 == 0 save interval

# -- Automatic Search for Best ELO Checkpoint ---------------------------------------------
def find_best_elo_ckpt(model_dir: Path, run_name: str, fallback_ep: int = 900) -> Path:
    """
    Read runs/SAC_Self_Play/<run_name>/elo_history.csv 
    and return the episode where the highest ELO actual checkpoint (.pth) exists.
    If CSV does not exist, return fallback_ep episode checkpoint.
    """
    elo_csv = RUNS_DIR / run_name / "elo_history.csv"

    if elo_csv.exists():
        rows = []
        with open(elo_csv, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        rows.append((int(float(row[0])), float(row[1])))
                    except ValueError:
                        continue

        # Sort by ELO in descending order and select episode with actual checkpoint
        rows.sort(key=lambda x: -x[1])
        for ep, elo in rows:
            # Checkpoints exist only at 20ep intervals
            snap_ep = (ep // CKPT_SAVE_INTERVAL) * CKPT_SAVE_INTERVAL
            ckpt = model_dir / f"selfplay_epi_{snap_ep:05d}.pth"
            if ckpt.exists():
                log(f"[BestELO] run={run_name}, best_ep={ep} (ELO={elo:.1f})"
                    f" -> snap_ep={snap_ep} -> {ckpt.name}")
                return ckpt
        log(f"[BestELO] {elo_csv} exists but no matching checkpoint. fallback ep{fallback_ep}")
    else:
        log(f"[BestELO] ELO CSV missing: {elo_csv}. fallback ep{fallback_ep}")

    fallback = model_dir / f"selfplay_epi_{fallback_ep:05d}.pth"
    if fallback.exists():
        return fallback

    # If neither exists, use the latest checkpoint
    all_ckpts = sorted(model_dir.glob("selfplay_epi_*.pth"))
    if all_ckpts:
        log(f"[BestELO] fallback ep{fallback_ep} missing. Using latest checkpoint: {all_ckpts[-1].name}")
        return all_ckpts[-1]

    raise FileNotFoundError(f"Checkpoint not found in: {model_dir}")


# -- Experiment Definition (SP ablation determines FT checkpoint at runtime) ------------------
#   After SP experiment is completed, fill the path dynamically with find_best_elo_ckpt
EXPERIMENTS = [
    # -- Ablation 1a: Self-Play PD Only --------------------
    {
        "name":      "SP_PD_Only",
        "ablation":  1,
        "desc":      "Self-Play 1000ep - ratio_mode=pd_only",
        "type":      "sp",
        "sp_model":  "Ablation_SP_PD_Only",
        "sp_run":    "ablation_pd_only",
        "script":    SP_SCRIPT,
        "args": [
            "--episodes",   "1000",
            "--ratio_mode", "pd_only",
            "--model_name", "Ablation_SP_PD_Only",
            "--run_name",   "ablation_pd_only",
        ],
    },
    # -- Ablation 1b: Self-Play Fixed 0.5 ------------------
    {
        "name":      "SP_Fixed_50",
        "ablation":  1,
        "desc":      "Self-Play 1000ep - ratio_mode=fixed 0.5",
        "type":      "sp",
        "sp_model":  "Ablation_SP_Fixed_50",
        "sp_run":    "ablation_fixed_50",
        "script":    SP_SCRIPT,
        "args": [
            "--episodes",    "1000",
            "--ratio_mode",  "fixed",
            "--fixed_ratio", "0.5",
            "--model_name",  "Ablation_SP_Fixed_50",
            "--run_name",    "ablation_fixed_50",
        ],
    },
    # -- Ablation 1c: SAM FT from SP_PD_Only (checkpoint determined at runtime) --
    {
        "name":          "FT_from_PD_Only",
        "ablation":      1,
        "desc":          "SAM FT 500ep - PD-only SP BestELO transfer (actor_only)",
        "type":          "ft",
        "depends_sp":    "SP_PD_Only",   # Determine checkpoint after this SP completion
        "script":        FT_SCRIPT,
        "fixed_args": [
            "--episodes",      "500",
            "--transfer_mode", "actor_only",
            "--model_name",    "Ablation_FT_from_PD_Only",
            "--run_name",      "ablation_ft_pd_only",
        ],
        # "--init_from" is added at runtime
    },
    # -- Ablation 1d: SAM FT from SP_Fixed_50 (checkpoint determined at runtime) -
    {
        "name":          "FT_from_Fixed_50",
        "ablation":      1,
        "desc":          "SAM FT 500ep - Fixed-0.5 SP BestELO transfer (actor_only)",
        "type":          "ft",
        "depends_sp":    "SP_Fixed_50",
        "script":        FT_SCRIPT,
        "fixed_args": [
            "--episodes",      "500",
            "--transfer_mode", "actor_only",
            "--model_name",    "Ablation_FT_from_Fixed_50",
            "--run_name",      "ablation_ft_fixed_50",
        ],
    },
    # -- Ablation 2: Full Transfer --------------------------
    {
        "name":     "FT_Full",
        "ablation": 2,
        "desc":     "SAM FT 500ep - Actor+Critic transfer (Existing ep900, ELO 1314)",
        "type":     "ft_fixed",
        "script":   FT_SCRIPT,
        "args": [
            "--episodes",      "500",
            "--init_from",     EXISTING_SP_CKPT,
            "--transfer_mode", "full",
            "--model_name",    "Ablation_FT_Full",
            "--run_name",      "ablation_ft_full",
        ],
    },
    # -- Ablation 3: Actor Freeze Warmup -------------------
    {
        "name":     "FT_Freeze100",
        "ablation": 3,
        "desc":     "SAM FT 500ep - Actor 100ep freeze (Existing ep900, ELO 1314)",
        "type":     "ft_fixed",
        "script":   FT_SCRIPT,
        "args": [
            "--episodes",              "500",
            "--init_from",             EXISTING_SP_CKPT,
            "--transfer_mode",         "actor_only",
            "--actor_freeze_episodes", "100",
            "--model_name",            "Ablation_FT_Freeze100",
            "--run_name",              "ablation_ft_freeze100",
        ],
    },
]

# SP Name -> Experiment dict quick lookup
SP_MAP = {e["name"]: e for e in EXPERIMENTS if e.get("type") == "sp"}

# -- Utils ----------------------------------------------------------------------
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def fmt_elapsed(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

def run_cmd(name: str, cmd: list, log_path: Path) -> tuple[bool, float]:
    """Run subprocess. Returns (success, elapsed seconds)."""
    log(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(PROJECT_ROOT),
            )
            for line in proc.stdout:
                lf.write(line)
                if any(kw in line for kw in ["[CONTROL]", "[Freeze]", "[BestELO]",
                                              "[Init]", "Episode", "SUCCESS", "FAIL",
                                              "Error", "Traceback", "score"]):
                    print(f"    {line.rstrip()}", flush=True)
            proc.wait()
        elapsed = time.time() - t0
        ok = (proc.returncode == 0)
        return ok, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  EXCEPTION: {e}")
        return False, elapsed


# -- Main ----------------------------------------------------------------------
def main():
    if not Path(EXISTING_SP_CKPT).exists():
        print(f"ERROR: Existing SP checkpoint missing: {EXISTING_SP_CKPT}")
        sys.exit(1)

    overall_start = time.time()
    log("=" * 65)
    log("Ablation Study Weekend Runner starting")
    log(f"Total number of experiments: {len(EXPERIMENTS)}")
    log(f"Existing SP ckpt: {EXISTING_SP_CKPT}")
    log(f"Log directory: {LOGS_DIR}")
    log("=" * 65)

    results   = {}   # name -> {status, elapsed_s, ckpt_used, log}
    sp_ckpts  = {}   # SP name -> selected checkpoint Path

    for i, exp in enumerate(EXPERIMENTS, 1):
        name = exp["name"]
        log(f"\n[{i}/{len(EXPERIMENTS)}] -- {name} --------------")
        log(f"  Ablation {exp['ablation']}: {exp['desc']}")
        log_path = LOGS_DIR / f"{name}.log"

        # -- SP Experiment -----------------------------------------
        if exp["type"] == "sp":
            cmd = [PYTHON, exp["script"]] + exp["args"]
            ok, elapsed = run_cmd(name, cmd, log_path)
            status = "success" if ok else f"failed(rc)"
            log(f"  DONE: {status} ({fmt_elapsed(elapsed)})")

            # Search for best ELO checkpoint after completion
            if ok:
                model_dir = MODELS_DIR / exp["sp_model"]
                try:
                    best_ckpt = find_best_elo_ckpt(model_dir, exp["sp_run"])
                    sp_ckpts[name] = best_ckpt
                    log(f"  Best ELO ckpt: {best_ckpt}")
                except FileNotFoundError as e:
                    log(f"  WARNING: {e}")
                    sp_ckpts[name] = None
            else:
                sp_ckpts[name] = None

            results[name] = {"status": status, "elapsed_s": elapsed,
                             "ckpt_used": str(sp_ckpts.get(name, "")),
                             "log": str(log_path)}

        # -- FT (Dependent on SP ablation, checkpoint determined at runtime) -----
        elif exp["type"] == "ft":
            depends = exp["depends_sp"]
            ckpt = sp_ckpts.get(depends)

            if ckpt is None:
                msg = f"SKIP - {depends} incomplete or checkpoint missing"
                log(f"  {msg}")
                results[name] = {"status": "skipped", "reason": msg}
                continue

            cmd = ([PYTHON, exp["script"]]
                   + exp["fixed_args"]
                   + ["--init_from", str(ckpt)])
            ok, elapsed = run_cmd(name, cmd, log_path)
            status = "success" if ok else "failed(rc)"
            log(f"  DONE: {status} ({fmt_elapsed(elapsed)}) | ckpt={ckpt.name}")
            results[name] = {"status": status, "elapsed_s": elapsed,
                             "ckpt_used": str(ckpt), "log": str(log_path)}

        # -- FT (Fixed checkpoint, Ablation 2/3) -------------
        elif exp["type"] == "ft_fixed":
            cmd = [PYTHON, exp["script"]] + exp["args"]
            ok, elapsed = run_cmd(name, cmd, log_path)
            status = "success" if ok else "failed(rc)"
            log(f"  DONE: {status} ({fmt_elapsed(elapsed)})")
            results[name] = {"status": status, "elapsed_s": elapsed,
                             "ckpt_used": EXISTING_SP_CKPT, "log": str(log_path)}

    # -- Final Summary --------------------------------------------
    total = time.time() - overall_start
    log("\n" + "=" * 65)
    log("Final Summary")
    log("=" * 65)
    log(f"{'Name':<25} {'Status':<22} {'Elapsed Time':<12} {'Checkpoint'}")

    summary_rows = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        r = results.get(name, {"status": "not_run"})
        elapsed  = r.get("elapsed_s", 0)
        ckpt_str = Path(r.get("ckpt_used", "")).name if r.get("ckpt_used") else "-"
        icon = {"success": "OK  ", "skipped": "SKIP"}.get(
            r["status"] if r["status"] in ("success","skipped") else "fail", "FAIL")
        log(f"  [{icon}] {name:<22} {r['status']:<22} {fmt_elapsed(elapsed):<12} {ckpt_str}")
        summary_rows.append({"name": name, **r})

    log(f"\nTotal elapsed time: {fmt_elapsed(total)}")

    summary_path = LOGS_DIR / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"total_elapsed_s": total, "experiments": summary_rows},
                  f, indent=2, ensure_ascii=False)
    log(f"Summary saved: {summary_path}")

    failed = [r for r in summary_rows
              if r.get("status", "") not in ("success", "skipped")]
    if failed:
        log(f"Failed experiments: {[r['name'] for r in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
