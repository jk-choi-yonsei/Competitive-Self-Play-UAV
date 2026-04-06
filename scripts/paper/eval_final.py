#!/usr/bin/env python3
"""
Mission 1 (Return) & Mission 2 (SAM) Final Evaluation Script (for Paper Tables)

- 50 episode deterministic evaluation for all models across both missions
- Deterministic: tanh(mu), no sampling
- Results: paper/results/eval_return_final.csv, paper/results/eval_sam_final.csv

Usage:
  cd "Self-Play UAV by RL"
  python scripts/paper/eval_final.py              # Execute all
  python scripts/paper/eval_final.py --mission return  # Return mission only
  python scripts/paper/eval_final.py --mission sam     # SAM mission only
  python scripts/paper/eval_final.py --n 50 --seed 100 # Custom settings
"""

import argparse
import csv
import math
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Dynamic module loader ─────────────────────────────────────────────────────
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Import environments ───────────────────────────────────────────────────────
_ret_mod = _load_module("ret_mod", ROOT / "scripts" / "20260113_SAC_Return.py")
_sam_mod = _load_module("sam_mod", ROOT / "scripts" / "20260219_SAC_SAM_FineTune.py")

NavMissionConfig          = _ret_mod.NavMissionConfig
JSBSimF16NavigationEnv    = _ret_mod.JSBSimF16NavigationEnv
SAMEvasionConfig          = _sam_mod.SAMEvasionConfig
JSBSimF16SAMEvasionEnv    = _sam_mod.JSBSimF16SAMEvasionEnv

# ── Actor classes ─────────────────────────────────────────────────────────────
from sac_agent.models  import Actor    as SACActorCls
from ppo_agent.models  import PPOActor as PPOActorCls

# ── Model registry ────────────────────────────────────────────────────────────
# fmt: (algo, model_dir, label)
RETURN_MODELS = [
    ("sac", ROOT / "models" / "20260113_SAC_Return_20260119_1057",                  "SAC_Scratch"),
    ("sac", ROOT / "models" / "20260210_SAC_Return_FineTune_20260212_1249",          "SAC_FT_ActorOnly"),
    ("ppo", ROOT / "models" / "20260226_PPO_Return_Scratch_20260225_2153",           "PPO_Scratch"),
    ("ppo", ROOT / "models" / "20260226_PPO_Return_FineTune_20260316_0927",          "PPO_FT"),
]

SAM_MODELS = [
    ("sac", ROOT / "models" / "20260219_SAC_SAM_Scratch_20260226_1638",                      "SAC_Scratch"),
    ("sac", ROOT / "models" / "20260219_SAC_SAM_FineTune_20260306_1730",                     "SAC_FT_Self"),
    ("sac", ROOT / "models" / "20260306_SAC_SAM_FineTune_From_Return_20260310_1052",         "SAC_FT_Return"),
    ("ppo", ROOT / "models" / "20260310_PPO_SAM_FineTune_20260311_0901",                     "PPO_FT_Self"),
    ("ppo", ROOT / "models" / "20260311_PPO_SAM_Scratch_20260312_0902",                      "PPO_Scratch"),
]

RETURN_STATE_SIZE = 21
SAM_STATE_SIZE    = 25
ACTION_SIZE       = 4

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_ckpt(model_dir: Path, episode: int = 499) -> Path:
    ckpt = model_dir / f"epi_{episode:05d}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found")
    return ckpt


def load_actor(algo: str, ckpt_path: Path, state_size: int, device):
    if algo == "sac":
        actor = SACActorCls(state_size, ACTION_SIZE).to(device)
    else:
        actor = PPOActorCls(state_size, ACTION_SIZE).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # checkpoint may store {"actor": state_dict} or be the state_dict itself
    sd = ckpt.get("actor", ckpt)
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ── Episode runners ───────────────────────────────────────────────────────────
def run_return_episode(env, actor, device):
    """Deterministic Return mission episode."""
    state = env.reset()
    done  = False
    step  = 0
    total_reward = 0.0

    while not done and step < 50000:
        with torch.no_grad():
            t  = torch.tensor(state, dtype=torch.float32, device=device)
            mu, _ = actor(t)
            action = torch.tanh(mu).detach().cpu().numpy().flatten().astype(np.float32)

        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        step += 1

    success  = int(bool(info.get("success", False)))
    crashed  = int(done and not success)
    wp       = int(info.get("wp_idx", 0))
    return {"success": success, "crashed": crashed,
            "waypoints_reached": wp, "steps": step,
            "total_reward": round(total_reward, 3)}


def run_sam_episode(env, actor, device):
    """Deterministic SAM evasion episode."""
    state = env.reset()
    done  = False
    step  = 0
    total_reward = 0.0

    while not done and step < 50000:
        with torch.no_grad():
            t  = torch.tensor(state, dtype=torch.float32, device=device)
            mu, _ = actor(t)
            action = torch.tanh(mu).detach().cpu().numpy().flatten().astype(np.float32)

        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        step += 1

    success = int(bool(info.get("success",       False)))
    killed  = int(bool(info.get("killed_by_sam", False)))
    crashed = int(bool(info.get("crashed",       False)))
    return {"success": success, "killed": killed, "crashed": crashed,
            "steps": step, "total_reward": round(total_reward, 3)}


# ── Evaluator ─────────────────────────────────────────────────────────────────
def evaluate_mission(mission_name: str, model_list, n_eps: int, base_seed: int, device):
    """Run n_eps deterministic episodes for every model in model_list."""
    is_sam    = (mission_name == "sam")
    state_sz  = SAM_STATE_SIZE if is_sam else RETURN_STATE_SIZE
    rows      = []

    for algo, model_dir, label in model_list:
        try:
            ckpt = find_ckpt(model_dir, episode=499)
        except FileNotFoundError as e:
            print(f"  [SKIP] {label}: {e}")
            continue

        print(f"\n  [{label}]  algo={algo}  ckpt={ckpt.name}")
        actor = load_actor(algo, ckpt, state_sz, device)

        # Create env ONCE per model (reuse with reset per episode — same as training)
        if is_sam:
            env = JSBSimF16SAMEvasionEnv(mission=SAMEvasionConfig(), seed=base_seed)
        else:
            env = JSBSimF16NavigationEnv(mission=NavMissionConfig(), seed=base_seed)

        ep_results = []
        for ep in range(n_eps):
            if is_sam:
                meta = run_sam_episode(env, actor, device)
            else:
                meta = run_return_episode(env, actor, device)
            ep_results.append(meta)

            tag = ("SUCCESS" if meta["success"]
                   else ("KILLED" if meta.get("killed") else "FAIL"))
            print(f"    ep {ep:3d} | {tag:7s} | steps={meta['steps']:5d} | R={meta['total_reward']:8.1f}")

        # Aggregate
        n = len(ep_results)
        success_rate = sum(r["success"] for r in ep_results) / n
        avg_reward   = sum(r["total_reward"] for r in ep_results) / n
        avg_steps    = sum(r["steps"] for r in ep_results) / n

        row = {
            "label":        label,
            "algo":         algo.upper(),
            "n_episodes":   n,
            "success_rate": round(success_rate * 100, 1),
            "avg_reward":   round(avg_reward, 2),
            "avg_steps":    round(avg_steps, 1),
        }

        if is_sam:
            row["kill_rate"]   = round(sum(r["killed"]  for r in ep_results) / n * 100, 1)
            row["crash_rate"]  = round(sum(r["crashed"] for r in ep_results) / n * 100, 1)
        else:
            row["crash_rate"]  = round(sum(r["crashed"] for r in ep_results) / n * 100, 1)
            row["avg_wp"]      = round(sum(r["waypoints_reached"] for r in ep_results) / n, 2)

        print(f"  → SR={row['success_rate']}%  AvgR={row['avg_reward']}")
        rows.append(row)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", default="all", choices=["all", "return", "sam"])
    parser.add_argument("--n",    type=int, default=50, help="Episodes per model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir  = ROOT / "paper" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  Episodes/model: {args.n}  |  Base seed: {args.seed}\n")

    # ── Mission 1: Return ──────────────────────────────────────────────────
    if args.mission in ("all", "return"):
        print("=" * 60)
        print("MISSION 1: RETURN")
        print("=" * 60)
        rows = evaluate_mission("return", RETURN_MODELS, args.n, args.seed, device)

        out_path = out_dir / "eval_return_final.csv"
        fields = ["label", "algo", "n_episodes", "success_rate",
                  "avg_reward", "avg_steps", "crash_rate", "avg_wp"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] {out_path}")

    # ── Mission 2: SAM ────────────────────────────────────────────────────
    if args.mission in ("all", "sam"):
        print("\n" + "=" * 60)
        print("MISSION 2: SAM EVASION")
        print("=" * 60)
        rows = evaluate_mission("sam", SAM_MODELS, args.n, args.seed, device)

        out_path = out_dir / "eval_sam_final.csv"
        fields = ["label", "algo", "n_episodes", "success_rate",
                  "avg_reward", "avg_steps", "kill_rate", "crash_rate"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
