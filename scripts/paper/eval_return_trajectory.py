#!/usr/bin/env python3
"""
Return 미션 궤적 평가 스크립트 (논문용)

사용법:
  # FT(Return) 기본 시나리오 10 에피소드
  python scripts/eval_return_trajectory.py --model ft_return --n 10 --seed 42

  # Scratch 기본 시나리오
  python scripts/eval_return_trajectory.py --model scratch --n 10 --seed 42

출력:
  figures/paper_figures/traj_data/{model}_seed{seed}/
    episode_000.csv  (step별 위치/상태)
    summary.csv      (에피소드 요약)

CSV 컬럼 (step):
  step, time_s, lat, lon, alt_m, heading_deg, speed_fps,
  wp_idx, goal_lat, goal_lon, goal_alt,
  dist_to_goal_m, reward, success, crashed

summary.csv:
  episode, seed, success, crashed, total_reward, steps, waypoints_reached
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Import env from training script (dynamic load) ────────────────────────
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ret_mod = _load_module("ret_scratch", ROOT / "scripts" / "20260113_SAC_Return.py")

NavMissionConfig          = _ret_mod.NavMissionConfig
JSBSimF16NavigationEnv    = _ret_mod.JSBSimF16NavigationEnv

from sac_agent.models import Actor  # noqa: E402

# ── Model paths ────────────────────────────────────────────────────────────
MODELS = {
    "ft_return": ROOT / "models" / "20260210_SAC_Return_FineTune_20260212_1249" / "epi_00499.pth",
    "scratch":   ROOT / "models" / "20260113_SAC_Return_20260119_1057"          / "epi_00499.pth",
}

STATE_SIZE  = 21
ACTION_SIZE = 4

# ── Helpers ────────────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def load_actor(ckpt_path, device):
    actor = Actor(STATE_SIZE, ACTION_SIZE).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd    = ckpt.get("actor", ckpt)
    actor.load_state_dict(sd)
    actor.eval()
    return actor

# ── Episode runner ─────────────────────────────────────────────────────────
def run_episode(env, actor, device, max_steps=50000):
    """
    Stochastic evaluation episode (rsample from Normal distribution).
    Returns list of step dicts and episode summary dict.
    """
    state = env.reset()
    done  = False
    step  = 0
    records = []

    while not done and step < max_steps:
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=device)
            mu, std = actor(t)
            # Stochastic: rsample from Normal, then tanh squash
            dist   = torch.distributions.Normal(mu, std)
            sample = dist.rsample()
            action = torch.tanh(sample).detach().cpu().numpy().flatten().astype(np.float32)

        next_state, reward, done, info = env.step(action)

        # Position
        lat, lon, alt = env.get_position()
        glat, glon, galt = env.get_waypoint()

        dist_goal = haversine_m(lat, lon, glat, glon)
        heading   = float(env.fdm.get_property_value("attitude/psi-deg"))
        speed_fps = float(env.fdm.get_property_value("velocities/u-fps"))
        time_s    = float(info.get("sim_time_s",
                          env.fdm.get_property_value("simulation/sim-time-sec")))

        success = int(bool(info.get("success", False)))
        # crashed: episode ended without mission success
        crashed = int(done and not bool(info.get("success", False))) if done else 0

        rec = {
            "step":           step,
            "time_s":         round(time_s,    3),
            "lat":            round(lat,        6),
            "lon":            round(lon,        6),
            "alt_m":          round(alt,        2),
            "heading_deg":    round(heading,    2),
            "speed_fps":      round(speed_fps,  2),
            "wp_idx":         int(info.get("wp_idx", 0)),
            "goal_lat":       round(glat,       6),
            "goal_lon":       round(glon,       6),
            "goal_alt":       round(galt,       2),
            "dist_to_goal_m": round(dist_goal,  1),
            "reward":         round(float(reward), 4),
            "success":        success,
            "crashed":        crashed,
        }
        records.append(rec)
        state = next_state
        step += 1

    # Episode summary
    last = records[-1] if records else {}
    waypoints_reached = int(info.get("wp_idx", 0))  # number of WPs passed
    meta = {
        "steps":             step,
        "success":           last.get("success",  0),
        "crashed":           last.get("crashed",  0),
        "total_reward":      round(sum(r["reward"] for r in records), 3),
        "waypoints_reached": waypoints_reached,
    }
    return records, meta

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ft_return",
                        choices=list(MODELS.keys()),
                        help="ft_return | scratch")
    parser.add_argument("--n",    type=int, default=10,
                        help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default=None,
                        help="Custom checkpoint path (overrides --model)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = Path(args.ckpt) if args.ckpt else MODELS[args.model]
    if not ckpt.exists():
        parent = ckpt.parent
        pths   = sorted(parent.glob("epi_*.pth"))
        if pths:
            ckpt = pths[-1]
            print(f"[INFO] Using last checkpoint: {ckpt.name}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {parent}")

    print(f"[Model]    {args.model}  ({ckpt})")
    print(f"[Seed]     {args.seed}  /  Episodes: {args.n}")
    print(f"[Eval]     stochastic (rsample from Normal)")

    actor   = load_actor(ckpt, device)
    mission = NavMissionConfig()

    out_dir = ROOT / "figures" / "paper_figures" / "traj_data" / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []

    for ep in range(args.n):
        ep_seed = args.seed + ep   # 에피소드마다 다른 seed -> 다양한 초기 조건

        env = JSBSimF16NavigationEnv(mission=mission, seed=ep_seed)

        records, meta = run_episode(env, actor, device)
        meta["episode"] = ep
        meta["seed"]    = ep_seed
        all_meta.append(meta)

        # Save step CSV
        step_path = out_dir / f"episode_{ep:03d}.csv"
        if records:
            with open(step_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        status = "SUCCESS" if meta["success"] else "FAILED"
        print(f"  ep {ep:3d} | {status:7s} | steps={meta['steps']:4d} | "
              f"R={meta['total_reward']:8.1f} | wp_reached={meta['waypoints_reached']}")

    # Save summary CSV
    summary_path = out_dir / "summary.csv"
    summary_fields = ["episode", "seed", "success", "crashed", "total_reward",
                      "steps", "waypoints_reached"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(all_meta)

    n_succ  = sum(m["success"] for m in all_meta)
    n_crash = sum(m["crashed"] for m in all_meta)
    print(f"\n[Result] Success={n_succ}/{args.n}  Crash/Timeout={n_crash}")
    print(f"[Saved]  {out_dir}")


if __name__ == "__main__":
    main()
