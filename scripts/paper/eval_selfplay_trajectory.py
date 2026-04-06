#!/usr/bin/env python3
"""
SAC Self-Play 궤적 평가 스크립트 (논문용)

사용법:
  python scripts/eval_selfplay_trajectory.py --n 10 --seed 42

출력:
  figures/paper_figures/traj_data/selfplay_seed{seed}/
    episode_000.csv  (step별 위치/상태)
    summary.csv      (에피소드 요약)

CSV 컬럼 (step):
  step, time_s,
  agent_lat, agent_lon, agent_alt_m, agent_heading_deg, agent_speed_fps,
  tgt_lat, tgt_lon, tgt_alt_m, tgt_heading_deg,
  range_m, ata_deg, aot_deg,
  reward, success, killed
"""

import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributions as D

# ── Project root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Dynamic import from training script ───────────────────────────────────
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_sp_mod = _load_module("sac_selfplay", ROOT / "scripts" / "20260109_SAC_Self_Play.py")

JSBSimF16ChaseEnv = _sp_mod.JSBSimF16ChaseEnv
PdOpponent        = _sp_mod.PdOpponent
FrameStacker      = _sp_mod.FrameStacker
build_frame       = _sp_mod.build_frame

from sac_agent.models import Actor  # noqa: E402

# ── Checkpoint path ────────────────────────────────────────────────────────
CKPT_PATH = ROOT / "models" / "20260109_SAC_Self_Play" / "selfplay_pool_01999.pth"

# ── Helpers ────────────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def load_actor(ckpt_path, state_size, device):
    actor = Actor(state_size, 4).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd    = ckpt.get("actor", ckpt)
    actor.load_state_dict(sd)
    actor.eval()
    return actor


# ── Episode runner ─────────────────────────────────────────────────────────
def run_episode(env, actor, opponent, device, max_steps=2000):
    """
    Stochastic evaluation episode (rsample from Normal distribution).
    Returns (records, meta).
    """
    state = env.reset()          # shape (1, 5, 20)
    opponent.reset(env)
    done  = False
    step  = 0
    records = []

    while not done and step < max_steps:
        with torch.no_grad():
            t   = torch.tensor(state, dtype=torch.float32, device=device)
            mu, std = actor(t)
            # Stochastic: rsample from Normal, then tanh-squash
            dist   = D.Normal(mu, std)
            sample = dist.rsample()
            action = torch.tanh(sample).detach().cpu().numpy().flatten().astype(np.float32)

        # PastSelf opponent needs to supply target_action; PD uses env-internal controller
        if opponent.requires_action:
            env.target_action = opponent.act(env)
        next_state, reward, done, info = env.step(action)

        # Agent position (main FDM)
        agent_lat = float(env.fdm.get_property_value("position/lat-gc-deg"))
        agent_lon = float(env.fdm.get_property_value("position/long-gc-deg"))
        agent_alt = float(env.fdm.get_property_value("position/h-sl-meters"))
        agent_hdg = float(env.fdm.get_property_value("attitude/psi-deg"))
        agent_spd = float(env.fdm.get_property_value("velocities/u-fps"))
        time_s    = float(env.fdm.get_property_value("simulation/sim-time-sec"))

        # Target position (target FDM)
        tgt_lat = float(env.fdm_target.get_property_value("position/lat-gc-deg"))
        tgt_lon = float(env.fdm_target.get_property_value("position/long-gc-deg"))
        tgt_alt = float(env.fdm_target.get_property_value("position/h-sl-meters"))
        tgt_hdg = float(env.fdm_target.get_property_value("attitude/psi-deg"))

        # Engagement geometry from info (populated by env)
        range_m = float(info.get("range_3d_m", haversine_m(agent_lat, agent_lon, tgt_lat, tgt_lon)))
        ata_deg = float(info.get("ata_deg", float("nan")))
        aot_deg = float(info.get("aot_deg", float("nan")))

        rec = {
            "step":             step,
            "time_s":           round(time_s,    3),
            "agent_lat":        round(agent_lat,  6),
            "agent_lon":        round(agent_lon,  6),
            "agent_alt_m":      round(agent_alt,  2),
            "agent_heading_deg": round(agent_hdg, 2),
            "agent_speed_fps":  round(agent_spd,  2),
            "tgt_lat":          round(tgt_lat,    6),
            "tgt_lon":          round(tgt_lon,    6),
            "tgt_alt_m":        round(tgt_alt,    2),
            "tgt_heading_deg":  round(tgt_hdg,    2),
            "range_m":          round(range_m,    1),
            "ata_deg":          round(ata_deg,    4) if not math.isnan(ata_deg) else float("nan"),
            "aot_deg":          round(aot_deg,    4) if not math.isnan(aot_deg) else float("nan"),
            "reward":           round(float(reward), 4),
            "success":          int(bool(info.get("success", False))),
            "killed":           int(bool(info.get("killed",  False))),
        }
        records.append(rec)
        state = next_state
        step += 1

    # Episode summary
    last = records[-1] if records else {}
    meta = {
        "steps":        step,
        "success":      last.get("success",  0),
        "killed":       last.get("killed",   0),
        "total_reward": round(sum(r["reward"] for r in records), 3),
    }
    return records, meta


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=10,
                        help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default=None,
                        help="Custom checkpoint path (overrides default)")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt     = Path(args.ckpt) if args.ckpt else CKPT_PATH
    if not ckpt.exists():
        parent = ckpt.parent
        pths   = sorted(parent.glob("selfplay_pool_*.pth"))
        if not pths:
            pths = sorted(parent.glob("*.pth"))
        if pths:
            ckpt = pths[-1]
            print(f"[INFO] Using last checkpoint: {ckpt.name}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {parent}")

    print(f"[Checkpoint] {ckpt}")
    print(f"[Seed]       {args.seed}  /  Episodes: {args.n}")

    # Build a temporary env to determine state_size
    _tmp_env   = JSBSimF16ChaseEnv(target_policy="pd")
    _tmp_state = _tmp_env.reset()          # (1, 5, 20)
    state_size = _tmp_state.shape[2]       # 20
    print(f"[State size] {state_size}")

    actor    = load_actor(ckpt, state_size, device)
    opponent = PdOpponent()

    out_dir = ROOT / "figures" / "paper_figures" / "traj_data" / f"selfplay_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []

    for ep in range(args.n):
        ep_seed = args.seed + ep

        env = JSBSimF16ChaseEnv(
            target_policy="pd",
            init_noise_config={
                "enabled": True,
                "seed": ep_seed,
                "target_sigma_north_m": 500.0,
                "target_sigma_east_m":  500.0,
                "target_sigma_alt_m":   100.0,
                "agent_sigma_u_fps":    50.0,
                "agent_heading_uniform_deg":  20.0,
                "target_sigma_u_fps":   0.0,
                "target_heading_uniform_deg": 0.0,
            },
        )

        records, meta = run_episode(env, actor, opponent, device)
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

        status = "SUCCESS" if meta["success"] else ("KILLED" if meta["killed"] else "TIMEOUT")
        print(f"  ep {ep:3d} | {status:7s} | steps={meta['steps']:4d} | R={meta['total_reward']:8.1f}")

    # Save summary CSV
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_meta[0].keys()))
        writer.writeheader()
        writer.writerows(all_meta)

    n_succ = sum(m["success"] for m in all_meta)
    n_kill = sum(m["killed"]  for m in all_meta)
    print(f"\n[Result] Success={n_succ}/{args.n}  Kill={n_kill}")
    print(f"[Saved]  {out_dir}")


if __name__ == "__main__":
    main()
