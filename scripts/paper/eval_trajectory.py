#!/usr/bin/env python3
"""
SAM Evasion 궤적 평가 스크립트 (논문용)

사용법:
  # FT(Self-Play) 기본 시나리오 10 에피소드
  python scripts/eval_trajectory.py --model ft_self --n 10 --seed 42

  # Scratch 기본 시나리오
  python scripts/eval_trajectory.py --model scratch --n 10 --seed 42

  # FT(Return) 기본 시나리오
  python scripts/eval_trajectory.py --model ft_return --n 10 --seed 42

  # Pop-up SAM 시나리오
  python scripts/eval_trajectory.py --model ft_self --mode popup --n 10 --seed 42

출력:
  figures/paper_figures/traj_data/{model}_{mode}_seed{seed}/
    episode_000.csv  (step별 위치/상태)
    episode_000_meta.csv  (에피소드 요약)
    ...

CSV 컬럼 (step):
  step, time_s, lat, lon, alt_m, heading_deg, speed_fps,
  sam_lat, sam_lon, sam_alt,
  goal_lat, goal_lon, goal_alt,
  dist_to_sam_m, dist_to_goal_m,
  in_radar, threat_factor,
  reward, success, killed, crashed
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch

# ── Project root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Import env from training scripts ─────────────────────────────────────
# SAMEvasionConfig, JSBSimF16SAMEvasionEnv 은 FineTune 스크립트에 정의
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ft_mod  = _load_module("sam_ft",  ROOT / "scripts" / "20260219_SAC_SAM_FineTune.py")
_rob_mod = _load_module("sam_rob", ROOT / "scripts" / "20260304_SAM_Robustness_Test.py")

SAMEvasionConfig       = _ft_mod.SAMEvasionConfig
JSBSimF16SAMEvasionEnv = _ft_mod.JSBSimF16SAMEvasionEnv
PopupSAMEnv            = _rob_mod.PopupSAMEnv

from sac_agent.models import Actor

# ── Model paths ───────────────────────────────────────────────────────────
MODELS = {
    "ft_self":   ROOT / "models" / "20260219_SAC_SAM_FineTune_20260306_1730"  / "epi_00499.pth",
    "scratch":   ROOT / "models" / "20260219_SAC_SAM_Scratch_20260226_1638"   / "epi_00499.pth",
    "ft_return": ROOT / "models" / "20260306_SAC_SAM_FineTune_From_Return_20260310_1052" / "epi_00499.pth",
}

STATE_SIZE  = 25   # SAM env: 11 kine + 10 geo + 4 SAM = 25
ACTION_SIZE = 4

# ── Helpers ───────────────────────────────────────────────────────────────
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

# ── Episode runner ────────────────────────────────────────────────────────
def run_episode(env, actor, device, max_steps=50000):
    """
    Deterministic evaluation episode.
    Returns list of step dicts.
    """
    state = env.reset()
    done  = False
    step  = 0
    records = []

    # Pop-up fields (may not exist on base env)
    is_popup = isinstance(env, PopupSAMEnv)

    while not done and step < max_steps:
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=device)
            mu, _std = actor(t)
            # Deterministic: tanh(mu) — no sampling
            action = torch.tanh(mu).detach().cpu().numpy().flatten().astype(np.float32)

        next_state, reward, done, info = env.step(action)

        # Position
        lat, lon, alt = env.get_position()
        slat, slon, salt = env.get_sam_position()
        glat, glon, galt = env.get_waypoint()

        dist_sam  = haversine_m(lat, lon, slat, slon)
        dist_goal = haversine_m(lat, lon, glat, glon)
        heading   = float(env.fdm.get_property_value("attitude/psi-deg"))
        speed_fps = float(env.fdm.get_property_value("velocities/u-fps"))
        time_s    = float(env.fdm.get_property_value("simulation/sim-time-sec"))

        # threat_factor (tf): smaller = closer/more dangerous; in_radar if tf < 4.0
        tf       = float(info.get("threat_factor", float("nan")))
        in_radar = int(tf < 4.0) if not math.isnan(tf) else 0

        rec = {
            "step":           step,
            "time_s":         round(time_s,    3),
            "lat":            round(lat,        6),
            "lon":            round(lon,        6),
            "alt_m":          round(alt,        2),
            "heading_deg":    round(heading,    2),
            "speed_fps":      round(speed_fps,  2),
            "sam_lat":        round(slat,       6),
            "sam_lon":        round(slon,       6),
            "sam_alt":        round(salt,       2),
            "goal_lat":       round(glat,       6),
            "goal_lon":       round(glon,       6),
            "goal_alt":       round(galt,       2),
            "dist_to_sam_m":  round(dist_sam,   1),
            "dist_to_goal_m": round(dist_goal,  1),
            "in_radar":       in_radar,
            "threat_factor":  round(tf,         4),
            "r_threat":       round(float(info.get("r_threat", 0.0)), 4),
            "reward":         round(float(reward), 4),
            "success":        int(bool(info.get("success",      False))),
            "killed":         int(bool(info.get("killed_by_sam",False))),
            "crashed":        int(bool(info.get("crashed",      False))),
        }
        if is_popup:
            rec["popup_triggered"] = int(getattr(env, "_popup_triggered", False))
            rec["step_at_popup"]   = int(getattr(env, "_step_at_popup") or -1)

        records.append(rec)
        state = next_state
        step += 1

    # Episode summary
    last = records[-1] if records else {}
    meta = {
        "steps":    step,
        "success":  last.get("success", 0),
        "killed":   last.get("killed",  0),
        "crashed":  last.get("crashed", 0),
        "total_reward": round(sum(r["reward"] for r in records), 3),
    }
    if is_popup:
        meta["popup_triggered"] = last.get("popup_triggered", 0)
        meta["step_at_popup"]   = last.get("step_at_popup",  -1)

    return records, meta

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="ft_self",
                        choices=list(MODELS.keys()),
                        help="ft_self | scratch | ft_return")
    parser.add_argument("--mode",   default="standard",
                        choices=["standard", "popup"],
                        help="standard | popup")
    parser.add_argument("--n",      type=int, default=10,
                        help="Number of episodes")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--popup_radius_m", type=float, default=12000.0)
    parser.add_argument("--ckpt",   default=None,
                        help="Custom checkpoint path (overrides --model)")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt     = Path(args.ckpt) if args.ckpt else MODELS[args.model]
    if not ckpt.exists():
        # Try last available epi checkpoint
        parent = ckpt.parent
        pths   = sorted(parent.glob("epi_*.pth"))
        if pths:
            ckpt = pths[-1]
            print(f"[INFO] Using last checkpoint: {ckpt.name}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {parent}")

    print(f"[Model]  {args.model}  ({ckpt})")
    print(f"[Mode]   {args.mode}")
    print(f"[Seed]   {args.seed}  /  Episodes: {args.n}")

    actor  = load_actor(ckpt, device)
    mission = SAMEvasionConfig()

    out_dir = ROOT / "figures" / "paper_figures" / "traj_data" / f"{args.model}_{args.mode}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []

    for ep in range(args.n):
        ep_seed = args.seed + ep   # 에피소드마다 다른 seed -> 다양한 초기 조건

        if args.mode == "popup":
            env = PopupSAMEnv(mission=mission, popup_radius_m=args.popup_radius_m,
                              seed=ep_seed)
        else:
            env = JSBSimF16SAMEvasionEnv(mission=mission, seed=ep_seed)

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

        status = "SUCCESS" if meta["success"] else ("KILLED" if meta["killed"] else "CRASHED")
        print(f"  ep {ep:3d} | {status:7s} | steps={meta['steps']:4d} | R={meta['total_reward']:8.1f}")

    # Save meta summary
    meta_path = out_dir / "summary.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_meta[0].keys()))
        writer.writeheader()
        writer.writerows(all_meta)

    n_succ  = sum(m["success"] for m in all_meta)
    n_kill  = sum(m["killed"]  for m in all_meta)
    n_crash = sum(m["crashed"] for m in all_meta)
    print(f"\n[Result] Success={n_succ}/{args.n}  Kill={n_kill}  Crash={n_crash}")
    print(f"[Saved]  {out_dir}")


if __name__ == "__main__":
    main()
