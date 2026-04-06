#!/usr/bin/env python3
"""
Collect trajectories with the same seeding as paper result CSVs (single env, repeated reset())

- Standard SAM (seed=42, 50ep): ft_self / scratch / ft_return
  -> Same conditions as eval_sam_final.csv (deterministic)
- Popup SAM (seed=42, 50ep): ft_self / scratch
  -> Same conditions as robustness_popup_sam_*.csv (deterministic)
- Return mission (seed=42, 50ep): ft_return / scratch
  -> Same conditions as eval_return_final.csv (deterministic)

Output Directory (under traj_data):
  ft_self_standard_eval_seed42/
  scratch_standard_eval_seed42/
  ft_return_standard_eval_seed42/
  ft_self_popup_eval_seed42/
  scratch_popup_eval_seed42/
  ft_return_return_eval_seed42/
  scratch_return_eval_seed42/
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ft_mod  = _load_module("sam_ft",  ROOT / "scripts" / "20260219_SAC_SAM_FineTune.py")
_rob_mod = _load_module("sam_rob", ROOT / "scripts" / "20260304_SAM_Robustness_Test.py")
_ret_mod = _load_module("ret_sc",  ROOT / "scripts" / "20260113_SAC_Return.py")

SAMEvasionConfig       = _ft_mod.SAMEvasionConfig
JSBSimF16SAMEvasionEnv = _ft_mod.JSBSimF16SAMEvasionEnv
PopupSAMEnv            = _rob_mod.PopupSAMEnv
NavMissionConfig       = _ret_mod.NavMissionConfig
JSBSimF16NavigationEnv = _ret_mod.JSBSimF16NavigationEnv

from sac_agent.models import Actor

SAM_MODELS = {
    "ft_self":   ROOT / "models" / "20260219_SAC_SAM_FineTune_20260306_1730"  / "epi_00499.pth",
    "scratch":   ROOT / "models" / "20260219_SAC_SAM_Scratch_20260226_1638"   / "epi_00499.pth",
    "ft_return": ROOT / "models" / "20260306_SAC_SAM_FineTune_From_Return_20260310_1052" / "epi_00499.pth",
}
RET_MODELS = {
    "ft_return": ROOT / "models" / "20260210_SAC_Return_FineTune_20260212_1249" / "epi_00499.pth",
    "scratch":   ROOT / "models" / "20260113_SAC_Return_20260119_1057"          / "epi_00499.pth",
}
SAM_STATE_SIZE = 25
RET_STATE_SIZE = 21
ACTION_SIZE    = 4
TDATA = ROOT / "figures" / "paper_figures" / "traj_data"


def load_actor(ckpt_path, device, state_size=SAM_STATE_SIZE):
    actor = Actor(state_size, ACTION_SIZE).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd    = ckpt.get("actor", ckpt)
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def run_episode(env, actor, device, max_steps=50000):
    """Run single episode. env.reset() is called here."""
    state = env.reset()
    done  = False
    step  = 0
    records = []
    is_popup = isinstance(env, PopupSAMEnv)

    while not done and step < max_steps:
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=device)
            mu, _std = actor(t)
            action = torch.tanh(mu).detach().cpu().numpy().flatten().astype(np.float32)

        next_state, reward, done, info = env.step(action)

        lat, lon, alt   = env.get_position()
        slat, slon, salt = env.get_sam_position()
        glat, glon, galt = env.get_waypoint()

        dist_sam  = haversine_m(lat, lon, slat, slon)
        dist_goal = haversine_m(lat, lon, glat, glon)
        heading   = float(env.fdm.get_property_value("attitude/psi-deg"))
        speed_fps = float(env.fdm.get_property_value("velocities/u-fps"))
        time_s    = float(env.fdm.get_property_value("simulation/sim-time-sec"))

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
            "success":        int(bool(info.get("success",       False))),
            "killed":         int(bool(info.get("killed_by_sam", False))),
            "crashed":        int(bool(info.get("crashed",       False))),
        }
        if is_popup:
            rec["popup_triggered"] = int(getattr(env, "_popup_triggered", False))
            rec["step_at_popup"]   = int(getattr(env, "_step_at_popup") or -1)

        records.append(rec)
        state = next_state
        step += 1

    last = records[-1] if records else {}
    meta = {
        "steps":        step,
        "success":      last.get("success", 0),
        "killed":       last.get("killed",  0),
        "crashed":      last.get("crashed", 0),
        "total_reward": round(sum(r["reward"] for r in records), 3),
    }
    if is_popup:
        meta["popup_triggered"] = last.get("popup_triggered", 0)
        meta["step_at_popup"]   = last.get("step_at_popup",  -1)
    return records, meta


def run_return_episode(env, actor, device, max_steps=50000):
    """Single episode for Return mission (deterministic, same as eval_final.py)."""
    state = env.reset()
    done  = False
    step  = 0
    records = []

    while not done and step < max_steps:
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=device)
            mu, _std = actor(t)
            action = torch.tanh(mu).detach().cpu().numpy().flatten().astype(np.float32)

        next_state, reward, done, info = env.step(action)

        lat, lon, alt  = env.get_position()
        glat, glon, galt = env.get_waypoint()
        dist_goal = haversine_m(lat, lon, glat, glon)
        heading   = float(env.fdm.get_property_value("attitude/psi-deg"))
        speed_fps = float(env.fdm.get_property_value("velocities/u-fps"))
        time_s    = float(info.get("sim_time_s",
                          env.fdm.get_property_value("simulation/sim-time-sec")))
        success = int(bool(info.get("success", False)))
        crashed = int(done and not bool(info.get("success", False))) if done else 0

        records.append({
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
        })
        state = next_state
        step += 1

    last = records[-1] if records else {}
    meta = {
        "steps":             step,
        "success":           last.get("success",  0),
        "crashed":           last.get("crashed",  0),
        "total_reward":      round(sum(r["reward"] for r in records), 3),
        "waypoints_reached": int(info.get("wp_idx", 0)),
    }
    return records, meta


def collect_return(model_key, seed, n_eps, device):
    """Collect Return mission trajectories (same conditions as eval_return_final.csv)."""
    mission = NavMissionConfig()
    env     = JSBSimF16NavigationEnv(mission=mission, seed=seed)
    actor   = load_actor(RET_MODELS[model_key], device, state_size=RET_STATE_SIZE)

    out_dir_name = f"{model_key}_return_eval_seed{seed}"
    out_dir = TDATA / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for ep in range(n_eps):
        records, meta = run_return_episode(env, actor, device)
        meta["episode"] = ep
        meta["seed"]    = seed
        all_meta.append(meta)

        step_path = out_dir / f"episode_{ep:03d}.csv"
        if records:
            with open(step_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        status = "SUCCESS" if meta["success"] else "CRASHED"
        print(f"  ep {ep:3d} | {status:7s} | steps={meta['steps']:4d} | "
              f"R={meta['total_reward']:8.1f} | wp={meta['waypoints_reached']}")

    meta_path = out_dir / "summary.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_meta[0].keys()))
        writer.writeheader()
        writer.writerows(all_meta)

    n_s = sum(m["success"] for m in all_meta)
    n_c = sum(m["crashed"] for m in all_meta)
    print(f"[Result] {out_dir_name}: Success={n_s}/{n_eps}  Crash={n_c}")
    print(f"[Saved]  {out_dir}")
    return all_meta


def collect(model_key, mode, seed, n_eps, device, out_dir_name=None):
    """Collect n_eps episodes with a single env. Same method as robustness/eval_final.py."""
    mission = SAMEvasionConfig()
    if mode == "popup":
        env = PopupSAMEnv(mission=mission, popup_radius_m=12000.0, seed=seed)
    else:
        env = JSBSimF16SAMEvasionEnv(mission=mission, seed=seed)

    actor = load_actor(SAM_MODELS[model_key], device)

    if out_dir_name is None:
        out_dir_name = f"{model_key}_{mode}_eval_seed{seed}"
    out_dir = TDATA / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for ep in range(n_eps):
        records, meta = run_episode(env, actor, device)
        meta["episode"] = ep
        meta["seed"]    = seed   # No per-episode seed -> record common seed
        all_meta.append(meta)

        step_path = out_dir / f"episode_{ep:03d}.csv"
        if records:
            with open(step_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        status = "SUCCESS" if meta["success"] else ("KILLED" if meta["killed"] else "CRASHED")
        popup_info = ""
        if mode == "popup":
            trig = meta.get("popup_triggered", 0)
            sp   = meta.get("step_at_popup", -1)
            popup_info = f"  popup={'Y' if trig else 'N'}@{sp}"
        print(f"  ep {ep:3d} | {status:7s} | steps={meta['steps']:4d} | R={meta['total_reward']:8.1f}{popup_info}")

    meta_path = out_dir / "summary.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_meta[0].keys()))
        writer.writeheader()
        writer.writerows(all_meta)

    n_s = sum(m["success"] for m in all_meta)
    n_k = sum(m["killed"]  for m in all_meta)
    n_c = sum(m["crashed"] for m in all_meta)
    print(f"[Result] {out_dir_name}: Success={n_s}/{n_eps}  Kill={n_k}  Crash={n_c}")
    print(f"[Saved]  {out_dir}")
    return all_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["standard", "popup", "return", "all"], default="all")
    ap.add_argument("--n",    type=int, default=50)
    ap.add_argument("--standard_seed", type=int, default=42,
                    help="Standard SAM seed (based on eval_sam_final.csv: 42)")
    ap.add_argument("--popup_seed",    type=int, default=42,
                    help="Popup SAM seed (based on robustness CSV: 42)")
    ap.add_argument("--return_seed",   type=int, default=42,
                    help="Return mission seed (based on eval_return_final.csv: 42)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_standard = args.scenario in ("standard", "all")
    run_popup    = args.scenario in ("popup",    "all")
    run_return   = args.scenario in ("return",   "all")

    if run_standard:
        print("\n" + "="*60)
        print(f"Standard SAM — seed={args.standard_seed}, {args.n} eps")
        print("="*60)
        for model_key in ("ft_self", "scratch", "ft_return"):
            print(f"\n[{model_key}]")
            collect(model_key, "standard", args.standard_seed, args.n, device)

    if run_popup:
        print("\n" + "="*60)
        print(f"Popup SAM — seed={args.popup_seed}, {args.n} eps")
        print("="*60)
        for model_key in ("ft_self", "scratch"):
            print(f"\n[{model_key}]")
            collect(model_key, "popup", args.popup_seed, args.n, device)

    if run_return:
        print("\n" + "="*60)
        print(f"Return mission — seed={args.return_seed}, {args.n} eps")
        print("="*60)
        for model_key in ("ft_return", "scratch"):
            print(f"\n[{model_key}]")
            collect_return(model_key, args.return_seed, args.n, device)

    print("\nDONE.")


if __name__ == "__main__":
    main()
