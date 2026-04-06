"""
robustness_test.py
==================
Self-Play UAV - SAM Evasion Robustness Test Script (zero-shot evaluation without training)

Scenarios:
  1) Threat Scaling   : Evaluate with SAM radius expanded to Base / 120% / 150%
  2) Initial Perturb  : Random perturbations for start altitude (2000-4000m) and heading (+-30 deg)
  3) Pop-up SAM       : Radar ON when within 12km -> Evaluate Break Turn response

Usage:
  python scripts/robustness_test.py \\
      --scratch_dir  models/20260219_SAC_SAM_Scratch_20260226_1638 \\
      --finetune_dir models/20260219_SAC_SAM_FineTune_20260227_2235 \\
      --ft_return_dir models/20260306_SAC_SAM_FineTune_From_Return_20260308_1115 \\
      --episodes 50 \\
      --scenario all          # all | threat | perturb | popup

Results:
  figures/robustness_test/{scenario}/{model_tag}/  -> PNG trajectory
  results/robustness_test_{scenario}_{model_tag}.csv
"""

import argparse
import csv
import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# -- Project Root Setup --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from sac_agent.models import Actor                              # noqa: E402
                                                               # noqa: E402
from envs.init_noise import meters_to_latlon_deg               # noqa: E402
from envs.target_controllers import bearing_deg, wrap180       # noqa: E402

try:
    from haversine import haversine as haversine               # type: ignore
except Exception:
    def haversine(p1, p2) -> float:
        import math
        lat1, lon1 = p1; lat2, lon2 = p2
        r = 6371.0088
        phi1 = math.radians(float(lat1)); phi2 = math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlmb = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
        return float(r * 2.0 * math.asin(math.sqrt(a)))

try:
    import jsbsim
except ImportError:
    raise ImportError("jsbsim package not found.")

KTS_PER_FPS = 1.0 / 1.6878098571011957
FT_PER_M    = 3.28084

# -- Results Directory --
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_BASE    = PROJECT_ROOT / "figures" / "robustness_test"
FIG_BASE.mkdir(parents=True, exist_ok=True)

# -- Existing Environment & Config (Copied from 20260219_SAC_SAM_Scratch.py) --

@dataclass
class SAMEvasionConfig:
    goal_dist_min_m: float = 30000.0
    goal_dist_max_m: float = 40000.0
    goal_los_deg: float    = 10.0
    goal_up_min_m: float   = -300.0
    goal_up_max_m: float   = 300.0
    sam_place_frac_min: float = 0.4
    sam_place_frac_max: float = 0.6
    sam_lateral_offset_m: float = 0.0
    r_horiz_m: float  = 10000.0
    r_vert_m: float   = 5000.0
    radar_floor_m: float = 0.0
    threat_alpha: float  = 0.1
    success_radius_m: float = 1500.0
    success_alt_m: float    = 1000.0
    shaping_gamma: float    = 1.0
    progress_scale: float   = 0.001
    align_weight: float     = 0.05
    step_penalty: float     = -0.01
    goal_bonus: float       = 1000.0
    kill_penalty: float     = -500.0
    alt_hold_deadzone_m: float = 1000.0
    alt_hold_weight: float     = 2.0e-5
    vert_speed_weight: float   = 5.0e-5
    min_alt_m: float           = 150.0
    max_sim_time_s: float      = 400.0
    fail_penalty: float        = -100.0
    crash_penalty: float       = -500.0


@dataclass
class Waypoint:
    lat_deg: float; lon_deg: float; alt_m: float; los_req_deg: float; dist_m: float


def latlon_to_ne_m(ref_lat, ref_lon, lat, lon):
    north = haversine((ref_lat, ref_lon), (lat, ref_lon)) * 1000.0
    east  = haversine((ref_lat, ref_lon), (ref_lat, lon)) * 1000.0
    if lat < ref_lat: north *= -1.0
    if lon < ref_lon: east  *= -1.0
    return float(north), float(east)


class JSBSimF16SAMEvasionEnv:
    """Existing SAM Evasion environment (Same implementation as training script, training logic removed)."""

    def __init__(self, *, mission: SAMEvasionConfig, agent_steps=5, settle_steps=5, seed=None):
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        self.fdm.set_dt(1.0 / 50.0)
        if not self.fdm.load_model("f16"):
            raise RuntimeError("Failed to load JSBSim model f16")

        self._base_ic = {
            "ic/h-sl-ft": 10000, "ic/u-fps": 360, "ic/v-fps": 0, "ic/w-fps": 0,
            "ic/long-gc-deg": 2.3, "ic/lat-gc-deg": 2.3,
            "ic/terrain-elevation-ft": 10, "ic/psi-true-deg": 90, "ic/roc-fpm": 0,
        }
        self.agent_steps  = int(agent_steps)
        self.settle_steps = int(settle_steps)
        self.mission      = mission
        self._rng         = np.random.default_rng(seed)

        self._waypoints: List[Waypoint] = []
        self._wp_idx = 0
        self._prev_phi = 0.0; self._prev_phi_wp_idx = 0
        self._frame_stack: Deque[np.ndarray] = deque(maxlen=5)
        self._sam_lat = 0.0; self._sam_lon = 0.0; self._sam_alt = 0.0
        self._sam_north_m = 0.0; self._sam_east_m = 0.0

        self._extra_ic: dict = {}      # Hook for subclasses or scenarios to override IC
        self._apply_initial_conditions()

    # -- Apply IC --
    def _apply_initial_conditions(self):
        ic = {**self._base_ic, **self._extra_ic}
        for prop, val in ic.items():
            self.fdm.set_property_value(prop, val)
        self.fdm.reset_to_initial_conditions(0)
        self.fdm.set_property_value("propulsion/starter_cmd", 1)
        self.fdm.set_property_value("propulsion/engine/set-running", 1)
        self.fdm.set_property_value("fcs/throttle-cmd-norm", 1.0)
        self.fdm.set_property_value("gear/gear-cmd-norm", 0)
        for _ in range(self.settle_steps):
            self.fdm.run()
        self._sample_waypoints()
        self._wp_idx = 0
        r0 = self._range_2d_to_wp_idx(0)
        self._prev_phi = -float(r0); self._prev_phi_wp_idx = 0
        self._frame_stack.clear()
        for _ in range(5):
            self.fdm.run(); self._update_frame()

    def _sample_waypoints(self):
        lat0 = float(self.fdm.get_property_value("position/lat-gc-deg"))
        lon0 = float(self.fdm.get_property_value("position/long-gc-deg"))
        alt0 = float(self.fdm.get_property_value("position/h-sl-meters"))
        heading0 = float(self.fdm.get_property_value("attitude/psi-deg"))
        m = self.mission

        dist_m  = float(self._rng.uniform(m.goal_dist_min_m, m.goal_dist_max_m))
        offset  = float(self._rng.uniform(-m.goal_los_deg, m.goal_los_deg))
        bearing = (heading0 + offset) % 360.0
        br      = math.radians(bearing)
        north_m = dist_m * math.cos(br); east_m = dist_m * math.sin(br)
        up_m    = float(self._rng.uniform(m.goal_up_min_m, m.goal_up_max_m))
        dlat, dlon = meters_to_latlon_deg(north_m, east_m, ref_lat_deg=lat0)
        goal_lat = lat0 + dlat; goal_lon = lon0 + dlon; goal_alt = alt0 + up_m
        self._waypoints = [Waypoint(lat_deg=goal_lat, lon_deg=goal_lon, alt_m=goal_alt,
                                     los_req_deg=abs(m.goal_los_deg), dist_m=dist_m)]

        frac  = float(self._rng.uniform(m.sam_place_frac_min, m.sam_place_frac_max))
        sam_n = frac * north_m; sam_e = frac * east_m
        if abs(m.sam_lateral_offset_m) > 1.0:
            perp  = br + math.pi / 2
            sam_n += m.sam_lateral_offset_m * math.cos(perp)
            sam_e += m.sam_lateral_offset_m * math.sin(perp)
        self._sam_north_m = sam_n; self._sam_east_m = sam_e
        sdlat, sdlon = meters_to_latlon_deg(sam_n, sam_e, ref_lat_deg=lat0)
        self._sam_lat = lat0 + sdlat; self._sam_lon = lon0 + sdlon; self._sam_alt = 0.0

    # -- Reset & Observation --
    def reset(self):
        self._apply_initial_conditions()
        return self._get_observation()

    def get_position(self):
        return (float(self.fdm.get_property_value("position/lat-gc-deg")),
                float(self.fdm.get_property_value("position/long-gc-deg")),
                float(self.fdm.get_property_value("position/h-sl-meters")))

    def get_waypoint(self):
        idx = min(int(self._wp_idx), max(len(self._waypoints)-1, 0))
        wp  = self._waypoints[idx]
        return float(wp.lat_deg), float(wp.lon_deg), float(wp.alt_m)

    def get_waypoints(self):
        return [(float(w.lat_deg), float(w.lon_deg), float(w.alt_m)) for w in self._waypoints]

    def get_sam_position(self):
        return self._sam_lat, self._sam_lon, self._sam_alt

    def _range_2d_to_wp_idx(self, wp_idx):
        if not self._waypoints: return 0.0
        idx = min(max(int(wp_idx), 0), len(self._waypoints)-1)
        lat, lon, _ = self.get_position(); wp = self._waypoints[idx]
        return float(haversine((lat, lon), (wp.lat_deg, wp.lon_deg)) * 1000.0 + 1e-6)

    def _threat_factor(self):
        lat, lon, alt_m = self.get_position()
        d2d  = haversine((lat, lon), (self._sam_lat, self._sam_lon)) * 1000.0
        dalt = alt_m - self._sam_alt
        return float((d2d / self.mission.r_horiz_m)**2 + (dalt / self.mission.r_vert_m)**2)

    def _dist_to_sam_2d(self):
        lat, lon, _ = self.get_position()
        return float(haversine((lat, lon), (self._sam_lat, self._sam_lon)) * 1000.0 + 1e-6)

    def _get_state(self):
        alt_m = float(self.fdm.get_property_value("position/h-sl-meters"))
        return np.array([
            float(self.fdm.get_property_value("velocities/u-fps"))   / 500.0,
            float(self.fdm.get_property_value("velocities/v-fps"))   / 100.0,
            float(self.fdm.get_property_value("velocities/w-fps"))   / 100.0,
            alt_m / 10000.0,
            float(self.fdm.get_property_value("velocities/p-rad_sec")) / 3.0,
            float(self.fdm.get_property_value("velocities/q-rad_sec")) / 3.0,
            float(self.fdm.get_property_value("velocities/r-rad_sec")) / 3.0,
            float(self.fdm.get_property_value("attitude/phi-deg"))   / 180.0,
            float(self.fdm.get_property_value("attitude/theta-deg")) / 90.0,
            float(self.fdm.get_property_value("attitude/psi-deg"))   / 180.0,
            float(self.fdm.get_property_value("attitude/pitch-rad")) / 1.57,
        ], dtype=np.float32)

    def _positional_geo(self):
        lat, lon, alt_m = self.get_position()
        wp_lat, wp_lon, wp_alt = self.get_waypoint()
        north_err, east_err = latlon_to_ne_m(lat, lon, wp_lat, wp_lon)
        up_err    = float(wp_alt - alt_m)
        range_2d  = float(haversine((lat, lon), (wp_lat, wp_lon)) * 1000.0 + 1e-6)
        heading   = float(self.fdm.get_property_value("attitude/psi-deg"))
        brg       = float(bearing_deg(lat, lon, wp_lat, wp_lon))
        berr_rad  = math.radians(float(wrap180(brg - heading)))
        psi       = math.radians(heading); c, s = math.cos(psi), math.sin(psi)
        fwd  = float(c * north_err + s * east_err)
        left = float(-s * north_err + c * east_err)
        return np.array([
            fwd/10000.0, left/10000.0, up_err/5000.0, range_2d/10000.0,
            math.sin(berr_rad), math.cos(berr_rad),
            1.0, -1.0, -1.0, -1.0,
        ], dtype=np.float32)

    def _sam_obs(self):
        lat, lon, alt_m = self.get_position()
        sam_n, sam_e = latlon_to_ne_m(lat, lon, self._sam_lat, self._sam_lon)
        heading = float(self.fdm.get_property_value("attitude/psi-deg"))
        psi  = math.radians(heading); c, s = math.cos(psi), math.sin(psi)
        fwd  = float(c * sam_n + s * sam_e)
        left = float(-s * sam_n + c * sam_e)
        tf   = self._threat_factor()
        # * _threat_visible can be overridden to False in subclasses (Pop-up SAM)
        if not getattr(self, "_threat_visible", True) or alt_m <= self.mission.radar_floor_m:
            in_radar = 0.0
        else:
            in_radar = 1.0 if tf < 4.0 else 0.0
        return np.array([tf / 10.0, fwd / 100000.0, left / 100000.0, in_radar], dtype=np.float32)

    def _update_frame(self):
        frame = np.concatenate([self._get_state(), self._positional_geo(), self._sam_obs()])
        self._frame_stack.append(frame.astype(np.float32))

    def _get_observation(self):
        if len(self._frame_stack) == 0: self._update_frame()
        while len(self._frame_stack) < 5:
            self._frame_stack.append(self._frame_stack[-1].copy())
        obs = np.expand_dims(np.array(list(self._frame_stack), dtype=np.float32), axis=0)
        return np.reshape(obs, (1, 5, -1))

    # -- Reward & Termination --
    def _reward_done(self, *, phi_prev, wp_idx_prev):
        lat, lon, alt_m = self.get_position()
        wp_idx_prev = min(max(int(wp_idx_prev), 0), max(len(self._waypoints)-1, 0))
        wp    = self._waypoints[wp_idx_prev]
        wp_lat, wp_lon, wp_alt = float(wp.lat_deg), float(wp.lon_deg), float(wp.alt_m)
        range_2d = float(haversine((lat, lon), (wp_lat, wp_lon)) * 1000.0 + 1e-6)
        up_err   = float(wp_alt - alt_m)
        range_3d = float(math.sqrt(range_2d**2 + up_err**2))
        heading  = float(self.fdm.get_property_value("attitude/psi-deg"))
        brg      = float(bearing_deg(lat, lon, wp_lat, wp_lon))
        berr_rad = math.radians(float(wrap180(brg - heading)))
        align    = float(math.cos(berr_rad))
        m        = self.mission

        phi_next = -float(range_2d)
        shaping  = (1.0 * phi_next) - float(phi_prev)
        progress = shaping * float(m.progress_scale)
        reward   = progress + m.align_weight * align + m.step_penalty

        dz         = m.alt_hold_deadzone_m if m.alt_hold_deadzone_m > 0 else m.success_alt_m
        alt_penalty  = -m.alt_hold_weight * max(0.0, abs(up_err) - dz)
        w_fps        = float(self.fdm.get_property_value("velocities/w-fps"))
        vert_penalty = -m.vert_speed_weight * abs(w_fps)
        reward      += alt_penalty + vert_penalty

        tf      = self._threat_factor()
        tf_radar = 4.0
        r_threat = 0.0
        killed_by_sam = False

        # No threat reward/termination if _threat_visible is False in Pop-up SAM etc.
        threat_live = getattr(self, "_threat_visible", True) and alt_m > m.radar_floor_m

        if threat_live:
            if tf <= 1.0:
                killed_by_sam = True
                reward = float(m.kill_penalty)
                done   = True
                self._prev_phi = phi_next; self._prev_phi_wp_idx = int(wp_idx_prev)
                return float(reward), True, self._build_info(
                    False, False, None, wp_idx_prev, range_2d, range_3d,
                    float(wrap180(brg - heading)), align, up_err, w_fps,
                    alt_penalty, vert_penalty, progress, tf, r_threat,
                    killed_by_sam, False, 0.0)
            elif tf < tf_radar:
                factor   = (tf_radar - tf) / (tf_radar - 1.0)
                r_threat = -m.threat_alpha * (factor ** 2)
                reward  += r_threat

        mission_success = False
        wp_reached = False; reached_idx = None
        if range_2d < m.success_radius_m and abs(up_err) < m.success_alt_m:
            wp_reached = True; reached_idx = int(wp_idx_prev)
            reward     += m.goal_bonus
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                mission_success = True

        if mission_success:
            self._prev_phi = 0.0; self._prev_phi_wp_idx = int(self._wp_idx)
        elif self._wp_idx != int(wp_idx_prev):
            r_new = self._range_2d_to_wp_idx(int(self._wp_idx))
            self._prev_phi = -float(r_new); self._prev_phi_wp_idx = int(self._wp_idx)
        else:
            self._prev_phi = phi_next; self._prev_phi_wp_idx = int(wp_idx_prev)

        done      = bool(mission_success)
        sim_time  = float(self.fdm.get_property_value("simulation/sim-time-sec"))
        crashed   = False
        if alt_m < m.min_alt_m:
            done = True; crashed = True
            if not mission_success: reward += m.crash_penalty
        elif sim_time > m.max_sim_time_s:
            done = True
            if not mission_success: reward += m.fail_penalty

        info = self._build_info(mission_success, wp_reached, reached_idx, wp_idx_prev,
                                range_2d, range_3d, float(wrap180(brg - heading)), align,
                                up_err, w_fps, alt_penalty, vert_penalty, progress,
                                tf, r_threat, killed_by_sam, crashed, sim_time)
        return float(reward), bool(done), info

    def _build_info(self, mission_success, wp_reached, reached_idx, wp_idx_prev,
                    range_2d, range_3d, berr, align, up_err, w_fps,
                    alt_pen, vert_pen, progress, dist_sam, r_threat, killed, crashed, sim_time):
        return {
            "success": bool(mission_success), "mission_success": bool(mission_success),
            "wp_idx": int(self._wp_idx), "wp_reached": bool(wp_reached),
            "reached_wp_idx": reached_idx, "wp_idx_for_reward": int(wp_idx_prev),
            "success_radius_m": float(self.mission.success_radius_m),
            "success_alt_m":    float(self.mission.success_alt_m),
            "progress":         float(progress),
            "range_2d_m":       float(range_2d),  "range_3d_m":  float(range_3d),
            "bearing_err_deg":  float(berr),       "align":       float(align),
            "up_err_m":         float(up_err),     "w_fps":       float(w_fps),
            "alt_penalty":      float(alt_pen),    "vert_penalty": float(vert_pen),
            "fail_penalty":     0.0,               "sim_time_s":  float(sim_time),
            "threat_factor":    float(dist_sam),   "r_threat":    float(r_threat),
            "killed_by_sam":    bool(killed),      "crashed":     bool(crashed),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        aileron, elevator, rudder, throttle = action
        wp_idx_prev = min(int(self._wp_idx), max(len(self._waypoints)-1, 0))
        phi_prev    = -float(self._range_2d_to_wp_idx(wp_idx_prev))
        self._prev_phi = phi_prev; self._prev_phi_wp_idx = int(wp_idx_prev)

        self.fdm.set_property_value("fcs/aileron-cmd-norm",  float(np.clip(aileron,  -1, 1)))
        self.fdm.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elevator, -1, 1)))
        self.fdm.set_property_value("fcs/rudder-cmd-norm",   float(np.clip(rudder,   -1, 1)))
        self.fdm.set_property_value("fcs/throttle-cmd-norm", float(np.clip(throttle,  0, 1)))
        for _ in range(self.agent_steps):
            self.fdm.run(); self._update_frame()
        self._on_post_step()   # Subclass hook (Pop-up SAM etc.)
        reward, done, info = self._reward_done(phi_prev=phi_prev, wp_idx_prev=wp_idx_prev)
        self._update_frame()
        return self._get_observation(), reward, done, info

    def _on_post_step(self):
        """Post-step hook that can be overridden in subclasses."""
        pass


# -- Subclass for Scenario 3 Only --
class PopupSAMEnv(JSBSimF16SAMEvasionEnv):
    """
    Pop-up SAM Environment.
    - At episode start: _threat_visible=False (Radar OFF)
    - When distance to SAM < popup_radius_m: _threat_visible=True (Radar ON)
    """
    def __init__(self, *, mission: SAMEvasionConfig, popup_radius_m: float = 12000.0,
                 agent_steps=5, settle_steps=5, seed=None):
        self.popup_radius_m = float(popup_radius_m)
        super().__init__(mission=mission, agent_steps=agent_steps,
                         settle_steps=settle_steps, seed=seed)

    def reset(self):
        self._threat_visible    = False
        self._popup_triggered   = False
        self._step_at_popup     = None
        self._popup_step_count  = 0    # Full step counter on reset
        return super().reset()

    def _on_post_step(self):
        self._popup_step_count += 1
        if not self._threat_visible:
            dist = self._dist_to_sam_2d()
            if dist < self.popup_radius_m:
                self._threat_visible  = True
                self._popup_triggered = True
                self._step_at_popup   = self._popup_step_count
                print(f"    [POP-UP] Radar ON! dist={dist/1000:.2f}km "
                      f"at step {self._popup_step_count}")


# -- Model Load Helper --
def find_latest_model_dir(base_name: str) -> str:
    """Find the latest folder starting with base_name in models/ directory"""
    model_root = PROJECT_ROOT / "models"
    dirs = [d for d in model_root.glob(f"{base_name}*") if d.is_dir()]
    if not dirs:
        return f"models/{base_name}"
    # Sort by name (Date_Time format, so latest is last)
    latest_dir = sorted(dirs, key=lambda x: x.name)[-1]
    return str(latest_dir.relative_to(PROJECT_ROOT))


def find_best_episode(model_dir: Path, window: int = 50, max_episode: int = 500) -> int:
    """
    Find episode_metrics.csv in the corresponding runs/ folder under model_dir
    By default, only consider data before max_episode(500).
    """
    # model_dir 예: models/20260219_SAC_SAM_FineTune_20260306_1730
    # 대응하는 csv: runs/20260219_SAC_SAM_FineTune/20260219_SAC_SAM_FineTune_20260306_1730/episode_metrics.csv
    parts = model_dir.parts
    if "models" not in parts:
        print(f"[WARN] 경로에 'models'가 없어 runs/ 매핑 불가: {model_dir}")
        return -1
    
    # "20260219_SAC_SAM_FineTune_20260306_1730"
    dir_name = model_dir.name 
    # "20260219_SAC_SAM_FineTune" (Extract base name)
    base_name = "_".join(dir_name.split("_")[:4]) if "FineTune" in dir_name or "Scratch" in dir_name else dir_name
    
    csv_path = PROJECT_ROOT / "runs" / base_name / dir_name / "episode_metrics.csv"
    
    if not csv_path.exists():
        print(f"[WARN] episode_metrics.csv를 찾을 수 없음: {csv_path}")
        return -1

    import csv
    import numpy as np
    
    scores = []
    episodes = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epi = int(row['episode'])
                if epi >= max_episode:
                    break
                episodes.append(epi)
                scores.append(float(row['score']))
                
        if len(scores) < window:
            print(f"[WARN] Less than {window} data points. Returning the last episode.")
            return episodes[-1] if episodes else -1
            
        # 50-episode rolling average
        best_avg = -float('inf')
        best_epi = -1
        
        for i in range(len(scores) - window + 1):
            window_scores = scores[i:i+window]
            avg = np.mean(window_scores)
            if avg > best_avg:
                best_avg = avg
                # Return the last episode of the interval
                best_epi = episodes[i + window - 1]
                
        print(f"  [Best Episode Info] {dir_name}")
        print(f"  └ 50-ep Moving Avg Best: {best_avg:.1f} (끝 에피소드: {best_epi})")
        return best_epi
        
    except Exception as e:
        print(f"[WARN] CSV 파싱 중 오류 발생: {e}")
        return -1


def load_actor(model_dir: Path, device: torch.device,
               episode: Optional[int] = None,
               state_size: int = 25, action_size: int = 4,
               max_throttle: float = 1.0) -> Actor:
    """Load .pth file of a specific episode or the latest episode from model directory."""
    if episode is not None:
        target_name = f"epi_{episode:05d}.pth"
        path = model_dir / target_name
        if not path.exists():
            raise FileNotFoundError(f"Requested episode file not found: {path}")
        selected = path
    else:
        pth_files = sorted(model_dir.glob("epi_*.pth"))
        if not pth_files:
            raise FileNotFoundError(f".pth 파일이 없습니다: {model_dir}")
        selected = pth_files[-1]

    print(f"  [Load] {selected.name} from {model_dir.name}")
    ckpt  = torch.load(selected, map_location=device, weights_only=False)
    actor = Actor(state_size, action_size, multiplier=max_throttle).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


# -- Run Single Episode --
def run_episode(env: JSBSimF16SAMEvasionEnv, actor: Actor, device: torch.device,
                max_steps: int = 50000, extra_info: Optional[dict] = None):
    state = env.reset()
    done  = False; score = 0.0; steps = 0
    ep_killed = False; ep_crashed = False; min_tf = float("inf"); in_radar_steps = 0
    sum_progress = 0.0
    traj_lat, traj_lon, traj_alt = [], [], []
    traj_r2d, traj_tf = [], []

    lat0, lon0, alt0 = env.get_position()
    traj_lat.append(lat0); traj_lon.append(lon0); traj_alt.append(alt0)
    traj_r2d.append(env._range_2d_to_wp_idx(0))
    traj_tf.append(env._threat_factor())

    while not done and steps < max_steps:
        with torch.no_grad():
            st  = torch.tensor(state, dtype=torch.float32, device=device)
            mu, std = actor(st)
            # deterministic evaluation: tanh(mu)
            act_np = torch.tanh(mu).cpu().numpy().flatten()
            act_np[:3] = np.clip(act_np[:3], -1, 1)
            act_np[3]  = np.clip(act_np[3], 0, 1)

        next_state, reward, done, info = env.step(act_np)
        lat, lon, alt = env.get_position()
        tf = float(info.get("threat_factor", 999.0))
        min_tf = min(min_tf, tf)

        # threat_visible 상태 시에만 in_radar_steps 카운트
        threat_vis = getattr(env, "_threat_visible", True)
        if tf < 4.0 and alt > env.mission.radar_floor_m and threat_vis:
            in_radar_steps += 1
        if info.get("killed_by_sam"):  ep_killed = True
        if info.get("crashed"):        ep_crashed = True
        sum_progress += float(info.get("progress", 0.0))

        traj_lat.append(lat); traj_lon.append(lon); traj_alt.append(alt)
        traj_r2d.append(float(info.get("range_2d_m", 0.0)))
        traj_tf.append(tf)
        state = next_state; score += reward; steps += 1

    msuc = bool(info.get("mission_success", False))
    ep_info = {
        "success": msuc, "killed_by_sam": ep_killed, "crashed": ep_crashed,
        "score": score, "steps": steps, "min_tf": min_tf if math.isfinite(min_tf) else 999.0,
        "in_radar_steps": in_radar_steps,
        "progress_mean": sum_progress / max(steps, 1),
        "range2d_last_m": traj_r2d[-1] if traj_r2d else 0.0,
    }
    if extra_info:
        ep_info.update(extra_info)
    traj = dict(lat=traj_lat, lon=traj_lon, alt=traj_alt, r2d=traj_r2d, tf=traj_tf,
                wps=env.get_waypoints(), sam=env.get_sam_position(),
                ref_lat=lat0, ref_lon=lon0, ref_alt=alt0)
    return ep_info, traj


# -- Trajectory Plot --
def plot_episode(ep_idx: int, ep_info: dict, traj: dict,
                 mission: SAMEvasionConfig, save_dir: Path,
                 scenario_label: str = "", extra_title: str = ""):
    try:
        ref_lat = traj["ref_lat"]; ref_lon = traj["ref_lon"]; ref_alt = traj["ref_alt"]
        axy = [latlon_to_ne_m(ref_lat, ref_lon, la, lo)
               for la, lo in zip(traj["lat"], traj["lon"])]
        an = np.array([p[0] for p in axy], dtype=np.float32)
        ae = np.array([p[1] for p in axy], dtype=np.float32)
        az = np.array(traj["alt"],  dtype=np.float32) - ref_alt

        wn, we, wz = [], [], []
        for wl, wlo, wa in traj["wps"]:
            n, e = latlon_to_ne_m(ref_lat, ref_lon, wl, wlo)
            wn.append(n); we.append(e); wz.append(wa - ref_alt)
        wn, we, wz = np.array(wn), np.array(we), np.array(wz)
        sn, se = latlon_to_ne_m(ref_lat, ref_lon, traj["sam"][0], traj["sam"][1])
        sz = traj["sam"][2] - ref_alt

        r_horiz = float(mission.r_horiz_m)
        r_vert  = float(mission.r_vert_m)
        r_s     = float(mission.success_radius_m)

        fig = plt.figure(figsize=(18, 8))
        gs  = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 0.7], hspace=0.35, wspace=0.25)
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[1, 1])
        ax_leg = fig.add_subplot(gs[:, 2]); ax_leg.axis("off")

        # 3D
        pts = np.column_stack([ae, an, az]).astype(np.float32)
        if len(pts) >= 2:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc   = Line3DCollection(segs, colors="C0", linewidths=1.8, alpha=0.85)
            ax3d.add_collection3d(lc)
        z_floor = float(min(az.min(), sz) - 500)
        ax3d.scatter(ae[0], an[0], az[0], s=80, color="C0", edgecolors="k", label="Start")
        ax3d.scatter(we[0], wn[0], wz[0], s=100, color="C2", edgecolors="k", label="Goal")
        ax3d.scatter(se, sn, sz, s=100, color="red", edgecolors="k", marker="^", label="SAM")
        # SAM Threat Sphere (Outer: 2x r_horiz, Inner: 1x r_horiz)
        theta_s = np.linspace(0, 2*math.pi, 60, dtype=np.float32)
        phi_s   = np.linspace(0, math.pi,   30, dtype=np.float32)
        for r_draw, clr, a_ in [(r_horiz*2, "red", 0.04), (r_horiz, "darkred", 0.09)]:
            uu, vv = np.meshgrid(theta_s, phi_s)
            ax3d.plot_wireframe(
                float(se) + r_draw * np.cos(uu) * np.sin(vv),
                float(sn) + r_draw * np.sin(uu) * np.sin(vv),
                float(sz) + r_draw * np.cos(vv),
                color=clr, linewidth=0.4, alpha=a_)

        ax3d.set_title("3D Trajectory + SAM Threat")
        ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Up (m)")
        try: ax3d.set_proj_type("ortho")
        except: pass
        ax3d.view_init(elev=22, azim=-55)
        m_s = max(r_horiz * 1.2, 1000)
        all_e = np.concatenate([ae, we, [se]]); all_n = np.concatenate([an, wn, [sn]])
        ax3d.set_xlim(all_e.min()-m_s, all_e.max()+m_s)
        ax3d.set_ylim(all_n.min()-m_s, all_n.max()+m_s)
        ax3d.set_zlim(min(z_floor, az.min())-200, az.max()+500)

        # XY (탑뷰)
        ax_xy.plot(ae, an, color="C0", linewidth=1.8, alpha=0.85, label="Agent")
        ax_xy.scatter(ae[0], an[0], color="C0", s=40, edgecolors="k")
        ax_xy.scatter(we[0], wn[0], color="C2", s=60, edgecolors="k")
        for r_c, clr, alph in [(r_horiz*2, "red", 0.22), (r_horiz, "darkred", 0.30)]:
            t_ = np.linspace(0, 2*math.pi, 200)
            ax_xy.plot(float(se)+r_c*np.cos(t_), float(sn)+r_c*np.sin(t_),
                       color=clr, linewidth=1.5, alpha=alph)
        ax_xy.scatter(se, sn, color="red", marker="^", s=80, edgecolors="k")
        ax_xy.set_title("Top-Down (XY)")
        ax_xy.set_aspect("equal", adjustable="box"); ax_xy.grid(True, ls="--", alpha=0.4)

        # XZ (측면)
        ax_xz.plot(ae, az, color="C0", linewidth=1.8, alpha=0.85)
        ax_xz.scatter(we[0], wz[0], color="C2", s=60, edgecolors="k")
        ax_xz.scatter(se, sz, color="red", marker="^", s=80, edgecolors="k")
        ax_xz.axvspan(float(se)-r_horiz*2, float(se)+r_horiz*2, color="red", alpha=0.05)
        ax_xz.axvspan(float(se)-r_horiz,   float(se)+r_horiz,   color="darkred", alpha=0.07)
        ax_xz.set_title("Side View (XZ)"); ax_xz.grid(True, ls="--", alpha=0.4)

        # Result Determination
        if ep_info.get("killed_by_sam"): res = "KILLED BY SAM"
        elif ep_info.get("success"):     res = "SUCCESS"
        elif ep_info.get("crashed"):     res = "CRASHED"
        else:                            res = "Failed (Timeout)"

        fig.suptitle(
            f"[{scenario_label}] Ep {ep_idx:03d} — {res}{extra_title}\n"
            f"Score: {ep_info['score']:.1f} | Steps: {ep_info['steps']} "
            f"| min_TF: {ep_info['min_tf']:.2f} | rH={r_horiz/1000:.1f}km",
            fontsize=12, y=0.99)

        box_txt = (f"SAM rH={r_horiz/1000:.1f}km rV={r_vert/1000:.1f}km\n"
                   f"Min TF: {ep_info['min_tf']:.2f}\n"
                   f"In-Radar: {ep_info['in_radar_steps']} steps\n"
                   f"Killed: {ep_info['killed_by_sam']} | Crashed: {ep_info['crashed']}")
        ax_leg.text(0, 0.5, box_txt, transform=ax_leg.transAxes, fontsize=10, va="center",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="gray"))

        save_dir.mkdir(parents=True, exist_ok=True)
        fname = "succ" if ep_info.get("success") else ("kill" if ep_info.get("killed_by_sam") else "fail")
        fig.savefig(save_dir / f"ep{ep_idx:04d}_{fname}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as ex:
        print(f"    [Plot Error] ep{ep_idx}: {ex}")


# -- Save CSV --
def save_csv(rows: list, out_path: Path):
    if not rows: return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"  [CSV] {out_path}")


# -- Summary Output --
def print_summary(label: str, rows: list):
    if not rows:
        print(f"  {label}: (No data)"); return
    n       = len(rows)
    n_succ  = sum(r.get("success", 0)       for r in rows)
    n_kill  = sum(r.get("killed_by_sam", 0) for r in rows)
    n_crash = sum(r.get("crashed", 0)        for r in rows)
    avg_tf  = np.mean([r.get("min_tf", 999) for r in rows])
    avg_rad = np.mean([r.get("in_radar_steps", 0) for r in rows])
    print(f"  {label:<40} | n={n:>3} | "
          f"Succ={n_succ/n*100:5.1f}% | Kill={n_kill/n*100:5.1f}% | "
          f"Crash={n_crash/n*100:5.1f}% | "
          f"AvgMinTF={avg_tf:6.3f} | AvgRadarSteps={avg_rad:6.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# 시나리오별 실행 함수
# ══════════════════════════════════════════════════════════════════════════════

def scenario_threat_scaling(models: dict, device: torch.device, episodes: int, seed: int):
    """
    시나리오 1: Threat Scaling
    SAM 반경을 Base(10km/5km) / 120%(12km/6km) / 150%(15km/7.5km)로 바꿔 평가.
    """
    configs = [
        ("Base (100%)",  10000.0, 5000.0),
        ("Scale 120%",   12000.0, 6000.0),
        ("Scale 150%",   15000.0, 7500.0),
    ]
    print("\n" + "="*90)
    print("Scenario 1: Threat Scaling Test")
    print("="*90)

    for model_tag, actor in models.items():
        all_rows = []
        for cname, r_h, r_v in configs:
            mission = SAMEvasionConfig(r_horiz_m=r_h, r_vert_m=r_v)
            env     = JSBSimF16SAMEvasionEnv(mission=mission, seed=seed)
            fig_dir = FIG_BASE / "threat_scaling" / model_tag / cname.replace(" ", "_")
            rows    = []
            print(f"\n  [{model_tag}] {cname} (rH={r_h/1000:.0f}km rV={r_v/1000:.1f}km)  "
                  f"{episodes}ep ...")
            for ep in range(episodes):
                ei, traj = run_episode(env, actor, device,
                                       extra_info={"config": cname,
                                                   "r_horiz_km": r_h/1000,
                                                   "r_vert_km":  r_v/1000})
                rows.append({"episode": ep, "model": model_tag, "config": cname,
                             "r_horiz_km": r_h/1000, "r_vert_km": r_v/1000,
                             **{k: v for k, v in ei.items() if k not in ("config",)}})
                if ep % 10 == 0:
                    plot_episode(ep, ei, traj, mission, fig_dir,
                                 scenario_label=f"ThreatScale/{cname}",
                                 extra_title=f" [{model_tag}]")
                tag = ("S" if ei["success"] else
                       ("K" if ei["killed_by_sam"] else
                        ("C" if ei["crashed"] else ".")))
                print(f"    ep{ep:>3} {tag}  min_TF={ei['min_tf']:.2f}  "
                      f"rad={ei['in_radar_steps']:>4}  score={ei['score']:>8.1f}")
            all_rows.extend(rows)
            print_summary(f"  {model_tag} | {cname}", rows)

        save_csv(all_rows,
                 RESULTS_DIR / f"robustness_threat_scaling_{model_tag}.csv")


def scenario_initial_perturb(models: dict, device: torch.device, episodes: int, seed: int):
    """
    Scenario 2: Initial State Perturbation
    Start altitude 2000-4000m random, heading +-30 deg random perturbation.
    """
    print("\n" + "="*90)
    print("Scenario 2: Initial State Perturbation")
    print("="*90)

    rng = np.random.default_rng(seed + 100)

    for model_tag, actor in models.items():
        mission = SAMEvasionConfig()  # 기본 SAM 반경 유지
        env     = JSBSimF16SAMEvasionEnv(mission=mission, seed=seed)
        fig_dir = FIG_BASE / "init_perturb" / model_tag
        rows    = []
        print(f"\n  [{model_tag}] {episodes}ep ...")
        for ep in range(episodes):
            # 에피소드마다 고도·헤딩을 무작위로 덮어쓰기
            alt_m     = float(rng.uniform(2000.0, 4000.0))
            hdg_noise = float(rng.uniform(-30.0, 30.0))   # 기본 헤딩(90°)에서 ±30°
            env._extra_ic = {
                "ic/h-sl-ft":       alt_m * FT_PER_M,
                "ic/psi-true-deg":  90.0 + hdg_noise,
            }
            ei, traj = run_episode(env, actor, device,
                                   extra_info={"init_alt_m": alt_m,
                                               "init_hdg_noise_deg": hdg_noise})
            rows.append({"episode": ep, "model": model_tag,
                         "init_alt_m": round(alt_m, 1),
                         "init_hdg_noise_deg": round(hdg_noise, 2),
                         **{k: v for k, v in ei.items() if k not in ("init_alt_m", "init_hdg_noise_deg")}})
            if ep % 10 == 0:
                plot_episode(ep, ei, traj, mission, fig_dir,
                             scenario_label="InitPerturb",
                             extra_title=f" [{model_tag}] alt={alt_m:.0f}m hdg±{abs(hdg_noise):.1f}°")
            tag = ("✓" if ei["success"] else
                   ("☠" if ei["killed_by_sam"] else
                    ("X" if ei["crashed"] else ".")))
            print(f"    ep{ep:>3} {tag}  alt={alt_m:6.0f}m  hdg_off={hdg_noise:+6.1f}°  "
                  f"min_TF={ei['min_tf']:.2f}  score={ei['score']:>8.1f}")

        env._extra_ic = {}  # 초기화
        save_csv(rows, RESULTS_DIR / f"robustness_init_perturb_{model_tag}.csv")
        print_summary(f"  {model_tag} | Init Perturbation", rows)


def scenario_popup_sam(models: dict, device: torch.device, episodes: int, seed: int):
    """
    Scenario 3: Pop-up SAM
    Radar OFF at start -> Radar ON when within 12km -> Evaluate Break Turn response.
    """
    print("\n" + "="*90)
    print("Scenario 3: Pop-up SAM")
    print("="*90)

    POPUP_RADIUS = 12000.0  # 12 km

    for model_tag, actor in models.items():
        mission = SAMEvasionConfig()
        env     = PopupSAMEnv(mission=mission, popup_radius_m=POPUP_RADIUS, seed=seed)
        fig_dir = FIG_BASE / "popup_sam" / model_tag
        rows    = []
        print(f"\n  [{model_tag}] {episodes}ep  (popup_radius={POPUP_RADIUS/1000:.0f}km) ...")
        for ep in range(episodes):
            ei, traj = run_episode(env, actor, device)
            triggered    = getattr(env, "_popup_triggered",  False)
            step_popup   = getattr(env, "_step_at_popup",    None)
            rows.append({
                "episode": ep, "model": model_tag,
                "popup_triggered":    int(triggered),
                "step_at_popup":      step_popup if step_popup is not None else -1,
                **{k: v for k, v in ei.items()}
            })
            if ep % 10 == 0:
                plot_episode(ep, ei, traj, mission, fig_dir,
                             scenario_label="PopupSAM",
                             extra_title=f" [{model_tag}] triggered={triggered} @step{step_popup}")
            tag  = ("✓" if ei["success"] else
                    ("☠" if ei["killed_by_sam"] else
                     ("X" if ei["crashed"] else ".")))
            trig_str = f"triggered@{step_popup}" if triggered else "NOT triggered"
            print(f"    ep{ep:>3} {tag}  {trig_str:<22}  "
                  f"min_TF={ei['min_tf']:.2f}  score={ei['score']:>8.1f}")

        save_csv(rows, RESULTS_DIR / f"robustness_popup_sam_{model_tag}.csv")
        print_summary(f"  {model_tag} | Pop-up SAM", rows)

        # Pop-up 특화 요약
        n = len(rows)
        if n:
            n_trig   = sum(r["popup_triggered"] for r in rows)
            trig_rows = [r for r in rows if r["popup_triggered"]]
            n_surv   = sum(1 for r in trig_rows if not r.get("killed_by_sam", False))
            print(f"  ┌ Pop-up triggered: {n_trig}/{n} ep ({n_trig/n*100:.1f}%)")
            if trig_rows:
                avg_step = np.mean([r["step_at_popup"] for r in trig_rows])
                print(f"  ├ Avg step at popup trigger: {avg_step:.1f}")
                print(f"  └ Survived after popup:  {n_surv}/{n_trig} "
                      f"({n_surv/n_trig*100:.1f}%)" if n_trig else "")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SAM Evasion Robustness Test (Threat Scaling / Init Perturb / Pop-up SAM)")
    parser.add_argument("--scratch_dir",  type=str,
                        default=find_latest_model_dir("20260219_SAC_SAM_Scratch"),
                        help="Scratch model directory")
    parser.add_argument("--finetune_dir", type=str,
                        default=find_latest_model_dir("20260219_SAC_SAM_FineTune"),
                        help="Fine-tune model directory")
    parser.add_argument("--ft_return_dir", type=str,
                        default=find_latest_model_dir("20260306_SAC_SAM_FineTune_From_Return"),
                        help="FT(Return) model directory")
    parser.add_argument("--episodes",    type=int, default=50,
                        help="Number of episodes per scenario")
    parser.add_argument("--max_steps",   type=int, default=50000)
    parser.add_argument("--scenario",    type=str, default="all",
                        choices=["all", "threat", "perturb", "popup"],
                        help="Scenario to execute")
    parser.add_argument("--scratch_epi",  type=str, default=499, help="Scratch episode number (Latest if empty, use 'best' for highest 50ep moving average)")
    parser.add_argument("--finetune_epi", type=str, default=499, help="Fine-tune episode number (Latest if empty, use 'best' for highest 50ep moving average)")
    parser.add_argument("--ft_return_epi", type=str, default=499, help="FT(Return) episode number (Latest if empty, use 'best' for highest 50ep moving average)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Episodes per scenario: {args.episodes}")
    print(f"Scenario: {args.scenario}")

    scratch_dir  = PROJECT_ROOT / args.scratch_dir
    finetune_dir = PROJECT_ROOT / args.finetune_dir
    ft_return_dir = PROJECT_ROOT / args.ft_return_dir

    def resolve_episode(epi_arg, m_dir):
        if epi_arg is None or str(epi_arg).lower() == "latest":
            return None
        elif str(epi_arg).lower() == "best":
            best_e = find_best_episode(m_dir)
            return best_e if best_e >= 0 else None
        else:
            try:
                return int(epi_arg)
            except ValueError:
                return None

    s_epi = resolve_episode(args.scratch_epi, scratch_dir)
    f_epi = resolve_episode(args.finetune_epi, finetune_dir)
    fr_epi = resolve_episode(args.ft_return_epi, ft_return_dir)

    # 모델 로드 (state_size=25: 11 kinematics + 10 geo + 4 SAM)
    models = {}
    if scratch_dir.exists():
        models["Scratch"]  = load_actor(scratch_dir,  device, episode=s_epi)
    else:
        print(f"[WARN] Scratch 모델 디렉토리 없음: {scratch_dir}")

    if finetune_dir.exists():
        models["FineTune"] = load_actor(finetune_dir, device, episode=f_epi)
    else:
        print(f"[WARN] FineTune 모델 디렉토리 없음: {finetune_dir}")

    if ft_return_dir.exists():
        models["FT_Return"] = load_actor(ft_return_dir, device, episode=fr_epi)
    else:
        print(f"[WARN] FT_Return 모델 디렉토리 없음: {ft_return_dir}")

    if not models:
        raise RuntimeError("로드할 모델이 없습니다. 경로를 확인하세요.")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"\nExperiment started: {ts}")
    print("="*90)

    run_threat  = args.scenario in ("all", "threat")
    run_perturb = args.scenario in ("all", "perturb")
    run_popup   = args.scenario in ("all", "popup")

    if run_threat:
        scenario_threat_scaling(models, device, args.episodes, args.seed)
    if run_perturb:
        scenario_initial_perturb(models, device, args.episodes, args.seed)
    if run_popup:
        scenario_popup_sam(models, device, args.episodes, args.seed)

    # -- Global Summary --
    print("\n" + "="*90)
    print(f"Robustness test complete: {datetime.now().strftime('%Y%m%d_%H%M')}")
    print(f"Result CSVs -> {RESULTS_DIR}/")
    print(f"Trajectory PNGs -> {FIG_BASE}/")
    print("="*90)


if __name__ == "__main__":
    main()
