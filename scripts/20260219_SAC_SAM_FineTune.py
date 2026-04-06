import argparse
import csv
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import jsbsim
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore


class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None
    def close(self) -> None:
        return None


try:
    from haversine import haversine as haversine  # type: ignore
except Exception:
    def haversine(p1, p2) -> float:
        lat1, lon1 = p1; lat2, lon2 = p2
        r = 6371.0088
        phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlmb = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
        return float(r * 2.0 * math.asin(math.sqrt(a)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

SCRIPT_TAG = "20260219_SAC_SAM_FineTune"
FIG_DIR = PROJECT_ROOT / "figures" / SCRIPT_TAG
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "runs" / SCRIPT_TAG
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from sac_agent.models import Actor, Critic  # noqa: E402
from sac_agent.replay_buffer import ReplayBuffer  # noqa: E402
from sac_agent.utils import eval_action, get_action, hard_target_update, soft_target_update  # noqa: E402
from envs.init_noise import meters_to_latlon_deg, uniform_symmetric  # noqa: E402
from envs.target_controllers import bearing_deg, wrap180  # noqa: E402

KTS_PER_FPS = 1.0 / 1.6878098571011957


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def fdm_speed_kts(fdm) -> float:
    u = float(fdm.get_property_value("velocities/u-fps"))
    v = float(fdm.get_property_value("velocities/v-fps"))
    w = float(fdm.get_property_value("velocities/w-fps"))
    return float(math.sqrt(u*u + v*v + w*w) * KTS_PER_FPS)


def latlon_to_ne_m(ref_lat, ref_lon, lat, lon):
    north_m = haversine((ref_lat, ref_lon), (lat, ref_lon)) * 1000.0
    east_m = haversine((ref_lat, ref_lon), (ref_lat, lon)) * 1000.0
    if lat < ref_lat: north_m *= -1.0
    if lon < ref_lon: east_m *= -1.0
    return float(north_m), float(east_m)


def add_circle_xy(ax, cx, cy, r, *, color, alpha=0.18, lw=1.5):
    theta = np.linspace(0, 2*math.pi, 180, dtype=np.float32)
    ax.plot(float(cx)+float(r)*np.cos(theta), float(cy)+float(r)*np.sin(theta),
            color=color, linewidth=lw, alpha=alpha)


def add_sphere_3d(ax, cx, cy, cz, r, *, color, alpha=0.10, lw=0.5, n_u=20, n_v=12):
    u = np.linspace(0, 2*math.pi, n_u, dtype=np.float32)
    v = np.linspace(0, math.pi, n_v, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    ax.plot_wireframe(float(cx)+float(r)*np.cos(uu)*np.sin(vv),
                      float(cy)+float(r)*np.sin(uu)*np.sin(vv),
                      float(cz)+float(r)*np.cos(vv),
                      color=color, linewidth=lw, alpha=alpha)


def add_cylinder_3d(ax, cx, cy, cz, r, half_h, *, color, alpha=0.24, lw=1.0, n_theta=40):
    theta = np.linspace(0, 2*math.pi, n_theta, dtype=np.float32)
    x = float(cx)+float(r)*np.cos(theta); y = float(cy)+float(r)*np.sin(theta)
    z0, z1 = float(cz)-float(half_h), float(cz)+float(half_h)
    ax.plot(x, y, np.full_like(x, z0), color=color, linewidth=lw, alpha=alpha)
    ax.plot(x, y, np.full_like(x, z1), color=color, linewidth=lw, alpha=alpha)
    for k in range(0, len(theta), max(len(theta)//8, 1)):
        ax.plot([x[k],x[k]], [y[k],y[k]], [z0,z1], color=color, linewidth=lw, alpha=alpha)


# ============================================================
# SAM Evasion Mission Config
# ============================================================
@dataclass
class SAMEvasionConfig:
    """Single-waypoint navigation with SAM threat dome."""
    # Goal waypoint distance from start
    goal_dist_min_m: float = 30000.0
    goal_dist_max_m: float = 40000.0
    goal_los_deg: float = 10.0  # narrow LOS so SAM blocks the path
    goal_up_min_m: float = -300.0
    goal_up_max_m: float = 300.0

    # SAM placement: fraction along Start→Goal line (0.5 = midpoint)
    sam_place_frac_min: float = 0.4
    sam_place_frac_max: float = 0.6
    # SAM lateral offset from the line (meters). 0 = exactly on the line.
    sam_lateral_offset_m: float = 0.0

    # Threat dome radii (Ellipsoid)
    r_horiz_m: float = 10000.0
    r_vert_m: float = 5000.0
    radar_floor_m: float = 0.0

    # Threat reward weight
    threat_alpha: float = 0.1

    # Success condition
    success_radius_m: float = 1500.0
    success_alt_m: float = 1000.0

    # Reward shaping
    shaping_gamma: float = 1.0
    progress_scale: float = 0.001  # 1km = +1
    align_weight: float = 0.05
    step_penalty: float = -0.01
    goal_bonus: float = 1000.0
    kill_penalty: float = -500.0

    # Altitude safety
    alt_hold_deadzone_m: float = 1000.0
    alt_hold_weight: float = 2.0e-5
    vert_speed_weight: float = 5.0e-5

    # Termination
    min_alt_m: float = 150.0
    max_sim_time_s: float = 400.0
    fail_penalty: float = -100.0
    crash_penalty: float = -500.0


@dataclass
class Waypoint:
    lat_deg: float; lon_deg: float; alt_m: float; los_req_deg: float; dist_m: float


# ============================================================
# SAM Evasion Environment
# ============================================================
class JSBSimF16SAMEvasionEnv:
    """
    Single-waypoint navigation with SAM threat dome.

    Observation per frame:
    - agent kinematics (11)
    - relative geo to goal waypoint (10)   [wp_onehot always [1,0,0,0]]
    - SAM relative info (4)
    => 25 features, stacked over 5 frames -> (1, 5, 25)
    """

    def __init__(self, *, agent_steps=5, settle_steps=5, mission=None, seed=None):
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0); self.fdm.set_dt(1.0/50.0)
        self.model = "f16"
        if not self.fdm.load_model(self.model):
            raise RuntimeError("Failed to load JSBSim model f16")

        self.initial_conditions = {
            "ic/h-sl-ft": 10000, "ic/u-fps": 360, "ic/v-fps": 0, "ic/w-fps": 0,
            "ic/long-gc-deg": 2.3, "ic/lat-gc-deg": 2.3, "ic/terrain-elevation-ft": 10,
            "ic/psi-true-deg": 90, "ic/roc-fpm": 0,
        }
        self.agent_steps = int(agent_steps); self.settle_steps = int(settle_steps)
        self.mission = mission or SAMEvasionConfig()
        self._rng = np.random.default_rng(seed)
        self._waypoints: List[Waypoint] = []
        self._wp_idx = 0
        self._prev_phi = 0.0; self._prev_phi_wp_idx = 0
        self._frame_stack: Deque[np.ndarray] = deque(maxlen=5)
        # SAM position in lat/lon/alt (set at reset)
        self._sam_lat = 0.0; self._sam_lon = 0.0; self._sam_alt = 0.0
        # SAM position in local NE (meters from spawn) for plotting
        self._sam_north_m = 0.0; self._sam_east_m = 0.0
        self._apply_initial_conditions()

    def _apply_initial_conditions(self):
        for prop, val in self.initial_conditions.items():
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

        # Goal waypoint
        dist_m = float(self._rng.uniform(m.goal_dist_min_m, m.goal_dist_max_m))
        offset = float(self._rng.uniform(-m.goal_los_deg, m.goal_los_deg))
        bearing = (heading0 + offset) % 360.0
        br = math.radians(bearing)
        north_m = dist_m * math.cos(br); east_m = dist_m * math.sin(br)
        up_m = float(self._rng.uniform(m.goal_up_min_m, m.goal_up_max_m))
        dlat, dlon = meters_to_latlon_deg(north_m, east_m, ref_lat_deg=lat0)
        goal_lat = lat0 + dlat; goal_lon = lon0 + dlon; goal_alt = alt0 + up_m
        self._waypoints = [Waypoint(lat_deg=goal_lat, lon_deg=goal_lon, alt_m=goal_alt,
                                     los_req_deg=abs(m.goal_los_deg), dist_m=dist_m)]

        # SAM placement along Start→Goal line
        frac = float(self._rng.uniform(m.sam_place_frac_min, m.sam_place_frac_max))
        sam_n = frac * north_m; sam_e = frac * east_m
        # Optional lateral offset (perpendicular to bearing)
        if abs(m.sam_lateral_offset_m) > 1.0:
            perp = br + math.pi/2
            sam_n += m.sam_lateral_offset_m * math.cos(perp)
            sam_e += m.sam_lateral_offset_m * math.sin(perp)
        self._sam_north_m = sam_n; self._sam_east_m = sam_e
        sdlat, sdlon = meters_to_latlon_deg(sam_n, sam_e, ref_lat_deg=lat0)
        self._sam_lat = lat0 + sdlat; self._sam_lon = lon0 + sdlon
        self._sam_alt = 0.0  # SAM is on the ground (Alt=0)

    def reset(self):
        self._apply_initial_conditions()
        return self._get_observation()

    def get_position(self):
        return (float(self.fdm.get_property_value("position/lat-gc-deg")),
                float(self.fdm.get_property_value("position/long-gc-deg")),
                float(self.fdm.get_property_value("position/h-sl-meters")))

    def get_waypoint(self):
        idx = min(int(self._wp_idx), max(len(self._waypoints)-1, 0))
        wp = self._waypoints[idx]
        return float(wp.lat_deg), float(wp.lon_deg), float(wp.alt_m)

    def get_waypoints(self):
        return [(float(w.lat_deg), float(w.lon_deg), float(w.alt_m)) for w in self._waypoints]

    def get_wp_index(self): return int(self._wp_idx)

    def get_sam_position(self):
        return self._sam_lat, self._sam_lon, self._sam_alt

    def _current_range_2d_m(self): return self._range_2d_to_wp_idx(int(self._wp_idx))

    def _range_2d_to_wp_idx(self, wp_idx):
        if not self._waypoints: return 0.0
        idx = min(max(int(wp_idx), 0), len(self._waypoints)-1)
        lat, lon, _ = self.get_position(); wp = self._waypoints[idx]
        return float(haversine((lat, lon), (wp.lat_deg, wp.lon_deg)) * 1000.0 + 1e-6)

    def _threat_factor(self):
        lat, lon, alt_m = self.get_position()
        d2d = haversine((lat, lon), (self._sam_lat, self._sam_lon)) * 1000.0
        dalt = alt_m - self._sam_alt
        return float( (d2d / self.mission.r_horiz_m)**2 + (dalt / self.mission.r_vert_m)**2 )

    def _dist_to_sam_2d(self):
        lat, lon, _ = self.get_position()
        return float(haversine((lat, lon), (self._sam_lat, self._sam_lon)) * 1000.0 + 1e-6)

    # --- Observation ---
    def _get_state(self):
        alt_m = float(self.fdm.get_property_value("position/h-sl-meters"))
        return np.array([
            float(self.fdm.get_property_value("velocities/u-fps")) / 500.0,
            float(self.fdm.get_property_value("velocities/v-fps")) / 100.0,
            float(self.fdm.get_property_value("velocities/w-fps")) / 100.0,
            alt_m / 10000.0,
            float(self.fdm.get_property_value("velocities/p-rad_sec")) / 3.0,
            float(self.fdm.get_property_value("velocities/q-rad_sec")) / 3.0,
            float(self.fdm.get_property_value("velocities/r-rad_sec")) / 3.0,
            float(self.fdm.get_property_value("attitude/phi-deg")) / 180.0,
            float(self.fdm.get_property_value("attitude/theta-deg")) / 90.0,
            float(self.fdm.get_property_value("attitude/psi-deg")) / 180.0,
            float(self.fdm.get_property_value("attitude/pitch-rad")) / 1.57,
        ], dtype=np.float32)

    def _positional_geo(self):
        lat, lon, alt_m = self.get_position()
        wp_lat, wp_lon, wp_alt = self.get_waypoint()
        north_err, east_err = latlon_to_ne_m(lat, lon, wp_lat, wp_lon)
        up_err = float(wp_alt - alt_m)
        range_2d = float(haversine((lat, lon), (wp_lat, wp_lon))*1000.0 + 1e-6)
        heading = float(self.fdm.get_property_value("attitude/psi-deg"))
        brg = float(bearing_deg(lat, lon, wp_lat, wp_lon))
        berr_rad = math.radians(float(wrap180(brg - heading)))
        psi = math.radians(heading); c, s = math.cos(psi), math.sin(psi)
        fwd = float(c*north_err + s*east_err); left = float(-s*north_err + c*east_err)
        # Single waypoint: onehot = [1, -1, -1, -1]  (mapped to [-1,1])
        return np.array([
            fwd/10000.0, left/10000.0, up_err/5000.0, range_2d/10000.0,
            math.sin(berr_rad), math.cos(berr_rad),
            1.0, -1.0, -1.0, -1.0,  # wp_onehot mapped to [-1,1]
        ], dtype=np.float32)

    def _sam_obs(self):
        lat, lon, alt_m = self.get_position()
        sam_n, sam_e = latlon_to_ne_m(lat, lon, self._sam_lat, self._sam_lon)
        heading = float(self.fdm.get_property_value("attitude/psi-deg"))
        psi = math.radians(heading); c, s = math.cos(psi), math.sin(psi)
        fwd = float(c*sam_n + s*sam_e); left = float(-s*sam_n + c*sam_e)
        tf = self._threat_factor()
        # Terrain masking: if below radar floor, completely invisible to SAM
        if alt_m <= self.mission.radar_floor_m:
            tf = 999.0  # Safe factor
            in_radar = 0.0
        else:
            in_radar = 1.0 if tf < 4.0 else 0.0  # Radar reaches twice the horizontal range (factor < 4.0 -> r < 2R)
        return np.array([tf/10.0, fwd/100000.0, left/100000.0, in_radar], dtype=np.float32)

    def _update_frame(self):
        frame = np.concatenate([self._get_state(), self._positional_geo(), self._sam_obs()])
        self._frame_stack.append(frame.astype(np.float32))

    def _get_observation(self):
        if len(self._frame_stack) == 0: self._update_frame()
        while len(self._frame_stack) < 5:
            self._frame_stack.append(self._frame_stack[-1].copy())
        obs = np.expand_dims(np.array(list(self._frame_stack), dtype=np.float32), axis=0)
        return np.reshape(obs, (1, 5, -1))

    # --- Reward ---
    def _reward_done(self, *, phi_prev, wp_idx_prev):
        lat, lon, alt_m = self.get_position()
        wp_idx_prev = min(max(int(wp_idx_prev), 0), max(len(self._waypoints)-1, 0))
        wp = self._waypoints[wp_idx_prev]
        wp_lat, wp_lon, wp_alt = float(wp.lat_deg), float(wp.lon_deg), float(wp.alt_m)
        range_2d = float(haversine((lat, lon), (wp_lat, wp_lon))*1000.0 + 1e-6)
        up_err = float(wp_alt - alt_m)
        range_3d = float(math.sqrt(range_2d**2 + up_err**2))
        heading = float(self.fdm.get_property_value("attitude/psi-deg"))
        brg = float(bearing_deg(lat, lon, wp_lat, wp_lon))
        berr_rad = math.radians(float(wrap180(brg - heading)))
        align = float(math.cos(berr_rad))
        m = self.mission

        # Navigation reward (potential shaping)
        phi_next = -float(range_2d)
        shaping = (1.0 * phi_next) - float(phi_prev)
        progress = shaping * float(m.progress_scale)
        reward = progress + m.align_weight * align + m.step_penalty

        # Altitude penalty
        dz = m.alt_hold_deadzone_m if m.alt_hold_deadzone_m > 0 else m.success_alt_m
        alt_penalty = -m.alt_hold_weight * max(0.0, abs(up_err) - dz)
        w_fps = float(self.fdm.get_property_value("velocities/w-fps"))
        vert_penalty = -m.vert_speed_weight * abs(w_fps)
        reward += alt_penalty + vert_penalty

        # === Threat penalty ===
        tf = self._threat_factor()
        tf_radar = 4.0  # Outer radar ring at roughly 2x r_horiz
        r_threat = 0.0
        killed_by_sam = False
        
        if alt_m <= m.radar_floor_m:
            tf = 999.0  # Terrain masked, completely safe from SAM
            tf_radar = 0.0

        if tf <= 1.0 and alt_m > 0:
            killed_by_sam = True
            reward = float(m.kill_penalty)
            done = True
            self._prev_phi = phi_next; self._prev_phi_wp_idx = int(wp_idx_prev)
            info = self._build_info(False, False, None, wp_idx_prev, range_2d, range_3d,
                                     float(wrap180(brg-heading)), align, up_err, w_fps,
                                     alt_penalty, vert_penalty, progress, tf, r_threat,
                                     killed_by_sam, False, 0.0)
            return float(reward), True, info
        elif tf < tf_radar:
            # Danger gradient: 0 penalty at tf=4.0, max penalty at tf=1.0
            factor = (tf_radar - tf) / (tf_radar - 1.0)
            r_threat = -m.threat_alpha * (factor ** 2)
            reward += r_threat

        # Goal reached?
        mission_success = False
        wp_reached = False; reached_idx = None
        if range_2d < m.success_radius_m and abs(up_err) < m.success_alt_m:
            wp_reached = True; reached_idx = int(wp_idx_prev)
            reward += m.goal_bonus
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                mission_success = True

        # Update phi
        if mission_success:
            self._prev_phi = 0.0; self._prev_phi_wp_idx = int(self._wp_idx)
        elif self._wp_idx != int(wp_idx_prev):
            r_new = self._range_2d_to_wp_idx(int(self._wp_idx))
            self._prev_phi = -float(r_new); self._prev_phi_wp_idx = int(self._wp_idx)
        else:
            self._prev_phi = phi_next; self._prev_phi_wp_idx = int(wp_idx_prev)

        done = bool(mission_success)
        sim_time = float(self.fdm.get_property_value("simulation/sim-time-sec"))
        crashed = False
        if alt_m < m.min_alt_m:
            done = True; crashed = True
            if not mission_success: reward += m.crash_penalty
        elif sim_time > m.max_sim_time_s:
            done = True
            if not mission_success: reward += m.fail_penalty

        info = self._build_info(mission_success, wp_reached, reached_idx, wp_idx_prev,
                                 range_2d, range_3d, float(wrap180(brg-heading)), align,
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
            "success_alt_m": float(self.mission.success_alt_m),
            "progress": float(progress),
            "range_2d_m": float(range_2d), "range_3d_m": float(range_3d),
            "bearing_err_deg": float(berr), "align": float(align),
            "up_err_m": float(up_err), "w_fps": float(w_fps),
            "alt_penalty": float(alt_pen), "vert_penalty": float(vert_pen),
            "fail_penalty": 0.0, "sim_time_s": float(sim_time),
            "threat_factor": float(dist_sam), "r_threat": float(r_threat),
            "killed_by_sam": bool(killed),
            "crashed": bool(crashed),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        aileron, elevator, rudder, throttle = action
        wp_idx_prev = min(int(self._wp_idx), max(len(self._waypoints)-1, 0))
        phi_prev = -float(self._range_2d_to_wp_idx(wp_idx_prev))
        self._prev_phi = phi_prev; self._prev_phi_wp_idx = int(wp_idx_prev)

        self.fdm.set_property_value("fcs/aileron-cmd-norm", float(np.clip(aileron, -1, 1)))
        self.fdm.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elevator, -1, 1)))
        self.fdm.set_property_value("fcs/rudder-cmd-norm", float(np.clip(rudder, -1, 1)))
        self.fdm.set_property_value("fcs/throttle-cmd-norm", float(np.clip(throttle, 0, 1)))
        for _ in range(self.agent_steps):
            self.fdm.run(); self._update_frame()
        reward, done, info = self._reward_done(phi_prev=phi_prev, wp_idx_prev=wp_idx_prev)
        self._update_frame()
        return self._get_observation(), reward, done, info


# ============================================================
# SAC Update
# ============================================================
def sac_update(replay_buffer, batch_size, actor, critic, target_critic,
               alpha_optimizer, actor_optimizer, critic_optimizer,
               log_alpha, target_entropy, gamma, max_throttle, tau=0.005,
               freeze_actor=False):
    states, actions, rewards, next_states, masks = replay_buffer.sample(batch_size)
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        mu_n, std_n = actor(next_states)
        next_pol, next_lp = eval_action(mu_n, std_n, multiplier=max_throttle)
        tq1, tq2 = target_critic(next_states, next_pol)
        min_tq = torch.min(tq1, tq2).squeeze(1) - alpha * next_lp.squeeze(1)
        target = rewards + masks * gamma * min_tq
    q1, q2 = critic(states, actions)
    c_loss = torch.nn.functional.mse_loss(q1.squeeze(1), target) + \
             torch.nn.functional.mse_loss(q2.squeeze(1), target)
    critic_optimizer.zero_grad(); c_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); critic_optimizer.step()

    if not freeze_actor:
        mu, std = actor(states)
        pol, lp = eval_action(mu, std, multiplier=max_throttle)
        q1p, q2p = critic(states, pol)
        a_loss = ((alpha * lp) - torch.min(q1p, q2p)).mean()
        actor_optimizer.zero_grad(); a_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0); actor_optimizer.step()
    else:
        # Actor frozen: forward only for alpha update, no backward through actor
        with torch.no_grad():
            mu, std = actor(states)
            pol, lp = eval_action(mu, std, multiplier=max_throttle)
        a_loss = torch.tensor(0.0)

    al_loss = -(log_alpha * (lp.detach() + target_entropy).detach()).mean()
    alpha_optimizer.zero_grad(); al_loss.backward(); alpha_optimizer.step()
    soft_target_update(critic, target_critic, tau=tau)
    return a_loss.item(), c_loss.item(), al_loss.item(), log_alpha.exp().item(), lp.mean().item()


# ============================================================
# Smart Weight Loading (Actor-Only Transfer)
# ============================================================
def smart_load_weights(model, ckpt_path, device, key=None):
    """Load weights from checkpoint with shape checking. Skips mismatched layers."""
    print(f"[SmartLoad] Loading weights from {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    if key and key in checkpoint:
        state_dict = checkpoint[key]
    elif not key and 'actor' in checkpoint:
        state_dict = checkpoint['actor']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    pretrained, skipped = {}, []
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained[k] = v
            else:
                skipped.append(f"{k} (ckpt {v.shape} vs model {model_dict[k].shape})")
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    print(f"[SmartLoad] Loaded {len(pretrained)} layers.")
    if skipped:
        print("[SmartLoad] SKIPPED layers (shape mismatch):")
        for s in skipped: print(f"  - {s}")
    else:
        print("[SmartLoad] All layers loaded successfully (Exact Match).")


# ============================================================
# Train
# ============================================================
def train(args):
    if args.seed is not None:
        seed_everything(int(args.seed))
    if not (0.0 <= float(args.max_throttle) <= 1.0):
        raise ValueError(f"--max_throttle must be in [0,1]. Got: {args.max_throttle}")

    mission = SAMEvasionConfig(
        goal_dist_min_m=args.goal_dist_min_m, goal_dist_max_m=args.goal_dist_max_m,
        goal_los_deg=args.goal_los_deg, goal_up_min_m=args.goal_up_min_m,
        goal_up_max_m=args.goal_up_max_m,
        sam_place_frac_min=args.sam_place_frac_min, sam_place_frac_max=args.sam_place_frac_max,
        sam_lateral_offset_m=args.sam_lateral_offset_m,
        r_horiz_m=args.r_horiz_m, r_vert_m=args.r_vert_m,
        radar_floor_m=args.radar_floor_m,
        threat_alpha=args.threat_alpha,
        success_radius_m=args.success_radius_m, success_alt_m=args.success_alt_m,
        progress_scale=args.progress_scale, align_weight=args.align_weight,
        step_penalty=args.step_penalty, goal_bonus=args.goal_bonus,
        kill_penalty=args.kill_penalty,
        alt_hold_deadzone_m=args.alt_hold_deadzone_m if args.alt_hold_deadzone_m else args.success_alt_m,
        alt_hold_weight=args.alt_hold_weight, vert_speed_weight=args.vert_speed_weight,
        min_alt_m=args.min_alt_m, max_sim_time_s=args.max_sim_time_s,
        fail_penalty=args.fail_penalty,
    )

    env = JSBSimF16SAMEvasionEnv(agent_steps=args.agent_steps, settle_steps=args.settle_steps,
                                  mission=mission, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base_run = args.run_name or f"{SCRIPT_TAG}_{ts}"
    run_name = base_run
    if args.run_name is None:
        k = 2
        while (LOG_DIR/run_name).exists() or (MODELS_DIR/run_name).exists() or (FIG_DIR/run_name).exists():
            run_name = f"{base_run}_{k:02d}"; k += 1
    model_name = args.model_name or run_name

    if SummaryWriter is None:
        print("[Warn] tensorboard not installed."); writer = _NullWriter()
    else:
        k = 0
        while True:
            tb_dir = LOG_DIR / f"{run_name}_{k:02d}"
            if not tb_dir.exists(): break
            k += 1
        tb_dir.mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard Logging to: {tb_dir}"); writer = SummaryWriter(log_dir=str(tb_dir))

    run_log_dir = LOG_DIR / run_name; run_log_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_log_dir / "episode_metrics.csv"; csv_exists = metrics_csv.exists()
    fig_dir = FIG_DIR / model_name; fig_dir.mkdir(parents=True, exist_ok=True)
    succ_fig = fig_dir / "success"; fail_fig = fig_dir / "failed"
    succ_fig.mkdir(parents=True, exist_ok=True); fail_fig.mkdir(parents=True, exist_ok=True)
    weight_dir = MODELS_DIR / model_name; weight_dir.mkdir(parents=True, exist_ok=True)

    dummy = env.reset(); state_shape = dummy.shape[1:]; state_size = state_shape[1]
    action_size = 4
    actor = Actor(state_size, action_size, multiplier=args.max_throttle).to(device)
    critic = Critic(state_size, action_size).to(device)
    target_critic = Critic(state_size, action_size).to(device)
    hard_target_update(critic, target_critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

    # === Transfer + Alpha Reset ===
    # transfer_mode: actor_only (default), full (actor+critic), none (scratch)
    transfer_mode = getattr(args, 'transfer_mode', 'actor_only')
    if getattr(args, "init_from", None) and transfer_mode != 'none':
        ckpt_path = Path(str(args.init_from))
        if not ckpt_path.exists():
            ckpt_path = PROJECT_ROOT / args.init_from
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--init_from not found: {args.init_from}")
        if transfer_mode == 'full':
            print(f"[Init] Full Transfer (Actor+Critic) from: {ckpt_path}")
            smart_load_weights(actor, ckpt_path, device, key='actor')
            smart_load_weights(critic, ckpt_path, device, key='critic')
            smart_load_weights(target_critic, ckpt_path, device, key='target_critic')
            print("[Init] Actor + Critic transferred.")
        else:  # actor_only
            print(f"[Init] Actor-Only Transfer from: {ckpt_path}")
            smart_load_weights(actor, ckpt_path, device, key='actor')
            print("[Init] Critic: random init (no transfer)")
        with torch.no_grad():
            log_alpha.fill_(0.0)
        print(f"[Init] Alpha reset to {log_alpha.exp().item():.2f} (max exploration)")
    else:
        print("[Init] No transfer. Training from scratch.")

    # === Actor Freeze Warmup ===
    actor_freeze_eps = getattr(args, 'actor_freeze_episodes', 0)
    freeze_actor = actor_freeze_eps > 0
    if freeze_actor:
        for p in actor.parameters():
            p.requires_grad_(False)
        print(f"[Freeze] Actor frozen for first {actor_freeze_eps} episodes (Critic-only warmup).")

    replay_buffer = ReplayBuffer(args.replay_buffer, state_shape, action_size, device)

    start_ep = int(getattr(args, "start_episode", 0))
    for episode in range(start_ep, start_ep + int(args.episodes)):
        local_ep = episode - start_ep

        # === Actor Unfreeze Check ===
        if freeze_actor and local_ep >= actor_freeze_eps:
            for p in actor.parameters():
                p.requires_grad_(True)
            freeze_actor = False
            print(f"[Freeze] Actor unfrozen at local_ep={local_ep}. Joint training begins.")
        prog = min(max(float(local_ep) / max(int(args.episodes)-1, 1), 0), 1)
        te_start, te_end = -float(action_size), -float(action_size)*2
        cur_te = torch.tensor([te_start + prog*(te_end - te_start)], dtype=torch.float32, device=device)

        state = env.reset(); done = False; score = 0.0; steps = 0
        sum_alt_pen = 0.0; sum_vert_pen = 0.0; sum_progress = 0.0
        min_range2d = float("inf"); min_tf = float("inf"); in_radar_steps = 0
        ep_killed = False; ep_crashed = False
        traj_lat, traj_lon, traj_alt, traj_spd = [], [], [], []
        traj_r2d, traj_berr, traj_wpidx, traj_uperr, traj_tf = [], [], [], [], []
        last_al, last_cl, last_all = None, None, None; tot_ent = 0.0; ent_cnt = 0

        lat0, lon0, alt0 = env.get_position()
        wps0 = env.get_waypoints(); sam0 = env.get_sam_position()
        traj_lat.append(lat0); traj_lon.append(lon0); traj_alt.append(alt0)
        traj_spd.append(fdm_speed_kts(env.fdm))
        wp_lat0, wp_lon0, wp_alt0 = env.get_waypoint()
        r2d0 = haversine((lat0,lon0),(wp_lat0,wp_lon0))*1000+1e-6
        h0 = float(env.fdm.get_property_value("attitude/psi-deg"))
        berr0 = float(wrap180(bearing_deg(lat0,lon0,wp_lat0,wp_lon0)-h0))
        traj_r2d.append(r2d0); traj_berr.append(berr0); traj_wpidx.append(0)
        traj_uperr.append(float(wp_alt0-alt0)); traj_tf.append(env._threat_factor())

        while not done and steps < args.max_steps:
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32, device=device)
                mu, std = actor(st)
                act = get_action(mu, std, multiplier=args.max_throttle)
                act_env = np.asarray(act, dtype=np.float32).flatten()
                act_env[:3] = np.clip(act_env[:3], -1, 1); act_env[3] = np.clip(act_env[3], 0, 1)
            next_state, reward, done, info = env.step(act_env)
            trunc = (not done) and ((steps+1) >= args.max_steps)
            if trunc:
                reward = float(reward) + float(mission.fail_penalty); done = True

            r2 = float(info.get("range_2d_m", 0)); ue = float(info.get("up_err_m", 0))
            min_range2d = min(min_range2d, r2)
            tf = float(info.get("threat_factor", 999.0)); min_tf = min(min_tf, tf)
            
            lat, lon, alt = env.get_position()
            if tf < 4.0 and alt > mission.radar_floor_m: in_radar_steps += 1
            
            if info.get("killed_by_sam"): ep_killed = True
            if info.get("crashed"): ep_crashed = True
            sum_alt_pen += float(info.get("alt_penalty", 0))
            sum_vert_pen += float(info.get("vert_penalty", 0))
            sum_progress += float(info.get("progress", 0))

            traj_lat.append(lat); traj_lon.append(lon); traj_alt.append(alt)
            traj_spd.append(fdm_speed_kts(env.fdm))
            traj_r2d.append(r2); traj_berr.append(float(info["bearing_err_deg"]))
            traj_wpidx.append(int(info["wp_idx"])); traj_uperr.append(ue)
            traj_tf.append(tf)

            replay_buffer.push(state.squeeze(0), act_env, float(reward), next_state.squeeze(0), bool(done))
            state = next_state; score += reward; steps += 1
            if len(replay_buffer) >= args.batch_size:
                al, cl, all_, alp, ent = sac_update(
                    replay_buffer, args.batch_size, actor, critic, target_critic,
                    alpha_opt, actor_opt, critic_opt, log_alpha, cur_te, args.gamma, args.max_throttle,
                    freeze_actor=freeze_actor)
                last_al, last_cl, last_all = al, cl, all_; tot_ent += ent; ent_cnt += 1

        msuc = bool(info.get("mission_success", False))

        # === Plotting with SAM Dome ===
        try:
            ref_lat, ref_lon, ref_alt = float(traj_lat[0]), float(traj_lon[0]), float(traj_alt[0])
            axy = [latlon_to_ne_m(ref_lat, ref_lon, la, lo) for la, lo in zip(traj_lat, traj_lon)]
            an = np.array([p[0] for p in axy], dtype=np.float32)
            ae = np.array([p[1] for p in axy], dtype=np.float32)
            az = np.array(traj_alt, dtype=np.float32) - ref_alt
            # Waypoint
            wn, we, wz = [], [], []
            for wl, wlo, wa in wps0:
                n, e = latlon_to_ne_m(ref_lat, ref_lon, wl, wlo); wn.append(n); we.append(e); wz.append(wa - ref_alt)
            wn, we, wz = np.array(wn), np.array(we), np.array(wz)
            # SAM local coords
            sn, se = latlon_to_ne_m(ref_lat, ref_lon, sam0[0], sam0[1])
            sz = sam0[2] - ref_alt

            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(3, 3, width_ratios=[1.4, 1.0, 0.75], hspace=0.35, wspace=0.25)
            ax3d = fig.add_subplot(gs[:, 0], projection="3d")
            ax_xy = fig.add_subplot(gs[0, 1]); ax_xz = fig.add_subplot(gs[1, 1])
            ax_spd = fig.add_subplot(gs[2, 1]); ax_leg = fig.add_subplot(gs[:, 2]); ax_leg.axis("off")
            r_s, h_s = float(mission.success_radius_m), float(mission.success_alt_m)
            r_horiz, r_vert = float(mission.r_horiz_m), float(mission.r_vert_m)

            # 3D trajectory
            pts = np.column_stack([ae, an, az]).astype(np.float32)
            if len(pts) >= 2:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                lc = Line3DCollection(segs, colors="C0", linewidths=2.0, alpha=0.85)
                ax3d.add_collection3d(lc)
            z_floor = float(min(az.min(), wz.min(), sz) - 800)
            if len(pts) >= 2:
                pg = pts.copy(); pg[:, 2] = z_floor
                sg = np.stack([pg[:-1], pg[1:]], axis=1)
                ax3d.add_collection3d(Line3DCollection(sg, colors="k", linewidths=0.8, alpha=0.1))
            ax3d.scatter(ae[0], an[0], az[0], s=80, color="C0", edgecolors="k", label="Start")
            ax3d.scatter(we[0], wn[0], wz[0], s=120, color="C2", edgecolors="k", label="Goal")
            add_cylinder_3d(ax3d, we[0], wn[0], wz[0], r_s, h_s, color="C2")
            
            # SAM Threat Dome (MRSAM Ellipsoid)
            ax3d.scatter(se, sn, sz, s=100, color="red", edgecolors="k", marker="^", label="SAM")
            # Drawing simple sphere for visualization, although true shape is Ellipsoid
            add_sphere_3d(ax3d, se, sn, sz, r_horiz*2, color="red", alpha=0.06)
            add_sphere_3d(ax3d, se, sn, sz, r_horiz, color="darkred", alpha=0.12)

            ax3d.set_title("3D Trajectory + Threat Dome")
            ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Up (m)")
            try: ax3d.set_proj_type("ortho")
            except: pass
            ax3d.view_init(elev=22, azim=-55)
            m = max(r_horiz*1.2, r_s*1.6, 1000)
            all_e = np.concatenate([ae, we, [se]]); all_n = np.concatenate([an, wn, [sn]])
            all_z = np.concatenate([az, wz, [sz]])
            ax3d.set_xlim(all_e.min()-m, all_e.max()+m)
            ax3d.set_ylim(all_n.min()-m, all_n.max()+m)
            ax3d.set_zlim(min(z_floor, all_z.min())-200, all_z.max()+500)
            try: ax3d.set_box_aspect((max(all_e.ptp(),1), max(all_n.ptp(),1), max(all_z.ptp(),1)))
            except: pass

            # XY
            ax_xy.plot(ae, an, color="C0", linewidth=2, alpha=0.85, label="Agent")
            ax_xy.scatter(ae[0], an[0], color="C0", edgecolors="k", s=40, label="Start")
            ax_xy.scatter(we[0], wn[0], color="C2", edgecolors="k", s=60, label="Goal")
            add_circle_xy(ax_xy, we[0], wn[0], r_s, color="C2")
            # SAM circles
            add_circle_xy(ax_xy, se, sn, r_horiz*2, color="red", alpha=0.25, lw=2)
            add_circle_xy(ax_xy, se, sn, r_horiz, color="darkred", alpha=0.35, lw=2)
            ax_xy.scatter(se, sn, color="red", marker="^", s=80, edgecolors="k", label="SAM")
            ax_xy.set_title("Top-Down (XY) + Threat Zones")
            ax_xy.set_aspect("equal", adjustable="box"); ax_xy.grid(True, ls="--", alpha=0.4)

            # XZ
            ax_xz.plot(ae, az, color="C0", linewidth=2, alpha=0.85)
            ax_xz.scatter(we[0], wz[0], color="C2", edgecolors="k", s=60)
            ax_xz.scatter(se, sz, color="red", marker="^", s=80, edgecolors="k")
            ax_xz.axvspan(se-r_horiz*2, se+r_horiz*2, color="red", alpha=0.05)
            ax_xz.axvspan(se-r_horiz, se+r_horiz, color="darkred", alpha=0.08)
            ax_xz.set_title("Side View (XZ)"); ax_xz.grid(True, ls="--", alpha=0.4)

            # Speed
            ax_spd.plot(traj_spd, color="C4", linewidth=2)
            spd_arr = np.array(traj_spd)
            ax_spd.set_title(f"Speed (min {spd_arr.min():.0f}, avg {spd_arr.mean():.0f}, max {spd_arr.max():.0f} kts)")
            ax_spd.grid(True, ls="--", alpha=0.4)

            # Result
            if ep_killed: res = "KILLED BY SAM"
            elif msuc: res = "Goal Reached"
            elif ep_crashed: res = "CRASHED (Min Alt)"
            else: res = "Failed"
            fig.suptitle(f"Episode {episode} - {res} | Reward: {score:.1f} | Steps: {steps}", y=0.98, fontsize=14)

            box_txt = (f"MRSAM Evasion Mission\nKill rH: {r_horiz/1000:.1f}km rV: {r_vert/1000:.1f}km\n"
                       f"Min TF: {min_tf:.2f}\nIn radar: {in_radar_steps} steps\n"
                       f"Range2D(last): {traj_r2d[-1]:.0f}m\nKilled: {ep_killed} | Crashed: {ep_crashed}")
            lm = {}
            for a in [ax3d, ax_spd]:
                for h, l in zip(*a.get_legend_handles_labels()):
                    if l and l not in lm: lm[l] = h
            lm.setdefault(f"Radar Floor ({mission.radar_floor_m:.0f}m)", Line2D([0],[0], color="orange", lw=2, alpha=0.5))
            lm.setdefault(f"Kill ({r_horiz/1000:.0f}km)", Line2D([0],[0], color="darkred", lw=2, alpha=0.7))
            ax_leg.text(0, 1, "Legend", transform=ax_leg.transAxes, va="top", fontsize=12, weight="bold")
            ax_leg.legend(list(lm.values()), list(lm.keys()), loc="upper left",
                          bbox_to_anchor=(0, 0.93), frameon=True, fontsize=10)
            ax_leg.text(0, 0, box_txt, transform=ax_leg.transAxes, fontsize=10, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="gray"))
            sd = succ_fig if msuc else fail_fig
            fig.savefig(sd / f"episode_{episode:05d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Plot error ep {episode}: {e}")

        # Logging
        writer.add_scalar("episode/score", score, episode)
        writer.add_scalar("episode/steps", steps, episode)
        writer.add_scalar("episode/success", 1 if msuc else 0, episode)
        writer.add_scalar("episode/killed_by_sam", 1 if ep_killed else 0, episode)
        writer.add_scalar("episode/min_tf", min_tf if math.isfinite(min_tf) else 999.0, episode)
        writer.add_scalar("episode/in_radar_steps", in_radar_steps, episode)
        writer.add_scalar("episode/range_2d_last", traj_r2d[-1], episode)
        denom = max(steps, 1)
        writer.add_scalar("episode/progress_mean", sum_progress/denom, episode)
        if last_al is not None:
            writer.add_scalar("loss/actor", last_al, episode)
            writer.add_scalar("episode/alpha", log_alpha.exp().item(), episode)
            writer.add_scalar("episode/entropy", tot_ent/ent_cnt if ent_cnt else 0, episode)
        if last_cl is not None: writer.add_scalar("loss/critic", last_cl, episode)

        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                     "target_critic": target_critic.state_dict(),
                     "log_alpha": log_alpha, "mission": mission.__dict__, "args": vars(args)},
                    weight_dir / f"epi_{episode:05d}.pth")

        alpha_v = float(log_alpha.exp().item()) if last_al else 1.0
        avg_ent = tot_ent/ent_cnt if ent_cnt else 0
        tag = " [SUCCESS]" if msuc else (" [KILLED]" if ep_killed else (" [CRASHED]" if ep_crashed else ""))
        
        # Consistent formatting with Return_Finetune
        print(
            "{} episode | score: {:.2f} | alpha: {:.3f} | entropy: {:.2f} | min_TF: {:.2f} | radar_steps: {} | range2d: {:.0f}m{}".format(
                episode, score, alpha_v, avg_ent, min_tf, in_radar_steps, traj_r2d[-1], tag
            )
        )

        # CSV
        try:
            with metrics_csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "episode","steps","score","success","killed_by_sam","crashed",
                    "min_tf","in_radar_steps","range2d_last_m","progress_mean"])
                if not csv_exists: w.writeheader(); csv_exists = True
                w.writerow({"episode": episode, "steps": steps, "score": float(score),
                            "success": int(msuc), "killed_by_sam": int(ep_killed),
                            "crashed": int(ep_crashed),
                            "min_tf": float(min_tf) if math.isfinite(min_tf) else 999.0,
                            "in_radar_steps": in_radar_steps,
                            "range2d_last_m": float(traj_r2d[-1]),
                            "progress_mean": float(sum_progress/denom)})
        except Exception as e:
            print(f"CSV error: {e}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--agent_steps", type=int, default=5)
    parser.add_argument("--settle_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_buffer", type=int, default=50000)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_throttle", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    # Mission
    parser.add_argument("--goal_dist_min_m", type=float, default=30000.0)
    parser.add_argument("--goal_dist_max_m", type=float, default=40000.0)
    parser.add_argument("--goal_los_deg", type=float, default=10.0)
    parser.add_argument("--goal_up_min_m", type=float, default=-300.0)
    parser.add_argument("--goal_up_max_m", type=float, default=300.0)
    # SAM
    parser.add_argument("--sam_place_frac_min", type=float, default=0.4)
    parser.add_argument("--sam_place_frac_max", type=float, default=0.6)
    parser.add_argument("--sam_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--r_horiz_m", type=float, default=10000.0)
    parser.add_argument("--r_vert_m", type=float, default=5000.0)
    parser.add_argument("--radar_floor_m", type=float, default=0.0)
    parser.add_argument("--threat_alpha", type=float, default=0.1)
    # Reward
    parser.add_argument("--success_radius_m", type=float, default=1500.0)
    parser.add_argument("--success_alt_m", type=float, default=1000.0)
    parser.add_argument("--progress_scale", type=float, default=0.001)
    parser.add_argument("--align_weight", type=float, default=0.05)
    parser.add_argument("--step_penalty", type=float, default=-0.01)
    parser.add_argument("--goal_bonus", type=float, default=1000.0)
    parser.add_argument("--kill_penalty", type=float, default=-500.0)
    parser.add_argument("--alt_hold_deadzone_m", type=float, default=None)
    parser.add_argument("--alt_hold_weight", type=float, default=2e-5)
    parser.add_argument("--vert_speed_weight", type=float, default=5e-5)
    parser.add_argument("--min_alt_m", type=float, default=150.0)
    parser.add_argument("--max_sim_time_s", type=float, default=400.0)
    parser.add_argument("--fail_penalty", type=float, default=0.0)
    parser.add_argument("--crash_penalty", type=float, default=-500.0)
    # Run
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--init_from", type=str,
                        default="models/20260109_SAC_Self_Play/selfplay_epi_00900.pth",
                        help="Path to pretrained model checkpoint for transfer")
    # Ablation args
    parser.add_argument("--transfer_mode", type=str, default="actor_only",
                        choices=["actor_only", "full", "none"],
                        help="actor_only=Actor only, full=Actor+Critic, none=scratch")
    parser.add_argument("--actor_freeze_episodes", type=int, default=0,
                        help="Freeze actor for first N episodes (Critic-only warmup). 0=disabled")
    args = parser.parse_args()
    train(args)
