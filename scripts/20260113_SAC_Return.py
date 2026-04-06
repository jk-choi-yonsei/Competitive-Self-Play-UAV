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

# TensorBoard is optional: allow running even if `tensorboard` package is missing.
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:  # noqa: D401
        return None

    def close(self) -> None:
        return None

# Optional dependency: haversine (km). Provide a safe local fallback when not installed.
try:
    from haversine import haversine as haversine  # type: ignore
except Exception:
    def haversine(p1, p2) -> float:
        """Fallback haversine distance in kilometers between (lat, lon) pairs (degrees)."""
        lat1, lon1 = p1
        lat2, lon2 = p2
        r = 6371.0088  # mean Earth radius (km)
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlmb = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
        c = 2.0 * math.asin(math.sqrt(a))
        return float(r * c)

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "runs" / "20260119_SAC_Return"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from sac_agent.models import Actor, Critic  # noqa: E402
from sac_agent.replay_buffer import ReplayBuffer  # noqa: E402
from sac_agent.utils import eval_action, get_action, hard_target_update, soft_target_update  # noqa: E402
from envs.init_noise import meters_to_latlon_deg, uniform_symmetric  # noqa: E402
from envs.target_controllers import bearing_deg, wrap180  # noqa: E402


KTS_PER_FPS = 1.0 / 1.6878098571011957  # 1 knot = 1.68781 ft/s


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fdm_speed_kts(fdm) -> float:
    """Compute speed magnitude from JSBSim body velocities (ft/s) -> knots."""
    u = float(fdm.get_property_value("velocities/u-fps"))
    v = float(fdm.get_property_value("velocities/v-fps"))
    w = float(fdm.get_property_value("velocities/w-fps"))
    speed_fps = math.sqrt(u * u + v * v + w * w)
    return float(speed_fps * KTS_PER_FPS)


def latlon_to_ne_m(ref_lat: float, ref_lon: float, lat: float, lon: float) -> tuple[float, float]:
    """Local North/East meters relative to (ref_lat, ref_lon)."""
    north_m = haversine((ref_lat, ref_lon), (lat, ref_lon)) * 1000.0
    east_m = haversine((ref_lat, ref_lon), (ref_lat, lon)) * 1000.0
    if lat < ref_lat:
        north_m *= -1.0
    if lon < ref_lon:
        east_m *= -1.0
    return float(north_m), float(east_m)


def add_circle_xy(ax, cx: float, cy: float, r: float, *, color: str, alpha: float = 0.18, lw: float = 1.5) -> None:
    """Draw a 2D circle (XY plane) for success radius visualization."""
    theta = np.linspace(0.0, 2.0 * math.pi, 180, dtype=np.float32)
    x = float(cx) + float(r) * np.cos(theta)
    y = float(cy) + float(r) * np.sin(theta)
    ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)


def add_sphere_3d(
    ax,
    cx: float,
    cy: float,
    cz: float,
    r: float,
    *,
    color: str,
    alpha: float = 0.12,
    lw: float = 0.6,
    n_u: int = 20,
    n_v: int = 12,
) -> None:
    """Draw a wireframe sphere for 3D success radius visualization."""
    u = np.linspace(0.0, 2.0 * math.pi, n_u, dtype=np.float32)
    v = np.linspace(0.0, math.pi, n_v, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x = float(cx) + float(r) * np.cos(uu) * np.sin(vv)
    y = float(cy) + float(r) * np.sin(uu) * np.sin(vv)
    z = float(cz) + float(r) * np.cos(vv)
    ax.plot_wireframe(x, y, z, color=color, linewidth=lw, alpha=alpha)


def add_cylinder_3d(
    ax,
    cx: float,
    cy: float,
    cz: float,
    r: float,
    half_h: float,
    *,
    color: str,
    alpha: float = 0.24,
    lw: float = 1.0,
    n_theta: int = 40,
) -> None:
    """Draw a wireframe cylinder to visualize a (2D radius + altitude band) success gate."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, dtype=np.float32)
    x = float(cx) + float(r) * np.cos(theta)
    y = float(cy) + float(r) * np.sin(theta)
    z0 = float(cz) - float(half_h)
    z1 = float(cz) + float(half_h)
    # top/bottom rings
    ax.plot(x, y, np.full_like(x, z0), color=color, linewidth=lw, alpha=alpha)
    ax.plot(x, y, np.full_like(x, z1), color=color, linewidth=lw, alpha=alpha)
    # a few vertical lines
    for k in range(0, len(theta), max(len(theta) // 8, 1)):
        ax.plot([x[k], x[k]], [y[k], y[k]], [z0, z1], color=color, linewidth=lw, alpha=alpha)


@dataclass
class NavMissionConfig:
    """
    4-waypoint navigation mission (Return-to-Start).

    Each waypoint is generated at reset time using:
    - distance range [dist_min_m, dist_max_m]
    - LOS angle (deg) relative to initial agent heading (left/right randomly)

    WP1~WP3 are sampled as usual, and WP4 is fixed to the agent's initial spawn position (home).
    Success is defined as reaching each waypoint in order within success_radius_m (2D ground range) and
    within the altitude band success_alt_m.
    """

    # Waypoint generation spec: (los_deg, dist_min_m, dist_max_m)
    # Waypoint generation spec: (los_deg, dist_min_m, dist_max_m)
    wp1_los_deg: float = 30.0
    wp1_dist_min_m: float = 3000.0
    wp1_dist_max_m: float = 4000.0

    wp2_los_deg: float = 30.0
    wp2_dist_min_m: float = 6000.0
    wp2_dist_max_m: float = 7000.0

    wp3_los_deg: float = 30.0
    wp3_dist_min_m: float = 9000.0
    wp3_dist_max_m: float = 10000.0

    # Altitude of waypoint relative to spawn (meters)
    # Sample uniformly for each WP: up_m ~ U(wp_up_min_m, wp_up_max_m)
    wp_up_min_m: float = -500.0
    wp_up_max_m: float = 500.0

    # Success condition (per waypoint): 2D ground range
    success_radius_m: float = 1500.0
    # Altitude band for waypoint success (meters). Success requires |up_err| < success_alt_m.
    success_alt_m: float = 1000.0

    # Terminal failure penalty applied when an episode ends without mission success.
    # Set to 0.0 to disable (no extra terminal penalty).
    # If you want to strongly discourage failures, use a negative value (e.g., -3000, -6000).
    fail_penalty: float = 0.0

    # --- Altitude / safety shaping (small weights; should NOT dominate progress) ---
    # Deadzone for altitude error penalty (meters). Within this band, altitude penalty is zero.
    # Recommended to match success_alt_m.
    alt_hold_deadzone_m: float = 1000.0
    # Penalty weight for excess altitude error beyond deadzone:
    #   alt_penalty = -alt_hold_weight * max(0, |up_err_m| - alt_hold_deadzone_m)
    # Keep this much smaller than typical progress reward.
    alt_hold_weight: float = 2.0e-5
    # Penalty on vertical speed magnitude (ft/s). Helps avoid extreme dives/climbs:
    #   vert_speed_penalty = -vert_speed_weight * |w_fps|
    vert_speed_weight: float = 5.0e-5

    # Reward shaping weights
    # Potential shaping uses: F(s,a,s') = gamma_pot * Phi(s') - Phi(s), with Phi(s) = -range_2d_m.
    # IMPORTANT:
    # - If Phi(s) is NEGATIVE (e.g., -range) and you use gamma_pot < 1, then even with NO progress
    #   (range stays constant) you get a positive drift per step:
    #       F = gamma_pot*(-r) - (-r) = (1-gamma_pot)*r  > 0
    #   This is exactly how "failed but huge reward" trajectories happen (loiter far away for many steps).
    # - Therefore we fix gamma_pot = 1.0 here so that "no progress" => F = 0.
    # SAC's discount factor for return estimation remains args.gamma (separate from shaping).
    shaping_gamma: float = 1.0
    progress_scale: float = 1.0 / 100.0  # meters -> reward (applied to potential shaping term)
    align_weight: float = 0.05          # * cos(bearing_err)
    step_penalty: float = -0.01
    # Defaults for next experiment (user request)
    wp_bonus: float = 100.0
    mission_bonus: float = 800.0

    # Termination safety
    min_alt_m: float = 2000.0
    max_sim_time_s: float = 1500.0


@dataclass
class Waypoint:
    lat_deg: float
    lon_deg: float
    alt_m: float
    los_req_deg: float
    dist_m: float


class JSBSimF16NavigationEnv:
    """
    Single-aircraft navigation environment.

    Observation:
    - agent kinematics (11)
    - relative geo to waypoint (10)  # includes wp_onehot_1..4
    => 21 features, stacked over 5 frames -> (1, 5, 21)
    """

    def __init__(
        self,
        *,
        agent_steps: int = 5,
        settle_steps: int = 5,
        mission: Optional[NavMissionConfig] = None,
        seed: Optional[int] = None,
    ):
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        self.fdm.set_dt(1.0 / 50.0)

        self.model = "f16"
        if not self.fdm.load_model(self.model):
            raise RuntimeError("Failed to load JSBSim model f16 (agent)")

        self.initial_conditions = {
            "ic/h-sl-ft": 24000,
            "ic/u-fps": 360,
            "ic/v-fps": 0,
            "ic/w-fps": 0,
            "ic/long-gc-deg": 2.3,
            "ic/lat-gc-deg": 2.3,
            "ic/terrain-elevation-ft": 10,
            "ic/psi-true-deg": 90,
            "ic/roc-fpm": 0,
        }

        self.agent_steps = int(agent_steps)
        self.settle_steps = int(settle_steps)
        self.mission = mission or NavMissionConfig()

        self._rng = np.random.default_rng(seed)
        self._waypoints: List[Waypoint] = []
        self._wp_idx: int = 0
        # For potential shaping we store Phi(s) at the last control step, and the WP index it refers to.
        self._prev_phi: float = 0.0
        self._prev_phi_wp_idx: int = 0
        self._frame_stack: Deque[np.ndarray] = deque(maxlen=5)

        self._apply_initial_conditions()

    def _apply_initial_conditions(self) -> None:
        for prop, val in self.initial_conditions.items():
            self.fdm.set_property_value(prop, val)

        self.fdm.reset_to_initial_conditions(0)
        self.fdm.set_property_value("propulsion/starter_cmd", 1)
        self.fdm.set_property_value("propulsion/engine/set-running", 1)
        self.fdm.set_property_value("fcs/throttle-cmd-norm", 1.0)
        self.fdm.set_property_value("gear/gear-cmd-norm", 0)

        # Settle simulation
        for _ in range(self.settle_steps):
            self.fdm.run()

        self._sample_waypoints()
        self._wp_idx = 0
        r0 = self._range_2d_to_wp_idx(0)
        self._prev_phi = -float(r0)
        self._prev_phi_wp_idx = 0

        # Initialize frame stack WITHOUT advancing extra simulation in _stacked_observation.
        self._frame_stack.clear()
        for _ in range(5):
            self.fdm.run()
            self._update_frame()

    def _sample_waypoints(self) -> None:
        lat0 = float(self.fdm.get_property_value("position/lat-gc-deg"))
        lon0 = float(self.fdm.get_property_value("position/long-gc-deg"))
        alt0 = float(self.fdm.get_property_value("position/h-sl-meters"))
        heading0 = float(self.fdm.get_property_value("attitude/psi-deg"))

        specs = [
            (float(self.mission.wp1_los_deg), float(self.mission.wp1_dist_min_m), float(self.mission.wp1_dist_max_m)),
            (float(self.mission.wp2_los_deg), float(self.mission.wp2_dist_min_m), float(self.mission.wp2_dist_max_m)),
            (float(self.mission.wp3_los_deg), float(self.mission.wp3_dist_min_m), float(self.mission.wp3_dist_max_m)),
        ]

        self._waypoints = []
        for los_deg, dmin, dmax in specs:
            dist_m = float(self._rng.uniform(dmin, dmax))
            # Sample bearing within the LOS cone (±los_deg) relative to initial heading.
            # This matches "LOS 15deg / 30deg / 45deg" as a bound, not a fixed edge.
            offset = float(self._rng.uniform(-float(los_deg), float(los_deg)))
            bearing = (heading0 + offset) % 360.0  # 0=N, 90=E

            br = math.radians(bearing)
            north_m = dist_m * math.cos(br)
            east_m = dist_m * math.sin(br)

            # Sample waypoint altitude offset (Up) uniformly relative to spawn altitude.
            up_min = float(self.mission.wp_up_min_m)
            up_max = float(self.mission.wp_up_max_m)
            if up_min > up_max:
                up_min, up_max = up_max, up_min
            up_m = float(self._rng.uniform(up_min, up_max))

            dlat_deg, dlon_deg = meters_to_latlon_deg(north_m, east_m, ref_lat_deg=lat0)
            self._waypoints.append(
                Waypoint(
                    lat_deg=float(lat0 + dlat_deg),
                    lon_deg=float(lon0 + dlon_deg),
                    alt_m=float(alt0 + up_m),
                    # Store the allowed LOS bound for this WP (used for observation/debug).
                    los_req_deg=float(abs(los_deg)),
                    dist_m=float(dist_m),
                )
            )

        # WP4: Return-to-Start (home). Fixed to spawn position (lat0, lon0, alt0).
        # This lets you test if the agent can come back to the origin after visiting WP1~WP3.
        self._waypoints.append(
            Waypoint(
                lat_deg=float(lat0),
                lon_deg=float(lon0),
                alt_m=float(alt0),
                los_req_deg=180.0,
                dist_m=0.0,
            )
        )

    def reset(self):
        self._apply_initial_conditions()
        return self._get_observation()

    def get_position(self) -> tuple[float, float, float]:
        lat = float(self.fdm.get_property_value("position/lat-gc-deg"))
        lon = float(self.fdm.get_property_value("position/long-gc-deg"))
        alt_m = float(self.fdm.get_property_value("position/h-sl-meters"))
        return lat, lon, alt_m

    def get_waypoint(self) -> tuple[float, float, float]:
        """Return current (active) waypoint."""
        # When the mission is complete, _wp_idx may become len(_waypoints).
        # Clamp for safe observation/plotting.
        idx = min(int(self._wp_idx), max(len(self._waypoints) - 1, 0))
        wp = self._waypoints[idx]
        return float(wp.lat_deg), float(wp.lon_deg), float(wp.alt_m)

    def get_waypoints(self) -> List[tuple[float, float, float]]:
        """Return all waypoints (WP1~WP4) for plotting/debug."""
        return [(float(w.lat_deg), float(w.lon_deg), float(w.alt_m)) for w in self._waypoints]

    def get_wp_index(self) -> int:
        return int(self._wp_idx)

    def _current_range_2d_m(self) -> float:
        return self._range_2d_to_wp_idx(int(self._wp_idx))

    def _range_2d_to_wp_idx(self, wp_idx: int) -> float:
        """2D ground range to waypoint index (safe clamped)."""
        if len(self._waypoints) == 0:
            return 0.0
        idx = min(max(int(wp_idx), 0), len(self._waypoints) - 1)
        lat, lon, _ = self.get_position()
        wp = self._waypoints[idx]
        return float((haversine((lat, lon), (wp.lat_deg, wp.lon_deg)) * 1000.0) + 1e-6)

    def _get_state(self) -> np.ndarray:
        """Agent kinematics (11), same normalization style as chase env."""
        alt_m = float(self.fdm.get_property_value("position/h-sl-meters"))
        state_vals = [
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
        ]
        return np.array(state_vals, dtype=np.float32)

    def _positional_geo_raw(self) -> np.ndarray:
        """
        Navigation-friendly relative features to CURRENT waypoint (raw).

        10 features:
        0 forward_err_m (wp is ahead of agent => +)
        1 left_err_m    (wp is left of agent  => +)
        2 up_err_m      (wp above agent       => +)
        3 range_2d_m
        4 sin(bearing_err)
        5 cos(bearing_err)
        6 wp_onehot_1 (0/1)
        7 wp_onehot_2 (0/1)
        8 wp_onehot_3 (0/1)
        9 wp_onehot_4 (0/1)  # home/return waypoint
        """
        lat, lon, alt_m = self.get_position()
        wp_lat, wp_lon, wp_alt = self.get_waypoint()

        north_err_m, east_err_m = latlon_to_ne_m(ref_lat=lat, ref_lon=lon, lat=wp_lat, lon=wp_lon)
        up_err_m = float(wp_alt - alt_m)
        range_2d_m = float((haversine((lat, lon), (wp_lat, wp_lon)) * 1000.0) + 1e-6)

        heading_deg = float(self.fdm.get_property_value("attitude/psi-deg"))
        brg_to_wp = float(bearing_deg(lat, lon, wp_lat, wp_lon))
        bearing_err_deg = float(wrap180(brg_to_wp - heading_deg))
        bearing_err_rad = math.radians(bearing_err_deg)

        sin_err = float(math.sin(bearing_err_rad))
        cos_err = float(math.cos(bearing_err_rad))

        # Body-frame projection using heading only (good trade-off: simple + effective).
        psi = math.radians(heading_deg)
        c, s = math.cos(psi), math.sin(psi)
        forward_err_m = float(c * north_err_m + s * east_err_m)
        left_err_m = float(-s * north_err_m + c * east_err_m)

        # One-hot for active waypoint (clamp when mission complete).
        active_idx = min(int(self._wp_idx), 3)
        wp_onehot = np.zeros(4, dtype=np.float32)
        wp_onehot[active_idx] = 1.0

        return np.array(
            [
                forward_err_m,
                left_err_m,
                up_err_m,
                range_2d_m,
                sin_err,
                cos_err,
                float(wp_onehot[0]),
                float(wp_onehot[1]),
                float(wp_onehot[2]),
                float(wp_onehot[3]),
            ],
            dtype=np.float32,
        )

    def _positional_geo(self) -> np.ndarray:
        raw = self._positional_geo_raw()
        return np.array(
            [
                raw[0] / 10000.0,                # forward_err_m
                raw[1] / 10000.0,                # left_err_m
                raw[2] / 5000.0,                 # up_err_m
                raw[3] / 10000.0,                # range_2d_m
                raw[4],                          # sin(err)
                raw[5],                          # cos(err)
                raw[6] * 2.0 - 1.0,              # wp_onehot_1 -> [-1, 1]
                raw[7] * 2.0 - 1.0,              # wp_onehot_2 -> [-1, 1]
                raw[8] * 2.0 - 1.0,              # wp_onehot_3 -> [-1, 1]
                raw[9] * 2.0 - 1.0,              # wp_onehot_4 -> [-1, 1]
            ],
            dtype=np.float32,
        )

    def _update_frame(self) -> None:
        frame = np.hstack([self._get_state(), self._positional_geo()]).astype(np.float32, copy=False)
        self._frame_stack.append(frame)

    def _get_observation(self) -> np.ndarray:
        # If stack not full (shouldn't happen), pad with last frame.
        if len(self._frame_stack) == 0:
            self._update_frame()
        while len(self._frame_stack) < 5:
            self._frame_stack.append(self._frame_stack[-1].copy())
        obs = np.expand_dims(np.array(list(self._frame_stack), dtype=np.float32), axis=0)
        return np.reshape(obs, (1, 5, -1))

    def _reward_done(self, *, phi_prev: float, wp_idx_prev: int) -> tuple[float, bool, dict]:
        """Navigation reward and termination."""
        lat, lon, alt_m = self.get_position()
        wp_idx_prev = min(max(int(wp_idx_prev), 0), max(len(self._waypoints) - 1, 0))
        wp_prev = self._waypoints[wp_idx_prev]
        wp_lat, wp_lon, wp_alt = float(wp_prev.lat_deg), float(wp_prev.lon_deg), float(wp_prev.alt_m)

        range_2d_m = float((haversine((lat, lon), (wp_lat, wp_lon)) * 1000.0) + 1e-6)
        up_err_m = float(wp_alt - alt_m)
        range_3d_m = float(math.sqrt(range_2d_m * range_2d_m + up_err_m * up_err_m))

        heading_deg = float(self.fdm.get_property_value("attitude/psi-deg"))
        brg_to_wp = float(bearing_deg(lat, lon, wp_lat, wp_lon))
        bearing_err_deg = float(wrap180(brg_to_wp - heading_deg))
        bearing_err_rad = math.radians(bearing_err_deg)
        align = float(math.cos(bearing_err_rad))  # 1: perfectly aligned

        # Potential shaping (clean, consistent with WP transitions):
        # Phi(s) = -range_2d_m_to_active_wp
        # F = gamma_pot * Phi(s') - Phi(s)
        # NOTE: gamma_pot MUST be 1.0 for Phi(s)=-range to avoid positive drift at constant range.
        phi_next = -float(range_2d_m)
        gamma_pot = 1.0
        shaping = (float(gamma_pot) * float(phi_next)) - float(phi_prev)
        progress = float(shaping) * float(self.mission.progress_scale)

        reward = (
            float(progress)
            + float(self.mission.align_weight) * float(align)
            + float(self.mission.step_penalty)
        )

        # Altitude hold shaping (deadzone + small weight)
        dz = float(self.mission.alt_hold_deadzone_m)
        dz = dz if dz > 0 else float(self.mission.success_alt_m)
        alt_excess = max(0.0, abs(float(up_err_m)) - dz)
        alt_penalty = -float(self.mission.alt_hold_weight) * float(alt_excess)

        # Vertical speed shaping (discourage extreme dives/climbs; small weight)
        w_fps = float(self.fdm.get_property_value("velocities/w-fps"))
        vert_penalty = -float(self.mission.vert_speed_weight) * abs(w_fps)

        reward = float(reward) + float(alt_penalty) + float(vert_penalty)

        wp_reached = False
        reached_wp_idx = None
        mission_success = False

        # Waypoint reached (sequential)
        if (range_2d_m < float(self.mission.success_radius_m)) and (abs(up_err_m) < float(self.mission.success_alt_m)):
            wp_reached = True
            reached_wp_idx = int(wp_idx_prev)
            reward += float(self.mission.wp_bonus)
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                mission_success = True
                reward += float(self.mission.mission_bonus)

        # Update Phi for next step (this is the "strict timing" part):
        # - We computed shaping using the WP BEFORE transition (wp_idx_prev).
        # - If the WP advanced, we set Phi to the distance-to-NEW WP at the current position,
        #   so next step's shaping is consistent and we don't accidentally mix goals.
        if mission_success:
            self._prev_phi = 0.0
            self._prev_phi_wp_idx = int(self._wp_idx)
        elif self._wp_idx != int(wp_idx_prev):
            r_new = self._range_2d_to_wp_idx(int(self._wp_idx))
            self._prev_phi = -float(r_new)
            self._prev_phi_wp_idx = int(self._wp_idx)
        else:
            self._prev_phi = float(phi_next)
            self._prev_phi_wp_idx = int(wp_idx_prev)

        done = bool(mission_success)

        sim_time = float(self.fdm.get_property_value("simulation/sim-time-sec"))
        if (alt_m < float(self.mission.min_alt_m)) or (sim_time > float(self.mission.max_sim_time_s)):
            done = True
            # Large terminal penalty on failure to discourage "long high-reward failures".
            if not mission_success:
                reward += float(self.mission.fail_penalty)

        info = {
            "success": bool(done and mission_success),
            "mission_success": bool(mission_success),
            "wp_idx": int(self._wp_idx),
            "wp_reached": bool(wp_reached),
            "reached_wp_idx": reached_wp_idx,
            "wp_idx_for_reward": int(wp_idx_prev),
            "success_radius_m": float(self.mission.success_radius_m),
            "success_alt_m": float(self.mission.success_alt_m),
            "progress": float(progress),
            "current_wp_los_deg": float(self._waypoints[min(self._wp_idx, len(self._waypoints) - 1)].los_req_deg)
            if len(self._waypoints) > 0
            else 0.0,
            "current_wp_dist_m": float(self._waypoints[min(self._wp_idx, len(self._waypoints) - 1)].dist_m)
            if len(self._waypoints) > 0
            else 0.0,
            "range_2d_m": float(range_2d_m),
            "range_3d_m": float(range_3d_m),
            "bearing_err_deg": float(bearing_err_deg),
            "align": float(align),
            "up_err_m": float(up_err_m),
            "w_fps": float(w_fps),
            "alt_penalty": float(alt_penalty),
            "vert_penalty": float(vert_penalty),
            "fail_penalty": float(self.mission.fail_penalty) if (done and (not mission_success)) else 0.0,
            "sim_time_s": float(sim_time),
        }
        return float(reward), bool(done), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).flatten()
        aileron, elevator, rudder, throttle = action

        # Capture Phi(s) and the WP index it refers to BEFORE applying the action.
        wp_idx_prev = min(int(self._wp_idx), max(len(self._waypoints) - 1, 0))
        phi_prev = -float(self._range_2d_to_wp_idx(wp_idx_prev))
        self._prev_phi = float(phi_prev)
        self._prev_phi_wp_idx = int(wp_idx_prev)

        self.fdm.set_property_value("fcs/aileron-cmd-norm", float(np.clip(aileron, -1, 1)))
        self.fdm.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elevator, -1, 1)))
        self.fdm.set_property_value("fcs/rudder-cmd-norm", float(np.clip(rudder, -1, 1)))
        self.fdm.set_property_value("fcs/throttle-cmd-norm", float(np.clip(throttle, 0, 1)))

        for _ in range(self.agent_steps):
            self.fdm.run()
            self._update_frame()

        reward, done, info = self._reward_done(phi_prev=phi_prev, wp_idx_prev=wp_idx_prev)
        # Ensure the last frame reflects the current active waypoint (wp_idx may have advanced).
        self._update_frame()
        obs = self._get_observation()
        return obs, reward, done, info


def sac_update(
    replay_buffer,
    batch_size,
    actor,
    critic,
    target_critic,
    alpha_optimizer,
    actor_optimizer,
    critic_optimizer,
    log_alpha,
    target_entropy,
    gamma,
    max_throttle,
    tau=0.005,
):
    states, actions, rewards, next_states, masks = replay_buffer.sample(batch_size)
    alpha = log_alpha.exp().detach()

    with torch.no_grad():
        mu_next, std_next = actor(next_states)
        next_policy, next_log_policy = eval_action(mu_next, std_next, multiplier=max_throttle)
        target_q1, target_q2 = target_critic(next_states, next_policy)
        min_target_q = torch.min(target_q1, target_q2).squeeze(1) - alpha * next_log_policy.squeeze(1)
        target = rewards + masks * gamma * min_target_q

    q1, q2 = critic(states, actions)
    critic_loss = torch.nn.functional.mse_loss(q1.squeeze(1), target) + torch.nn.functional.mse_loss(
        q2.squeeze(1), target
    )
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    critic_optimizer.step()

    mu, std = actor(states)
    policy, log_policy = eval_action(mu, std, multiplier=max_throttle)
    q1_pi, q2_pi = critic(states, policy)
    min_q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = ((alpha * log_policy) - min_q_pi).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_policy + target_entropy).detach()).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    soft_target_update(critic, target_critic, tau=tau)

    return (
        actor_loss.item(),
        critic_loss.item(),
        alpha_loss.item(),
        log_alpha.exp().item(),
        log_policy.mean().item(),
    )


def smart_load_weights(model, ckpt_path, device, key=None):
    """
    Load weights from checkpoint with shape checking.
    Skips layers that don't match (e.g., input conv1 due to 20 vs 21 dim).
    """
    print(f"[SmartLoad] Loading weights from {{ckpt_path}}...")
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
    pretrained_dict = {}
    skipped_layers = []

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                skipped_layers.append(f"{{k}} (mismatch: ckpt {{v.shape}} vs model {{model_dict[k].shape}})")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f"[SmartLoad] Loaded {{len(pretrained_dict)}} layers.")
    if skipped_layers:
        print(f"[SmartLoad] SKIPPED layers due to shape mismatch:")
        for s in skipped_layers:
            print(f"  - {{s}}")
    else:
        print("[SmartLoad] All layers loaded successfully (Exact Match).")


def train(args) -> None:
    if args.seed is not None:
        seed_everything(int(args.seed))

    # Treat max_throttle as a hard limit on throttle command (0..1).
    if not (0.0 <= float(args.max_throttle) <= 1.0):
        raise ValueError(f"--max_throttle must be in [0,1]. Got: {args.max_throttle}")

    alt_hold_deadzone_m = args.alt_hold_deadzone_m
    if alt_hold_deadzone_m is None:
        alt_hold_deadzone_m = float(args.success_alt_m)

    mission = NavMissionConfig(
        wp1_los_deg=args.wp1_los_deg,
        wp1_dist_min_m=args.wp1_dist_min_m,
        wp1_dist_max_m=args.wp1_dist_max_m,
        wp2_los_deg=args.wp2_los_deg,
        wp2_dist_min_m=args.wp2_dist_min_m,
        wp2_dist_max_m=args.wp2_dist_max_m,
        wp3_los_deg=args.wp3_los_deg,
        wp3_dist_min_m=args.wp3_dist_min_m,
        wp3_dist_max_m=args.wp3_dist_max_m,
        wp_up_min_m=args.wp_up_min_m,
        wp_up_max_m=args.wp_up_max_m,
        success_radius_m=args.success_radius_m,
        success_alt_m=args.success_alt_m,
        fail_penalty=args.fail_penalty,
        alt_hold_deadzone_m=alt_hold_deadzone_m,
        alt_hold_weight=args.alt_hold_weight,
        vert_speed_weight=args.vert_speed_weight,
        progress_scale=args.progress_scale,
        align_weight=args.align_weight,
        step_penalty=args.step_penalty,
        wp_bonus=args.wp_bonus,
        mission_bonus=args.mission_bonus,
        min_alt_m=args.min_alt_m,
        max_sim_time_s=args.max_sim_time_s,
    )

    env = JSBSimF16NavigationEnv(
        agent_steps=args.agent_steps,
        settle_steps=args.settle_steps,
        mission=mission,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_tag = Path(__file__).stem
    # Folder naming: keep timestamp only up to minutes (YYYYMMDD_HHMM), per user request.
    # To avoid overwrites when re-running within the same minute, we add a small numeric suffix if needed.
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base_run_name = args.run_name or f"{script_tag}_{ts}"
    run_name = base_run_name
    if args.run_name is None:
        k = 2
        while (LOG_DIR / run_name).exists() or (MODELS_DIR / run_name).exists() or (FIG_DIR / run_name).exists():
            run_name = f"{base_run_name}_{k:02d}"
            k += 1
    # Default: keep all artifacts (models/figures) separated per run and easy to identify by script name.
    model_name = args.model_name or run_name

    if SummaryWriter is None:
        print("[Warn] tensorboard not installed. TensorBoard logging disabled (CSV logging still enabled).")
        writer = _NullWriter()
    else:
        # Unique TB Log Directory
        # Logic: runs/sac_f16_jsbsim_return/<run_name>_00, ...
        # But wait, LOG_DIR is runs/sac_f16_jsbsim_return
        # run_name is subfolder.
        # So we want runs/sac_f16_jsbsim_return/<run_name>_xx
        
        # User wants Model/Figure to stay as <run_name>.
        # TB logs to <run_name>_unique.
        
        k = 0
        while True:
            tb_dir = LOG_DIR / f"{run_name}_{k:02d}"
            if not tb_dir.exists():
                break
            k += 1
        tb_dir.mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard Logging to: {tb_dir}")
        writer = SummaryWriter(log_dir=str(tb_dir))

    run_log_dir = LOG_DIR / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = run_log_dir / "episode_metrics.csv"
    metrics_csv_exists = metrics_csv_path.exists()

    fig_dir = FIG_DIR / model_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    success_fig_dir = fig_dir / "success"
    failed_fig_dir = fig_dir / "failed"
    success_fig_dir.mkdir(parents=True, exist_ok=True)
    failed_fig_dir.mkdir(parents=True, exist_ok=True)
    weight_dir = MODELS_DIR / model_name
    weight_dir.mkdir(parents=True, exist_ok=True)

    dummy_state = env.reset()
    state_shape = dummy_state.shape[1:]
    state_size = state_shape[1]
    action_size = 4

    actor = Actor(state_size, action_size, multiplier=args.max_throttle).to(device)
    critic = Critic(state_size, action_size).to(device)
    target_critic = Critic(state_size, action_size).to(device)
    hard_target_update(critic, target_critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

    # Optional: initialize from a saved checkpoint (.pth)
    # === [MODIFIED] Smart Weight Loading ===
    if getattr(args, "init_from", None):
        ckpt_path = Path(str(args.init_from))
        if not ckpt_path.exists():
             # Try absolute path or project relative
             ckpt_path = PROJECT_ROOT / args.init_from
             if not ckpt_path.exists():
                 raise FileNotFoundError(f"--init_from not found: {{args.init_from}}")
        
        print(f"[Init] Using Smart Transfer Loading from: {{ckpt_path}}")
        # Load Actor (Key 'actor' usually)
        smart_load_weights(actor, ckpt_path, device, key='actor')
        # Load Critic (Key 'critic' usually)
        smart_load_weights(critic, ckpt_path, device, key='critic')
        # Target Critic sync
        target_critic.load_state_dict(critic.state_dict())
        
        # Load log_alpha if possible (optional)
        try:
             # Safer loading for log_alpha
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=device)

            if "log_alpha" in ckpt:
                with torch.no_grad():
                     # Handle both scalar and tensor cases
                    val = ckpt["log_alpha"]
                    if isinstance(val, torch.Tensor):
                        log_alpha.copy_(val.detach().to(device))
                    else:
                        log_alpha.copy_(torch.tensor([float(val)], device=device))
                print("[Init] log_alpha loaded.")
        except Exception as e:
            print(f"[Init] log_alpha skipped: {{e}}")

    else:
        print("[Init] No checkpoint specified (--init_from). Training from scratch.")

    replay_buffer = ReplayBuffer(args.replay_buffer, state_shape, action_size, device)

    start_episode = int(getattr(args, "start_episode", 0))
    for episode in range(start_episode, start_episode + int(args.episodes)):
        # ✅ Dynamic Target Entropy Schedule (linear): -|A| -> -2|A|
        # Use "local progress" so resume (--start_episode) behaves sensibly within this run.
        local_ep = int(episode - start_episode)
        denom = float(max(int(args.episodes) - 1, 1))
        progress = float(local_ep) / denom
        progress = float(min(max(progress, 0.0), 1.0))
        target_entropy_start = -float(action_size)       # e.g., -4.0
        target_entropy_end = -float(action_size) * 2.0   # e.g., -8.0
        current_target_entropy = torch.tensor(
            [target_entropy_start + progress * (target_entropy_end - target_entropy_start)],
            dtype=torch.float32,
            device=device,
        )

        state = env.reset()
        done = False
        score = 0.0
        steps = 0

        # Episode-level analytics (for offline analysis; lightweight CSV)
        sum_alt_penalty = 0.0
        sum_vert_penalty = 0.0
        sum_progress = 0.0
        min_range2d = float("inf")
        min_abs_up_err = float("inf")
        gate_steps = 0

        traj_lat, traj_lon, traj_alt = [], [], []
        traj_spd_kts = []
        traj_range2d_m, traj_bearing_err_deg, traj_wp_idx = [], [], []
        traj_up_err_m, traj_wp_idx_for_reward = [], []
        wp_reach_step = {0: None, 1: None, 2: None, 3: None}

        last_actor_loss = None
        last_critic_loss = None
        last_alpha_loss = None
        total_entropy = 0.0
        entropy_count = 0

        # Initial point + waypoints snapshot (fixed per episode)
        lat0, lon0, alt0 = env.get_position()
        wps0 = env.get_waypoints()
        traj_lat.append(lat0)
        traj_lon.append(lon0)
        traj_alt.append(alt0)
        traj_spd_kts.append(fdm_speed_kts(env.fdm))
        # Metrics at initial point (relative to current WP=0)
        wp_lat0, wp_lon0, wp_alt0 = env.get_waypoint()
        range2d0 = (haversine((lat0, lon0), (wp_lat0, wp_lon0)) * 1000.0) + 1e-6
        heading0 = float(env.fdm.get_property_value("attitude/psi-deg"))
        brg0 = float(bearing_deg(lat0, lon0, wp_lat0, wp_lon0))
        berr0 = float(wrap180(brg0 - heading0))
        up_err0 = float(wp_alt0 - alt0)
        traj_range2d_m.append(float(range2d0))
        traj_bearing_err_deg.append(float(berr0))
        traj_wp_idx.append(int(env.get_wp_index()))
        traj_wp_idx_for_reward.append(0)
        traj_up_err_m.append(up_err0)

        while not done and steps < args.max_steps:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                mu, std = actor(state_tensor)
                action = get_action(mu, std, multiplier=args.max_throttle)
                # Ensure the action stored in replay buffer matches the action actually applied to the env.
                action_env = np.asarray(action, dtype=np.float32).flatten()
                action_env[:3] = np.clip(action_env[:3], -1.0, 1.0)
                action_env[3] = np.clip(action_env[3], 0.0, 1.0)

            next_state, reward, done, info = env.step(action_env)

            # Time-limit truncation (max_steps): apply a terminal failure penalty to the LAST transition
            # so it affects learning (replay buffer), not just episode bookkeeping.
            time_limit_truncated = (not done) and ((steps + 1) >= int(args.max_steps))
            if time_limit_truncated:
                reward = float(reward) + float(mission.fail_penalty)
                done = True
                info = dict(info)
                info["time_limit_truncated"] = True
                info["fail_penalty"] = float(mission.fail_penalty)

            # Collect episode stats for analysis
            r2_now = float(info.get("range_2d_m", 0.0))
            up_now = float(info.get("up_err_m", 0.0))
            min_range2d = min(min_range2d, r2_now)
            min_abs_up_err = min(min_abs_up_err, abs(up_now))
            sum_alt_penalty += float(info.get("alt_penalty", 0.0))
            sum_vert_penalty += float(info.get("vert_penalty", 0.0))
            sum_progress += float(info.get("progress", 0.0))
            if (r2_now < float(mission.success_radius_m)) and (abs(up_now) < float(mission.success_alt_m)):
                gate_steps += 1

            lat, lon, alt = env.get_position()
            traj_lat.append(lat)
            traj_lon.append(lon)
            traj_alt.append(alt)
            traj_spd_kts.append(fdm_speed_kts(env.fdm))
            traj_range2d_m.append(float(info["range_2d_m"]))
            traj_bearing_err_deg.append(float(info["bearing_err_deg"]))
            traj_wp_idx.append(int(info["wp_idx"]))
            traj_wp_idx_for_reward.append(int(info.get("wp_idx_for_reward", max(int(info["wp_idx"]) - 1, 0))))
            traj_up_err_m.append(float(info["up_err_m"]))
            if bool(info.get("wp_reached", False)):
                ridx = info.get("reached_wp_idx", None)
                if isinstance(ridx, int) and (0 <= ridx <= 3) and (wp_reach_step.get(ridx) is None):
                    wp_reach_step[ridx] = int(steps)

            replay_buffer.push(state.squeeze(0), action_env, float(reward), next_state.squeeze(0), bool(done))
            state = next_state
            score += reward
            steps += 1

            if len(replay_buffer) >= args.batch_size:
                actor_loss, critic_loss, alpha_loss, alpha, entropy = sac_update(
                    replay_buffer,
                    args.batch_size,
                    actor,
                    critic,
                    target_critic,
                    alpha_opt,
                    actor_opt,
                    critic_opt,
                    log_alpha,
                    current_target_entropy,  # ✅ fixed -> dynamic
                    args.gamma,
                    args.max_throttle,
                )
                last_actor_loss = actor_loss
                last_critic_loss = critic_loss
                last_alpha_loss = alpha_loss
                total_entropy += entropy
                entropy_count += 1

        mission_success = bool(info.get("mission_success", False))

        # Plot & save
        try:
            ref_lat = float(traj_lat[0])
            ref_lon = float(traj_lon[0])
            ref_alt = float(traj_alt[0])

            agent_xy = [latlon_to_ne_m(ref_lat, ref_lon, la, lo) for la, lo in zip(traj_lat, traj_lon)]

            agent_n = np.array([p[0] for p in agent_xy], dtype=np.float32)
            agent_e = np.array([p[1] for p in agent_xy], dtype=np.float32)
            wp_neu = [latlon_to_ne_m(ref_lat, ref_lon, la, lo) for (la, lo, _) in wps0]
            wp_n = np.array([p[0] for p in wp_neu], dtype=np.float32)
            wp_e = np.array([p[1] for p in wp_neu], dtype=np.float32)
            wp_z = np.array([float(alt_wp) - ref_alt for (_, _, alt_wp) in wps0], dtype=np.float32)

            agent_z = np.array(traj_alt, dtype=np.float32) - ref_alt
            # last step index for mission success (if complete)
            idx_success = int(len(agent_e) - 1) if mission_success else -1

            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(3, 3, width_ratios=[1.4, 1.0, 0.75], hspace=0.35, wspace=0.25)
            ax3d = fig.add_subplot(gs[:, 0], projection="3d")
            ax_xy = fig.add_subplot(gs[0, 1])
            ax_xz = fig.add_subplot(gs[1, 1])
            ax_spd = fig.add_subplot(gs[2, 1])
            ax_leg = fig.add_subplot(gs[:, 2])
            ax_leg.axis("off")

            # Success gate params reused across subplots
            r_succ = float(mission.success_radius_m)
            h_succ = float(mission.success_alt_m)

            # 3D (E,N,Up)
            # Make 3D more intuitive:
            # - Color trajectory by "active WP stage" (to see intent/phase at a glance)
            # - Add a ground projection shadow
            # - Reduce perspective distortion (orthographic) + stable view angle + aspect ratio correction
            stage_colors = {0: "C1", 1: "C2", 2: "C3", 3: "C4"}  # to WP1/WP2/WP3/WP4(home)
            pts = np.column_stack([agent_e, agent_n, agent_z]).astype(np.float32, copy=False)
            if len(pts) >= 2:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                stages = np.array(traj_wp_idx_for_reward[: len(pts) - 1], dtype=np.int32)
                stages = np.clip(stages, 0, 3)
                seg_cols = [stage_colors.get(int(s), "C0") for s in stages]
                lc = Line3DCollection(segs, colors=seg_cols, linewidths=2.4, alpha=0.9)
                ax3d.add_collection3d(lc)
            else:
                ax3d.plot(agent_e, agent_n, agent_z, color="C0", linewidth=2, alpha=0.85, label="Agent")

            # Ground projection (shadow) to read lateral motion easier
            z_floor = float(min(float(agent_z.min()), float(wp_z.min())) - max(800.0, r_succ * 0.5))
            if len(pts) >= 2:
                pts_g = pts.copy()
                pts_g[:, 2] = z_floor
                segs_g = np.stack([pts_g[:-1], pts_g[1:]], axis=1)
                lc_g = Line3DCollection(segs_g, colors="k", linewidths=1.0, alpha=0.12)
                ax3d.add_collection3d(lc_g)

            ax3d.scatter(agent_e[0], agent_n[0], agent_z[0], s=80, color="C0", edgecolors="black", label="Start")
            ax3d.scatter(wp_e[0], wp_n[0], wp_z[0], s=120, color="C1", edgecolors="black", label="WP1")
            ax3d.scatter(wp_e[1], wp_n[1], wp_z[1], s=120, color="C2", edgecolors="black", label="WP2")
            ax3d.scatter(wp_e[2], wp_n[2], wp_z[2], s=120, color="C3", edgecolors="black", label="WP3")
            ax3d.scatter(wp_e[3], wp_n[3], wp_z[3], s=120, color="C4", edgecolors="black", label="WP4(Home)")
            # Success gate around each waypoint (visual aid): 2D radius + altitude band
            add_cylinder_3d(ax3d, wp_e[0], wp_n[0], wp_z[0], r_succ, h_succ, color="C1")
            add_cylinder_3d(ax3d, wp_e[1], wp_n[1], wp_z[1], r_succ, h_succ, color="C2")
            add_cylinder_3d(ax3d, wp_e[2], wp_n[2], wp_z[2], r_succ, h_succ, color="C3")
            add_cylinder_3d(ax3d, wp_e[3], wp_n[3], wp_z[3], r_succ, h_succ, color="C4")
            # Highlight trajectory segments that are inside the success gate (for the active WP at that time).
            in_gate = [
                (float(r2) < r_succ) and (abs(float(dz)) < h_succ)
                for r2, dz in zip(traj_range2d_m, traj_up_err_m)
            ]
            gate_label_added = False
            for i in range(len(agent_e) - 1):
                if i < len(in_gate) and in_gate[i]:
                    ax3d.plot(
                        agent_e[i : i + 2],
                        agent_n[i : i + 2],
                        agent_z[i : i + 2],
                        color="green",
                        linewidth=3.2,
                        alpha=0.95,
                        label="In Success Gate" if not gate_label_added else None,
                    )
                    gate_label_added = True
            for k, st in wp_reach_step.items():
                if st is not None and 0 <= st < len(agent_e):
                    ax3d.scatter(
                        agent_e[st],
                        agent_n[st],
                        agent_z[st],
                        s=160,
                        marker="*",
                        color=["C1", "C2", "C3", "C4"][k],
                        edgecolors="black",
                        label=f"Reached WP{k+1}",
                    )
            if idx_success >= 0:
                ax3d.scatter(
                    agent_e[idx_success],
                    agent_n[idx_success],
                    agent_z[idx_success],
                    s=180,
                    marker="*",
                    color="green",
                    edgecolors="black",
                    label="Success Point",
                )
            ax3d.set_title("3D Trajectory")
            ax3d.set_xlabel("East (m)")
            ax3d.set_ylabel("North (m)")
            ax3d.set_zlabel("Up (m)")
            # View / aspect improvements
            try:
                ax3d.set_proj_type("ortho")
            except Exception:
                pass
            ax3d.view_init(elev=22, azim=-55)
            m = float(max(r_succ * 1.6, 1000.0))
            ax3d.set_xlim(float(min(agent_e.min(), wp_e.min()) - m), float(max(agent_e.max(), wp_e.max()) + m))
            ax3d.set_ylim(float(min(agent_n.min(), wp_n.min()) - m), float(max(agent_n.max(), wp_n.max()) + m))
            ax3d.set_zlim(float(min(agent_z.min(), wp_z.min(), z_floor) - 200.0), float(max(agent_z.max(), wp_z.max()) + 200.0))
            try:
                xe = float(max(agent_e.max(), wp_e.max()) - min(agent_e.min(), wp_e.min()))
                yn = float(max(agent_n.max(), wp_n.max()) - min(agent_n.min(), wp_n.min()))
                zu = float(max(agent_z.max(), wp_z.max()) - min(agent_z.min(), wp_z.min()))
                ax3d.set_box_aspect((max(xe, 1.0), max(yn, 1.0), max(zu, 1.0)))
            except Exception:
                pass

            # XY
            ax_xy.plot(agent_e, agent_n, color="C0", linewidth=2, alpha=0.85, label="Agent")
            ax_xy.scatter(agent_e[0], agent_n[0], color="C0", edgecolors="black", s=40, label="Start")
            ax_xy.scatter(wp_e[0], wp_n[0], color="C1", edgecolors="black", s=60, label="WP1")
            ax_xy.scatter(wp_e[1], wp_n[1], color="C2", edgecolors="black", s=60, label="WP2")
            ax_xy.scatter(wp_e[2], wp_n[2], color="C3", edgecolors="black", s=60, label="WP3")
            ax_xy.scatter(wp_e[3], wp_n[3], color="C4", edgecolors="black", s=60, label="WP4(Home)")
            # Success radius (circle) around each waypoint (2D ground range)
            add_circle_xy(ax_xy, wp_e[0], wp_n[0], r_succ, color="C1")
            add_circle_xy(ax_xy, wp_e[1], wp_n[1], r_succ, color="C2")
            add_circle_xy(ax_xy, wp_e[2], wp_n[2], r_succ, color="C3")
            add_circle_xy(ax_xy, wp_e[3], wp_n[3], r_succ, color="C4")
            for i in range(len(agent_e) - 1):
                if i < len(in_gate) and in_gate[i]:
                    ax_xy.plot(agent_e[i : i + 2], agent_n[i : i + 2], color="green", linewidth=3.0, alpha=0.95)
            for k, st in wp_reach_step.items():
                if st is not None and 0 <= st < len(agent_e):
                    ax_xy.scatter(agent_e[st], agent_n[st], color=["C1", "C2", "C3", "C4"][k], edgecolors="black", s=120, marker="*")
            if idx_success >= 0:
                ax_xy.scatter(agent_e[idx_success], agent_n[idx_success], color="green", edgecolors="black", s=120, marker="*")
            ax_xy.set_title("Top-Down View (X-Y Plane)")
            ax_xy.set_xlabel("East (m)")
            ax_xy.set_ylabel("North (m)")
            # Fix ONLY Y-range (North) for consistent comparisons; keep X-range free.
            # With LOS=±30deg and max dist=10km, max |North| is ~5km (0.5*10km). Add margin.
            ax_xy.set_ylim(-6000.0, 6000.0)
            ax_xy.set_aspect("equal", adjustable="box")
            ax_xy.grid(True, linestyle="--", alpha=0.4)

            # XZ
            ax_xz.plot(agent_e, agent_z, color="C0", linewidth=2, alpha=0.85, label="Agent")
            ax_xz.scatter(agent_e[0], agent_z[0], color="C0", edgecolors="black", s=40, label="Start")
            ax_xz.scatter(wp_e[0], wp_z[0], color="C1", edgecolors="black", s=60, label="WP1")
            ax_xz.scatter(wp_e[1], wp_z[1], color="C2", edgecolors="black", s=60, label="WP2")
            ax_xz.scatter(wp_e[2], wp_z[2], color="C3", edgecolors="black", s=60, label="WP3")
            ax_xz.scatter(wp_e[3], wp_z[3], color="C4", edgecolors="black", s=60, label="WP4(Home)")
            # Success condition is 2D ground range (XY). In XZ we visualize a necessary condition as a vertical band:
            # |East - WP_East| < success_radius_m (North component is not shown in this plane).
            ax_xz.axvspan(wp_e[0] - r_succ, wp_e[0] + r_succ, color="C1", alpha=0.08, linewidth=0)
            ax_xz.axvspan(wp_e[1] - r_succ, wp_e[1] + r_succ, color="C2", alpha=0.08, linewidth=0)
            ax_xz.axvspan(wp_e[2] - r_succ, wp_e[2] + r_succ, color="C3", alpha=0.08, linewidth=0)
            ax_xz.axvspan(wp_e[3] - r_succ, wp_e[3] + r_succ, color="C4", alpha=0.08, linewidth=0)
            # Altitude success band (exact): |Up - WP_Up| < success_alt_m
            ax_xz.axhspan(wp_z[0] - h_succ, wp_z[0] + h_succ, color="C1", alpha=0.05, linewidth=0)
            ax_xz.axhspan(wp_z[1] - h_succ, wp_z[1] + h_succ, color="C2", alpha=0.05, linewidth=0)
            ax_xz.axhspan(wp_z[2] - h_succ, wp_z[2] + h_succ, color="C3", alpha=0.05, linewidth=0)
            ax_xz.axhspan(wp_z[3] - h_succ, wp_z[3] + h_succ, color="C4", alpha=0.05, linewidth=0)
            for i in range(len(agent_e) - 1):
                if i < len(in_gate) and in_gate[i]:
                    ax_xz.plot(agent_e[i : i + 2], agent_z[i : i + 2], color="green", linewidth=3.0, alpha=0.95)
            for k, st in wp_reach_step.items():
                if st is not None and 0 <= st < len(agent_e):
                    ax_xz.scatter(agent_e[st], agent_z[st], color=["C1", "C2", "C3", "C4"][k], edgecolors="black", s=120, marker="*")
            if idx_success >= 0:
                ax_xz.scatter(agent_e[idx_success], agent_z[idx_success], color="green", edgecolors="black", s=120, marker="*")
            ax_xz.set_title("Side View (X-Z Plane)")
            ax_xz.set_xlabel("East (m)")
            ax_xz.set_ylabel("Up (m)")
            # Fix ONLY Y-range (Up) for consistent comparisons; keep X-range free.
            ax_xz.set_ylim(-5000.0, 3000.0)
            ax_xz.grid(True, linestyle="--", alpha=0.4)

            # Speed
            t = np.arange(len(traj_spd_kts), dtype=np.int32)
            spd = np.array(traj_spd_kts, dtype=np.float32)
            ax_spd.plot(t, spd, color="C4", linewidth=2, label="Agent Speed")
            ax_spd.set_title(f"Speed Profile (min {spd.min():.0f}, avg {spd.mean():.0f}, max {spd.max():.0f} kts)")
            ax_spd.set_xlabel("Timestep")
            ax_spd.set_ylabel("Speed (kts)")
            ax_spd.grid(True, linestyle="--", alpha=0.4)

            result_str = "Goal Reached" if mission_success else "Failed"
            fig.suptitle(
                f"Episode {episode} - Result: {result_str} | Total Reward: {score:.2f} | Timesteps: {steps}",
                y=0.98,
                fontsize=14,
            )

            reached_cnt = sum(1 for v in wp_reach_step.values() if v is not None)
            idx = -1
            r2_v = float(traj_range2d_m[idx])
            berr_v = float(traj_bearing_err_deg[idx])
            wp_idx_v = int(traj_wp_idx[idx])
            ok_r = r2_v < float(mission.success_radius_m)
            up_v = float(traj_up_err_m[idx]) if len(traj_up_err_m) > 0 else 0.0
            ok_alt = abs(up_v) < float(mission.success_alt_m)
            box_text = (
                f"4-WP Return mission (WP4=Home)\n"
                f"Reached: {reached_cnt}/4   Current WP idx: {wp_idx_v}\n"
                f"Range2D: {r2_v:.0f} m (need < {mission.success_radius_m:.0f}) -> {'OK' if ok_r else 'NO'}\n"
                f"BearingErr: {berr_v:.1f}° (signed)\n"
                f"UpErr: {up_v:.0f} m (need |.| < {mission.success_alt_m:.0f}) -> {'OK' if ok_alt else 'NO'}\n"
                f"WP1: LOS {mission.wp1_los_deg:.0f}°, {mission.wp1_dist_min_m/1000:.1f}~{mission.wp1_dist_max_m/1000:.1f} km\n"
                f"WP2: LOS {mission.wp2_los_deg:.0f}°, {mission.wp2_dist_min_m/1000:.1f}~{mission.wp2_dist_max_m/1000:.1f} km\n"
                f"WP3: LOS {mission.wp3_los_deg:.0f}°, {mission.wp3_dist_min_m/1000:.1f}~{mission.wp3_dist_max_m/1000:.1f} km\n"
                f"WP4: Return to Start (Home)"
            )
            # Centralized legend + summary panel (keeps plots uncluttered)
            legend_map = {}

            def _collect(ax):
                hs, ls = ax.get_legend_handles_labels()
                for h, l in zip(hs, ls):
                    if l and (l not in legend_map):
                        legend_map[l] = h

            _collect(ax3d)
            _collect(ax_spd)
            # Add an explicit entry for success radius (visual aid)
            legend_map.setdefault(
                f"Success gate (r={mission.success_radius_m:.0f} m, |dz|<{mission.success_alt_m:.0f} m)",
                Line2D([0], [0], color="black", linewidth=2, alpha=0.25),
            )
            legend_map.setdefault("To WP1 path", Line2D([0], [0], color="C1", linewidth=3))
            legend_map.setdefault("To WP2 path", Line2D([0], [0], color="C2", linewidth=3))
            legend_map.setdefault("To WP3 path", Line2D([0], [0], color="C3", linewidth=3))
            legend_map.setdefault("To WP4(Home) path", Line2D([0], [0], color="C4", linewidth=3))
            legend_map.setdefault("In Success Gate", Line2D([0], [0], color="green", linewidth=3))

            ax_leg.text(0.0, 1.0, "Legend", transform=ax_leg.transAxes, va="top", ha="left", fontsize=12, weight="bold")
            ax_leg.legend(
                list(legend_map.values()),
                list(legend_map.keys()),
                loc="upper left",
                bbox_to_anchor=(0.0, 0.93),
                frameon=True,
                fontsize=10,
            )
            ax_leg.text(
                0.0,
                0.0,
                box_text,
                transform=ax_leg.transAxes,
                fontsize=10,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"),
            )

            save_dir = success_fig_dir if mission_success else failed_fig_dir
            fig.savefig(save_dir / f"episode_{episode:05d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Failure: could not save figure for episode {episode}. Error: {e}")

        # Logging
        writer.add_scalar("episode/score", score, episode)
        writer.add_scalar("episode/steps", steps, episode)
        writer.add_scalar("episode/range_2d_m_last", float(traj_range2d_m[-1]), episode)
        writer.add_scalar("episode/bearing_err_deg_last", float(traj_bearing_err_deg[-1]), episode)
        writer.add_scalar("episode/wp_idx_last", float(traj_wp_idx[-1]), episode)
        writer.add_scalar(
            "episode/wp_reached_count",
            float(sum(1 for v in wp_reach_step.values() if v is not None)),
            episode,
        )
        writer.add_scalar("episode/success", 1.0 if mission_success else 0.0, episode)
        writer.add_scalar("episode/min_range_2d_m", float(min_range2d if math.isfinite(min_range2d) else 0.0), episode)
        writer.add_scalar("episode/min_abs_up_err_m", float(min_abs_up_err if math.isfinite(min_abs_up_err) else 0.0), episode)
        writer.add_scalar("episode/gate_steps", float(gate_steps), episode)

        denom = float(max(int(steps), 1))
        writer.add_scalar("episode/alt_penalty_mean", float(sum_alt_penalty) / denom, episode)
        writer.add_scalar("episode/vert_penalty_mean", float(sum_vert_penalty) / denom, episode)
        writer.add_scalar("episode/progress_mean", float(sum_progress) / denom, episode)
        if last_actor_loss is not None:
            writer.add_scalar("loss/actor", last_actor_loss, episode)
            writer.add_scalar("episode/alpha", alpha, episode)
            writer.add_scalar("episode/entropy", float(total_entropy / entropy_count) if entropy_count > 0 else 0.0, episode)
            writer.add_scalar("episode/target_entropy", float(current_target_entropy.item()), episode)
        if last_critic_loss is not None:
            writer.add_scalar("loss/critic", last_critic_loss, episode)
        if last_alpha_loss is not None:
            writer.add_scalar("loss/alpha", last_alpha_loss, episode)

        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "actor_optimizer": actor_opt.state_dict(),
                "critic_optimizer": critic_opt.state_dict(),
                "alpha_optimizer": alpha_opt.state_dict(),
                "log_alpha": log_alpha,
                "mission": mission.__dict__,
                "args": vars(args),
            },
            weight_dir / f"epi_{episode:05d}.pth",
        )

        alpha_val = float(log_alpha.exp().item()) if last_actor_loss is not None else 1.0
        avg_entropy = total_entropy / entropy_count if entropy_count > 0 else 0.0
        te_val = float(current_target_entropy.item()) if last_actor_loss is not None else -float(action_size)
        success_mark = " [SUCCESS]" if mission_success else ""
        print(
            "{} episode | score: {:.2f} | alpha: {:.3f} | entropy: {:.2f} | target_entropy: {:.1f} | wp_reached: {}/4 | range2d(m): {:.0f}{}".format(
                episode,
                score,
                alpha_val,
                avg_entropy,
                te_val,
                sum(1 for v in wp_reach_step.values() if v is not None),
                float(traj_range2d_m[-1]),
                success_mark,
            )
        )

        # Append one row per episode for offline analysis
        try:
            with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode",
                        "steps",
                        "score",
                        "success",
                        "wp_reached_count",
                        "wp_idx_last",
                        "range2d_last_m",
                        "min_range2d_m",
                        "up_err_last_m",
                        "min_abs_up_err_m",
                        "gate_steps",
                        "alpha",
                        "avg_entropy",
                        "target_entropy",
                        "alt_penalty_mean",
                        "vert_penalty_mean",
                        "progress_mean",
                    ],
                )
                if not metrics_csv_exists:
                    w.writeheader()
                    metrics_csv_exists = True
                w.writerow(
                    {
                        "episode": int(episode),
                        "steps": int(steps),
                        "score": float(score),
                        "success": int(1 if mission_success else 0),
                        "wp_reached_count": int(sum(1 for v in wp_reach_step.values() if v is not None)),
                        "wp_idx_last": int(traj_wp_idx[-1]) if len(traj_wp_idx) > 0 else int(env.get_wp_index()),
                        "range2d_last_m": float(traj_range2d_m[-1]) if len(traj_range2d_m) > 0 else float("nan"),
                        "min_range2d_m": float(min_range2d if math.isfinite(min_range2d) else float("nan")),
                        "up_err_last_m": float(traj_up_err_m[-1]) if len(traj_up_err_m) > 0 else float("nan"),
                        "min_abs_up_err_m": float(min_abs_up_err if math.isfinite(min_abs_up_err) else float("nan")),
                        "gate_steps": int(gate_steps),
                        "alpha": float(alpha_val),
                        "avg_entropy": float(avg_entropy),
                        "target_entropy": float(current_target_entropy.item()),
                        "alt_penalty_mean": float(sum_alt_penalty) / float(max(int(steps), 1)),
                        "vert_penalty_mean": float(sum_vert_penalty) / float(max(int(steps), 1)),
                        "progress_mean": float(sum_progress) / float(max(int(steps), 1)),
                    }
                )
        except Exception as e:
            print(f"Warning: could not write episode_metrics.csv for episode {episode}: {e}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Episode index offset for logging/saving. Useful when continuing from a checkpoint (e.g., start at 1000).",
    )
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--agent_steps", type=int, default=5)
    parser.add_argument("--settle_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_buffer", type=int, default=50000)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--max_throttle",
        type=float,
        default=1.0,
        help="Throttle hard limit in [0,1]. 0=idle, 1=max power.",
    )
    parser.add_argument("--seed", type=int, default=None)

    # Mission: 3 waypoints (LOS angle relative to initial heading + distance range)
    parser.add_argument("--wp1_los_deg", type=float, default=30.0)
    parser.add_argument("--wp1_dist_min_m", type=float, default=3000.0)
    parser.add_argument("--wp1_dist_max_m", type=float, default=4000.0)
    parser.add_argument("--wp2_los_deg", type=float, default=30.0)
    parser.add_argument("--wp2_dist_min_m", type=float, default=6000.0)
    parser.add_argument("--wp2_dist_max_m", type=float, default=7000.0)
    parser.add_argument("--wp3_los_deg", type=float, default=30.0)
    parser.add_argument("--wp3_dist_min_m", type=float, default=9000.0)
    parser.add_argument("--wp3_dist_max_m", type=float, default=10000.0)
    parser.add_argument(
        "--wp_up_min_m",
        type=float,
        default=-500.0,
        help="Waypoint altitude offset min (meters) relative to spawn altitude (uniform).",
    )
    parser.add_argument(
        "--wp_up_max_m",
        type=float,
        default=500.0,
        help="Waypoint altitude offset max (meters) relative to spawn altitude (uniform).",
    )

    # Mission: success thresholds
    parser.add_argument(
        "--success_radius_m",
        type=float,
        default=1500.0,
        help="Per-waypoint success radius (2D ground range, meters).",
    )
    parser.add_argument(
        "--success_alt_m",
        type=float,
        default=1000.0,
        help="Per-waypoint altitude band (meters). Success requires |up_err| < this.",
    )
    parser.add_argument(
        "--fail_penalty",
        type=float,
        default=0.0,
        help="Terminal failure penalty (negative). Set 0 to disable.",
    )

    # Reward shaping
    parser.add_argument("--progress_scale", type=float, default=1.0 / 100.0)
    parser.add_argument("--align_weight", type=float, default=0.05)
    parser.add_argument("--step_penalty", type=float, default=-0.01)
    parser.add_argument("--wp_bonus", type=float, default=100.0)
    parser.add_argument("--mission_bonus", type=float, default=800.0)

    # Altitude hold regularization (keep weights MUCH smaller than progress shaping)
    parser.add_argument(
        "--alt_hold_deadzone_m",
        type=float,
        default=None,
        help="Altitude error deadzone (meters). If omitted, defaults to --success_alt_m.",
    )
    parser.add_argument(
        "--alt_hold_weight",
        type=float,
        default=2.0e-5,
        help="Penalty per meter beyond deadzone. Keep much smaller than progress reward.",
    )
    parser.add_argument(
        "--vert_speed_weight",
        type=float,
        default=5.0e-5,
        help="Penalty per |w_fps| (vertical speed in ft/s). Small regularizer to avoid extreme dives/climbs.",
    )

    # Safety termination
    parser.add_argument("--min_alt_m", type=float, default=2000.0)
    parser.add_argument("--max_sim_time_s", type=float, default=1500.0)

    # Run naming
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--model_name",
        default=None,
        help="If omitted, defaults to run_name to avoid overwriting previous runs.",
    )
    parser.add_argument("--init_from", type=str, default=None, help="Path to pretrained model checkpoint to load weights from")
    

    args = parser.parse_args()
    train(args)


