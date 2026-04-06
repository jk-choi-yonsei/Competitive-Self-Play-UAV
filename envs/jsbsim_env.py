import math
import jsbsim
import numpy as np
from geopy.distance import distance
from geopy.point import Point
from haversine import haversine

from .target_controllers import TargetPDConfig, TargetPDController, bearing_deg, wrap180
from .init_noise import InitNoiseConfig, meters_to_latlon_deg, uniform_symmetric


class JSBSimF16ChaseEnv:
    """
    Two-aircraft chase environment: agent pursues a target executing fixed control.
    Observation: agent kinematics (11) + relative geo to target (9) => 20 features, 
    stacked over 5 frames -> (1, 5, 20).
    """

    def __init__(
        self,
        agent_steps=5,
        settle_steps=5,
        target_action=None,
        target_offset_lat=0.05,
        target_offset_long=0.02,
        target_offset_alt=300.0,
        target_policy="fixed",
        target_pd_config=None,
        init_noise_config=None,
    ):
        # Initialize JSBSim FDMs
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm_target = jsbsim.FGFDMExec(None)
        for fdm in (self.fdm, self.fdm_target):
            fdm.set_debug_level(0)
            fdm.set_dt(1.0 / 50.0)

        self.model = "f16"
        if not self.fdm.load_model(self.model):
            raise RuntimeError("Failed to load JSBSim model f16 (agent)")
        if not self.fdm_target.load_model(self.model):
            raise RuntimeError("Failed to load JSBSim model f16 (target)")

        self.initial_conditions = {
            "ic/h-sl-ft": 24000,
            "ic/u-fps": 500,
            "ic/v-fps": 0,
            "ic/w-fps": 0,
            "ic/long-gc-deg": 2.3,
            "ic/lat-gc-deg": 2.3,
            "ic/terrain-elevation-ft": 10,
            "ic/psi-true-deg": 90,
            "ic/roc-fpm": 0,
        }

        self.agent_steps = agent_steps
        self.settle_steps = settle_steps

        self.target_action = (
            np.array([0.111, -0.47, 0.11, 0.38], dtype=np.float32)
            if target_action is None
            else np.asarray(target_action, dtype=np.float32)
        )
        self.target_offset_lat = target_offset_lat
        self.target_offset_long = target_offset_long
        self.target_offset_alt = target_offset_alt

        if init_noise_config is None:
            self.init_noise_config = InitNoiseConfig(enabled=False)
        elif isinstance(init_noise_config, dict):
            self.init_noise_config = InitNoiseConfig(**init_noise_config)
        elif isinstance(init_noise_config, InitNoiseConfig):
            self.init_noise_config = init_noise_config
        else:
            raise TypeError("init_noise_config must be None, dict, or InitNoiseConfig.")

        self._rng = np.random.default_rng(self.init_noise_config.seed)

        self.target_policy = str(target_policy).lower()
        if self.target_policy not in ("fixed", "pd"):
            raise ValueError(f"Invalid target_policy: {target_policy}. Use 'fixed' or 'pd'.")

        if target_pd_config is None:
            self.target_pd_config = TargetPDConfig(throttle_cmd=float(self.target_action[3]))
        elif isinstance(target_pd_config, dict):
            self.target_pd_config = TargetPDConfig(**target_pd_config)
        elif isinstance(target_pd_config, TargetPDConfig):
            self.target_pd_config = target_pd_config
        else:
            raise TypeError("target_pd_config must be None, dict, or TargetPDConfig.")

        self._target_controller = TargetPDController(self.target_pd_config)

        # Reward constants (based on SAC_Return.py)
        self.progress_scale = 1.0 / 100.0
        self.align_weight = 0.1    # Increased from 0.05
        self.aot_weight = 0.2      # [NEW] Incentivize getting behind
        self.step_penalty = -0.01
        self.aim_bonus = 0.5       # [NEW] Strong reward for precise aiming (ATA < 15)
        self.success_bonus = 1000.0
        self.fail_penalty = -50.0
        self.vert_speed_weight = 5.0e-5
        
        self.lock_on_timer = 0.0  # [NEW] Timer for sustained success

        self._apply_initial_conditions()

    def _apply_initial_conditions(self):
        """Initialize both agent and target aircraft."""
        for prop, val in self.initial_conditions.items():
            self.fdm.set_property_value(prop, val)

        # Target starts near agent with small offset
        for prop, val in self.initial_conditions.items():
            self.fdm_target.set_property_value(prop, val)

        base_lat = float(self.initial_conditions["ic/lat-gc-deg"])
        base_lon = float(self.initial_conditions["ic/long-gc-deg"])
        base_alt_ft = float(self.initial_conditions["ic/h-sl-ft"])
        base_u_fps = float(self.initial_conditions["ic/u-fps"])
        base_psi_deg = float(self.initial_conditions["ic/psi-true-deg"])

        # Apply initialization noise (no random noise by default)
        
        # Method A: Physical Advantage Implementation
        # Give Agent extra energy to break deadlock
        # +100 kts speed (~169 fps)
        # +500 m altitude (~1640 ft)
        agent_speed_advantage_fps = 169.0
        agent_alt_advantage_ft = 1640.0

        if self.init_noise_config.enabled:
            agent_u_noise = float(self._rng.normal(0.0, float(self.init_noise_config.agent_sigma_u_fps)))
            agent_psi_noise = uniform_symmetric(self._rng, self.init_noise_config.agent_heading_uniform_deg)

            # Agent: Base + Advantage + Noise
            self.fdm.set_property_value("ic/u-fps", base_u_fps + agent_speed_advantage_fps + agent_u_noise)
            self.fdm.set_property_value("ic/h-sl-ft", base_alt_ft + agent_alt_advantage_ft)
            self.fdm.set_property_value("ic/psi-true-deg", base_psi_deg + agent_psi_noise)

            tgt_u_noise = float(self._rng.normal(0.0, float(self.init_noise_config.target_sigma_u_fps)))
            tgt_psi_noise = uniform_symmetric(self._rng, self.init_noise_config.target_heading_uniform_deg)
            # Target: Base + Noise
            self.fdm_target.set_property_value("ic/u-fps", base_u_fps + tgt_u_noise)
            self.fdm_target.set_property_value("ic/psi-true-deg", base_psi_deg + tgt_psi_noise)

            # Target relative position noise in meters -> degrees
            north_m = float(self._rng.normal(0.0, float(self.init_noise_config.target_sigma_north_m)))
            east_m = float(self._rng.normal(0.0, float(self.init_noise_config.target_sigma_east_m)))
            dlat_deg, dlon_deg = meters_to_latlon_deg(north_m, east_m, ref_lat_deg=base_lat)
            alt_noise_m = float(self._rng.normal(0.0, float(self.init_noise_config.target_sigma_alt_m)))
        else:
            # Agent: Base + Advantage
            self.fdm.set_property_value("ic/u-fps", base_u_fps + agent_speed_advantage_fps)
            self.fdm.set_property_value("ic/h-sl-ft", base_alt_ft + agent_alt_advantage_ft)
            
            # Target: Base
            self.fdm_target.set_property_value("ic/u-fps", base_u_fps)
            
            dlat_deg, dlon_deg, alt_noise_m = 0.0, 0.0, 0.0

        self.fdm_target.set_property_value("ic/lat-gc-deg", base_lat + float(self.target_offset_lat) + dlat_deg)
        self.fdm_target.set_property_value("ic/long-gc-deg", base_lon + float(self.target_offset_long) + dlon_deg)
        self.fdm_target.set_property_value(
            "ic/h-sl-ft", base_alt_ft + float(self.target_offset_alt + alt_noise_m) * 3.28084
        )

        self.fdm.reset_to_initial_conditions(0)
        self.fdm_target.reset_to_initial_conditions(0)

        for sim in (self.fdm, self.fdm_target):
            sim.set_property_value("propulsion/starter_cmd", 1)
            sim.set_property_value("propulsion/engine/set-running", 1)

            sim.set_property_value("fcs/throttle-cmd-norm", 1.0)
            sim.set_property_value("gear/gear-cmd-norm", 0)

        # Settle simulation
        for _ in range(self.settle_steps):
            self.fdm.run()
            self._run_target()

    def _run_target_fixed(self):
        """Execute fixed control policy for target aircraft."""
        a = float(np.clip(self.target_action[0], -1, 1))
        e = float(np.clip(self.target_action[1], -1, 1))
        r = float(np.clip(self.target_action[2], -1, 1))
        t = float(np.clip(self.target_action[3], 0, 1))
        self.last_target_action = np.array([a, e, r, t], dtype=np.float32)
        self.fdm_target.set_property_value("fcs/aileron-cmd-norm", a)
        self.fdm_target.set_property_value("fcs/elevator-cmd-norm", e)
        self.fdm_target.set_property_value("fcs/rudder-cmd-norm", r)
        self.fdm_target.set_property_value("fcs/throttle-cmd-norm", t)
        self.fdm_target.run()

    def _run_target_pd(self):
        """Execute PD control policy for target aircraft (look-at agent)."""
        agent_lat = self.fdm.get_property_value("position/lat-gc-deg")
        agent_lon = self.fdm.get_property_value("position/long-gc-deg")
        agent_alt = self.fdm.get_property_value("position/h-sl-meters")

        tgt_lat = self.fdm_target.get_property_value("position/lat-gc-deg")
        tgt_lon = self.fdm_target.get_property_value("position/long-gc-deg")
        tgt_alt = self.fdm_target.get_property_value("position/h-sl-meters")

        # Target attitude and rates
        tgt_heading = self.fdm_target.get_property_value("attitude/psi-deg")
        tgt_pitch = self.fdm_target.get_property_value("attitude/theta-deg")
        tgt_roll = self.fdm_target.get_property_value("attitude/phi-deg")

        tgt_p = self.fdm_target.get_property_value("velocities/p-rad_sec")
        tgt_q = self.fdm_target.get_property_value("velocities/q-rad_sec")
        tgt_r = self.fdm_target.get_property_value("velocities/r-rad_sec")

        ground_range_m = (haversine((tgt_lat, tgt_lon), (agent_lat, agent_lon)) * 1000.0) + 1e-6

        aileron, elevator, rudder, throttle = self._target_controller.compute_action(
            agent_lat_deg=agent_lat,
            agent_lon_deg=agent_lon,
            agent_alt_m=agent_alt,
            target_lat_deg=tgt_lat,
            target_lon_deg=tgt_lon,
            target_alt_m=tgt_alt,
            target_heading_deg=tgt_heading,
            target_pitch_deg=tgt_pitch,
            target_roll_deg=tgt_roll,
            target_p_rad_s=tgt_p,
            target_q_rad_s=tgt_q,
            target_r_rad_s=tgt_r,
            ground_range_m=ground_range_m,
        )

        a = float(np.clip(aileron, -1, 1))
        e = float(np.clip(elevator, -1, 1))
        r = float(np.clip(rudder, -1, 1))
        t = float(np.clip(throttle, 0, 1))
        self.last_target_action = np.array([a, e, r, t], dtype=np.float32)
        self.fdm_target.set_property_value("fcs/aileron-cmd-norm", a)
        self.fdm_target.set_property_value("fcs/elevator-cmd-norm", e)
        self.fdm_target.set_property_value("fcs/rudder-cmd-norm", r)
        self.fdm_target.set_property_value("fcs/throttle-cmd-norm", t)
        self.fdm_target.run()

    def _run_target(self):
        """Run target step according to target_policy."""
        if self.target_policy == "pd":
            self._run_target_pd()
        else:
            self._run_target_fixed()

    def reset(self):
        """Reset environment and return initial observation."""
        self.lock_on_timer = 0.0
        self._apply_initial_conditions()
        return self._stacked_observation()

    def _get_range_3d(self) -> float:
        lat = self.fdm.get_property_value("position/lat-gc-deg")
        lon = self.fdm.get_property_value("position/long-gc-deg")
        alt = self.fdm.get_property_value("position/h-sl-meters")
        tgt_lat = self.fdm_target.get_property_value("position/lat-gc-deg")
        tgt_lon = self.fdm_target.get_property_value("position/long-gc-deg")
        tgt_alt = self.fdm_target.get_property_value("position/h-sl-meters")
        
        ground_range_m = (haversine((lat, lon), (tgt_lat, tgt_lon)) * 1000.0) + 1e-6
        return float(np.sqrt(ground_range_m**2 + float(alt - tgt_alt) ** 2))

    def _get_state(self):
        """Extract agent state features with normalization."""
        alt_m = self.fdm.get_property_value("position/h-sl-meters")
        
        # Normalize features to similar scales
        state_vals = [
            self.fdm.get_property_value("velocities/u-fps") / 500.0,        # ~[-2, 2]
            self.fdm.get_property_value("velocities/v-fps") / 100.0,        # ~[-2, 2]
            self.fdm.get_property_value("velocities/w-fps") / 100.0,        # ~[-2, 2]
            alt_m / 10000.0,                                                 # ~[0, 2]
            self.fdm.get_property_value("velocities/p-rad_sec") / 3.0,      # ~[-1, 1]
            self.fdm.get_property_value("velocities/q-rad_sec") / 3.0,      # ~[-1, 1]
            self.fdm.get_property_value("velocities/r-rad_sec") / 3.0,      # ~[-1, 1]
            self.fdm.get_property_value("attitude/phi-deg") / 180.0,        # ~[-1, 1]
            self.fdm.get_property_value("attitude/theta-deg") / 90.0,       # ~[-1, 1]
            self.fdm.get_property_value("attitude/psi-deg") / 180.0,        # ~[-2, 2]
            self.fdm.get_property_value("attitude/pitch-rad") / 1.57,       # ~[-1, 1]
        ]
        return np.array(state_vals, dtype=np.float32)

    def _positional_geo_raw(self):
        """Compute raw relative geometric features (for reward calculation)."""
        lat = round(self.fdm.get_property_value("position/lat-gc-deg"), 5)
        lon = round(self.fdm.get_property_value("position/long-gc-deg"), 5)
        alt_m = self.fdm.get_property_value("position/h-sl-meters")

        tgt_lat = round(self.fdm_target.get_property_value("position/lat-gc-deg"), 5)
        tgt_lon = round(self.fdm_target.get_property_value("position/long-gc-deg"), 5)
        tgt_alt = self.fdm_target.get_property_value("position/h-sl-meters")

        diff_long = lon - tgt_lon + 1e-10
        diff_lat = lat - tgt_lat + 1e-10
        diff_alt = alt_m - tgt_alt

        to_tgt = np.rad2deg(math.atan(diff_lat / diff_long))
        if diff_long > 0 and diff_lat > 0:
            to_tgt_hdg = 270 - to_tgt
        elif diff_long < 0 < diff_lat:
            to_tgt_hdg = 90 - to_tgt
        elif diff_long > 0 > diff_lat:
            to_tgt_hdg = 270 - to_tgt
        else:
            to_tgt_hdg = to_tgt

        a_pos = (lat, lon)
        dest = (tgt_lat, tgt_lon)
        range_km = (haversine(a_pos, dest) * 1000) + 1e-6
        to_tgt_pitch = math.atan(diff_alt / range_km)

        point = Point(lat, lon)
        point_target = Point(tgt_lat, tgt_lon)
        distance_2d = distance(point, point_target).kilometers
        distance_3d = np.sqrt(distance_2d**2 + (alt_m / 100 - tgt_alt / 100) ** 2)

        heading = self.fdm.get_property_value("attitude/psi-deg")
        heading_target = self.fdm_target.get_property_value("attitude/psi-deg")
        if heading > heading_target:
            hca = heading - heading_target
            angle_off = 360 - hca if hca > 180 else hca
        else:
            hca = heading_target - heading
            angle_off = 360 - hca if hca > 180 else hca

        tgt_x = np.sin(np.radians(heading_target))
        tgt_y = np.cos(np.radians(heading_target))
        long_back = tgt_lon - tgt_x
        lat_back = tgt_lat - tgt_y
        p1 = [long_back, lat_back]
        p2 = [tgt_lon, tgt_lat]
        p3 = [lon, lat]
        pt1 = (p1[0] - p2[0], p1[1] - p2[1])
        pt2 = (p3[0] - p2[0], p3[1] - p2[1])
        ang1 = np.arctan2(pt1[1], pt1[0])
        ang2 = np.arctan2(pt2[1], pt2[0])
        aspect = np.rad2deg(np.abs(ang1 - ang2))
        aspect_angle = aspect - (aspect - 180) * 2 if aspect > 180 else aspect
        ang1_deg = np.rad2deg(ang1)
        ang2_deg = np.rad2deg(ang2)
        if ang1_deg > 0:
            ang1_deg = -360 + ang1_deg
        if ang2_deg > 0:
            ang2_deg = -360 + ang2_deg
        if 0 >= ang1_deg >= -180:
            if ang1_deg > ang2_deg > ang1_deg - 180:
                aspect_angle *= -1
        else:
            if ang2_deg < ang1_deg or 0 > ang2_deg > ang1_deg + 180:
                aspect_angle *= -1

        return np.array(
            [
                aspect_angle,
                angle_off,
                distance_2d,
                distance_3d,
                diff_lat,
                diff_long,
                diff_alt,
                to_tgt_hdg,
                to_tgt_pitch,
            ],
            dtype=np.float32,
        )
    
    def _positional_geo(self):
        """Compute normalized relative geometric features (for observation)."""
        raw = self._positional_geo_raw()
        # Normalize for neural network input
        return np.array(
            [
                raw[0] / 180.0,        # aspect_angle: ~[-1, 1]
                raw[1] / 180.0,        # angle_off: ~[0, 1]
                raw[2] / 50.0,         # distance_2d: ~[0, 2]
                raw[3] / 50.0,         # distance_3d: ~[0, 2]
                raw[4] / 0.1,          # diff_lat: ~[-1, 1]
                raw[5] / 0.1,          # diff_long: ~[-1, 1]
                raw[6] / 1000.0,       # diff_alt: ~[-1, 1]
                raw[7] / 180.0,        # to_tgt_hdg: ~[-2, 2]
                raw[8] / 1.57,         # to_tgt_pitch: ~[-1, 1]
            ],
            dtype=np.float32,
        )

    def _stacked_observation(self):
        """Create stacked observation over 5 time steps."""
        frames = []
        for _ in range(5):
            self._run_target()
            self.fdm.run()
            frames.append(np.hstack([self._get_state(), self._positional_geo()]))
        obs = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)
        return np.reshape(obs, (1, 5, -1))

    def _reward_done(self, range_3d_prev: float):
        """Compute reward and check termination conditions (based on SAC_Return.py)."""
        # Retrieve current state
        lat = self.fdm.get_property_value("position/lat-gc-deg")
        lon = self.fdm.get_property_value("position/long-gc-deg")
        alt = self.fdm.get_property_value("position/h-sl-meters")
        heading = self.fdm.get_property_value("attitude/psi-deg")

        tgt_lat = self.fdm_target.get_property_value("position/lat-gc-deg")
        tgt_lon = self.fdm_target.get_property_value("position/long-gc-deg")
        tgt_alt = self.fdm_target.get_property_value("position/h-sl-meters")
        tgt_heading = self.fdm_target.get_property_value("attitude/psi-deg")

        ground_range_m = (haversine((lat, lon), (tgt_lat, tgt_lon)) * 1000.0) + 1e-6
        range_3d_m = float(np.sqrt(ground_range_m**2 + float(alt - tgt_alt) ** 2))

        # ATA: angle between agent heading and LOS (agent -> target)
        brg_agent_to_tgt = bearing_deg(lat, lon, tgt_lat, tgt_lon)
        ata_deg = abs(wrap180(brg_agent_to_tgt - float(heading)))

        # AOT: angle from target tail to LOS (target -> agent)
        brg_tgt_to_agent = bearing_deg(tgt_lat, tgt_lon, lat, lon)
        tgt_tail_deg = (float(tgt_heading) + 180.0) % 360.0
        aot_deg = abs(wrap180(brg_tgt_to_agent - tgt_tail_deg))

        self.last_range_3d_m = range_3d_m
        self.last_ata_deg = ata_deg
        self.last_aot_deg = aot_deg

        # Distance-based reward
        reward = -range_3d_m / 10.0
        if range_3d_m < 10:
            reward += np.exp(-range_3d_m / 3.0)

        done = False
        success = False
        
        # Success condition: immediate WEZ entry
        is_in_wez = (500.0 < range_3d_m < 2000.0) and (ata_deg < 30.0) and (aot_deg < 60.0)
        is_safe_altitude = alt > 3000.0

        if is_in_wez and is_safe_altitude:
            reward += self.success_bonus
            done = True
            success = True

        # Failure condition
        # Min altitude or timeout
        if alt < 3000 or self.fdm.get_property_value("simulation/sim-time-sec") > 1500.0:
            # Note: Success check takes precedence (if we locked on just before crashing floor? 
            # But is_safe_altitude > 3000 prevents lock-on near floor anyway).
            done = True
            if not success:
                reward += self.fail_penalty  # Usually 0 or negative
                # ChaseEnv previous: -50. SAC_Return: 0. 

        if np.isnan(reward) or np.isnan(range_3d_m):
            done = True
            reward = -10.0
        
        # Add timer to info for debugging
        self.last_lock_on_time = getattr(self, "lock_on_timer", 0.0)

        return float(reward), done, success

    def get_position(self):
        """Return current agent position (lat, lon, alt_m)."""
        lat = round(self.fdm.get_property_value("position/lat-gc-deg"), 5)
        lon = round(self.fdm.get_property_value("position/long-gc-deg"), 5)
        alt_m = self.fdm.get_property_value("position/h-sl-meters")
        return lat, lon, alt_m

    def get_target_position(self):
        """Return current target position (lat, lon, alt_m)."""
        tgt_lat = round(self.fdm_target.get_property_value("position/lat-gc-deg"), 5)
        tgt_lon = round(self.fdm_target.get_property_value("position/long-gc-deg"), 5)
        tgt_alt = self.fdm_target.get_property_value("position/h-sl-meters")
        return tgt_lat, tgt_lon, tgt_alt

    def step(self, action):
        """Execute action and return next observation, reward, done, info."""
        action = np.asarray(action, dtype=np.float32).flatten()
        aileron, elevator, rudder, throttle = action
        self.fdm.set_property_value("fcs/aileron-cmd-norm", float(np.clip(aileron, -1, 1)))
        self.fdm.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elevator, -1, 1)))
        self.fdm.set_property_value("fcs/rudder-cmd-norm", float(np.clip(rudder, -1, 1)))
        self.fdm.set_property_value("fcs/rudder-cmd-norm", float(np.clip(rudder, -1, 1)))
        
        # Throttle & Speedbrake mapping
        # action > 0: Throttle command (0~1), Speedbrake = 0
        # action < 0: Throttle = 0, Speedbrake command (0~1 based on magnitude)
        raw_throttle = float(throttle)
        if raw_throttle >= 0.0:
            throttle_cmd = float(np.clip(raw_throttle, 0.0, 1.0))
            sb_cmd = 0.0
        else:
            throttle_cmd = 0.0
            sb_cmd = float(np.clip(-raw_throttle, 0.0, 1.0))
            
        self.fdm.set_property_value("fcs/throttle-cmd-norm", throttle_cmd)
        self.fdm.set_property_value("fcs/speedbrake-cmd-norm", sb_cmd)

        # Capture range before stepping for potential shaping
        range_3d_prev = self._get_range_3d()

        for _ in range(self.agent_steps):
            self._run_target()
            self.fdm.run()

        obs = self._stacked_observation()
        reward, done, success = self._reward_done(range_3d_prev)
        info = {
            "success": success,
            "range_3d_m": float(getattr(self, "last_range_3d_m", float("nan"))),
            "ata_deg": float(getattr(self, "last_ata_deg", float("nan"))),
            "aot_deg": float(getattr(self, "last_aot_deg", float("nan"))),
            "lock_on_time": float(getattr(self, "last_lock_on_time", 0.0)),
            "target_action": np.asarray(getattr(self, "last_target_action", np.array([np.nan, np.nan, np.nan, np.nan])),
                                        dtype=np.float32),
        }
        return obs, reward, done, info
