import math
from dataclasses import dataclass
from typing import Tuple


def wrap180(deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    x = (deg + 180.0) % 360.0 - 180.0
    return x


def bearing_deg(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """
    Initial bearing from (lat1, lon1) to (lat2, lon2) in degrees [0, 360).
    Uses great-circle bearing formula.
    """
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    dlon = math.radians(lon2_deg - lon1_deg)

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def elevation_deg(alt_from_m: float, alt_to_m: float, ground_range_m: float) -> float:
    """Elevation angle from 'from' to 'to' (deg). Positive means target is above horizon."""
    gr = max(float(ground_range_m), 1e-3)
    return math.degrees(math.atan2(float(alt_to_m) - float(alt_from_m), gr))


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), float(lo)), float(hi))


@dataclass
class TargetPDConfig:
    # Outer loop: LOS heading/pitch -> desired bank/pitch
    k_heading_to_bank: float = 0.8  # bank_deg per heading_error_deg
    k_elev_to_pitch: float = 1.0    # pitch_cmd_deg per elev_error_deg

    # Inner loop: bank/pitch tracking
    k_bank_p: float = 0.06          # aileron per bank_error_deg
    k_bank_d: float = 0.02          # aileron per roll_rate_deg_s
    k_pitch_p: float = 0.05         # elevator per pitch_error_deg
    k_pitch_d: float = 0.02         # elevator per pitch_rate_deg_s

    # Damping / limits
    k_rudder_damp: float = 0.05     # rudder per yaw_rate_deg_s
    max_bank_deg: float = 55.0
    max_pitch_cmd_deg: float = 20.0

    # Surface sign conventions (JSBSim F-16 often uses negative elevator for nose-up)
    aileron_sign: float = 1.0
    elevator_sign: float = -1.0

    # Fixed throttle (0..1)
    throttle_cmd: float = 0.38


class TargetPDController:
    """
    PD target controller that turns the target aircraft to look at the agent.
    Outputs action: (aileron, elevator, rudder, throttle).
    """

    def __init__(self, config: TargetPDConfig):
        self.cfg = config

    def compute_action(
        self,
        *,
        agent_lat_deg: float,
        agent_lon_deg: float,
        agent_alt_m: float,
        target_lat_deg: float,
        target_lon_deg: float,
        target_alt_m: float,
        target_heading_deg: float,
        target_pitch_deg: float,
        target_roll_deg: float,
        target_p_rad_s: float,
        target_q_rad_s: float,
        target_r_rad_s: float,
        ground_range_m: float,
    ) -> Tuple[float, float, float, float]:
        # LOS desired angles (from target to agent)
        los_heading = bearing_deg(target_lat_deg, target_lon_deg, agent_lat_deg, agent_lon_deg)
        los_elev = elevation_deg(target_alt_m, agent_alt_m, ground_range_m)

        # Errors
        heading_err = wrap180(los_heading - float(target_heading_deg))
        pitch_err = wrap180(los_elev - float(target_pitch_deg))

        # Outer loop: map LOS errors to desired attitudes (bank and pitch)
        desired_bank = clamp(self.cfg.k_heading_to_bank * heading_err, -self.cfg.max_bank_deg, self.cfg.max_bank_deg)
        desired_pitch = clamp(
            float(target_pitch_deg) + self.cfg.k_elev_to_pitch * pitch_err,
            -self.cfg.max_pitch_cmd_deg,
            self.cfg.max_pitch_cmd_deg,
        )

        # Inner loop PD tracking using body rates as D term
        p_deg_s = math.degrees(float(target_p_rad_s))
        q_deg_s = math.degrees(float(target_q_rad_s))
        r_deg_s = math.degrees(float(target_r_rad_s))

        bank_err = wrap180(desired_bank - float(target_roll_deg))
        pitch_track_err = wrap180(desired_pitch - float(target_pitch_deg))

        aileron = self.cfg.aileron_sign * (self.cfg.k_bank_p * bank_err - self.cfg.k_bank_d * p_deg_s)
        elevator = self.cfg.elevator_sign * (self.cfg.k_pitch_p * pitch_track_err - self.cfg.k_pitch_d * q_deg_s)
        rudder = -self.cfg.k_rudder_damp * r_deg_s

        # Clamp to actuator ranges
        aileron = clamp(aileron, -1.0, 1.0)
        elevator = clamp(elevator, -1.0, 1.0)
        rudder = clamp(rudder, -1.0, 1.0)
        throttle = clamp(self.cfg.throttle_cmd, 0.0, 1.0)

        return aileron, elevator, rudder, throttle

