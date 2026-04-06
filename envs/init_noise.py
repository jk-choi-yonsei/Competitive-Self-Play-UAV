import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class InitNoiseConfig:
    enabled: bool = False
    seed: Optional[int] = None

    # Target relative position noise (meters, Gaussian sigma)
    target_sigma_north_m: float = 500.0
    target_sigma_east_m: float = 500.0
    target_sigma_alt_m: float = 100.0

    # Agent initial condition noise
    agent_sigma_u_fps: float = 50.0
    agent_heading_uniform_deg: float = 20.0  # uniform in [-x, x]

    # Optional target speed/heading noise (disabled by default)
    target_sigma_u_fps: float = 0.0
    target_heading_uniform_deg: float = 0.0


def meters_to_latlon_deg(north_m: float, east_m: float, ref_lat_deg: float) -> Tuple[float, float]:
    """Convert local N/E meters to delta lat/lon degrees at reference latitude."""
    meters_per_deg_lat = 111_111.0
    dlat = float(north_m) / meters_per_deg_lat
    cos_lat = max(math.cos(math.radians(float(ref_lat_deg))), 1e-6)
    meters_per_deg_lon = meters_per_deg_lat * cos_lat
    dlon = float(east_m) / meters_per_deg_lon
    return dlat, dlon


def uniform_symmetric(rng: np.random.Generator, half_range: float) -> float:
    hr = float(half_range)
    if hr <= 0.0:
        return 0.0
    return float(rng.uniform(-hr, hr))

