"""Vehicle fleet generation (columnar numpy arrays, no per-vehicle Python objects)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import VehicleConfig
from .dynamics import DEFAULT_PARAMS
from .models import (
    DIESEL_EFFICIENCY_BONUS, DRIVING_STYLES, FUEL_DIESEL, FUEL_HYBRID, FUEL_TYPES,
    HYBRID_EFFICIENCY_BONUS, VEHICLE_TYPE_SPECS, VEHICLE_TYPES,
)
from .rng import make_rng
from .routes import RouteCatalog

STYLE_SHARES = (0.25, 0.55, 0.20)
SPEED_FACTOR_SD, SPEED_FACTOR_RANGE = 0.035, (0.88, 1.12)
TANK_SD_L = 4.0
HIGHWAY_PREF_SD = 1.0
ROUTE_SIGMA_RANGE = (0.6, 1.4)
KMPL_GLOBAL_RANGE = (5.0, 32.0)


@dataclass
class VehicleFleet:
    n: int
    vehicle_type: np.ndarray
    fuel_type: np.ndarray
    driving_style: np.ndarray
    kmpl: np.ndarray
    speed_factor: np.ndarray
    tank_l: np.ndarray
    highway_pref: np.ndarray     # >0 likes highways, <0 avoids them
    route_sigma: np.ndarray      # how idiosyncratic this driver's route habits are
    wear: np.ndarray             # latent fuel-condition factor (not exported)
    home_node: np.ndarray
    work_node: np.ndarray
    poi0: np.ndarray
    poi1: np.ndarray
    id_width: int

    def ids(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        stop = self.n if stop is None else stop
        return np.array([f"V{i + 1:0{self.id_width}d}" for i in range(start, stop)])

    def to_frame(self, node_ids: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "vehicle_id": self.ids(),
            "vehicle_type": [VEHICLE_TYPES[t] for t in self.vehicle_type],
            "fuel_type": [FUEL_TYPES[t] for t in self.fuel_type],
            "fuel_efficiency_kmpl": self.kmpl,
            "driving_style": [DRIVING_STYLES[s] for s in self.driving_style],
            "average_speed_factor": self.speed_factor,
            "tank_capacity_l": self.tank_l,
            "highway_preference": self.highway_pref,
            "home_node": node_ids[self.home_node],
            "work_node": node_ids[self.work_node],
            "favorite_poi_1": node_ids[self.poi0],
            "favorite_poi_2": node_ids[self.poi1],
        })


def _sample_valid(valid: np.ndarray, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each row index pick a random column where ``valid[row, col]`` is True."""
    probs = valid[rows].astype(np.float64)
    dead = probs.sum(1) == 0
    probs[dead] = 1.0                       # (should not happen) fall back to uniform
    cum = np.cumsum(probs, axis=1)
    u = rng.random(len(rows)) * cum[:, -1]
    return np.minimum((cum < u[:, None]).sum(1), valid.shape[1] - 1)


def generate_fleet(n: int, catalog: RouteCatalog, vcfg: VehicleConfig, seed: int) -> VehicleFleet:
    """Generate ``n`` vehicle profiles (fully vectorised; ~ms for 100K)."""
    rng = make_rng(seed, 3)
    specs = [VEHICLE_TYPE_SPECS[t] for t in VEHICLE_TYPES]
    vtype = rng.choice(len(specs), n, p=np.array([s.share for s in specs]) / sum(s.share for s in specs)).astype(np.int8)

    fuel = np.empty(n, dtype=np.int8)
    for code, spec in enumerate(specs):
        idx = np.flatnonzero(vtype == code)
        shares = np.array(spec.fuel_shares) / sum(spec.fuel_shares)
        fuel[idx] = rng.choice(len(FUEL_TYPES), len(idx), p=shares)

    mean = np.array([s.kmpl_mean for s in specs])[vtype]
    sd = np.array([s.kmpl_sd for s in specs])[vtype]
    lo = np.array([s.kmpl_min for s in specs])[vtype]
    hi = np.array([s.kmpl_max for s in specs])[vtype]
    kmpl = np.clip(rng.normal(mean, sd), lo, hi)
    kmpl = kmpl * np.where(fuel == FUEL_DIESEL, DIESEL_EFFICIENCY_BONUS,
                           np.where(fuel == FUEL_HYBRID, HYBRID_EFFICIENCY_BONUS, 1.0))
    kmpl = np.clip(kmpl, *KMPL_GLOBAL_RANGE)

    tank = np.maximum(20.0, np.array([s.tank_l for s in specs])[vtype] + rng.normal(0, TANK_SD_L, n))
    style = rng.choice(len(DRIVING_STYLES), n, p=STYLE_SHARES).astype(np.int8)
    speed_factor = np.clip(rng.normal(1.0, SPEED_FACTOR_SD, n), *SPEED_FACTOR_RANGE)

    pools = catalog.pools
    h_idx = rng.integers(0, len(pools.home), n)
    w_idx = _sample_valid(catalog.valid_work, h_idx, rng)
    p0 = _sample_valid(catalog.valid_poi, h_idx, rng)
    p1 = _sample_valid(catalog.valid_poi, h_idx, rng)

    return VehicleFleet(
        n=n, vehicle_type=vtype, fuel_type=fuel, driving_style=style, kmpl=kmpl,
        speed_factor=speed_factor, tank_l=tank,
        highway_pref=rng.normal(0, HIGHWAY_PREF_SD, n),
        route_sigma=rng.uniform(*ROUTE_SIGMA_RANGE, n),
        wear=np.exp(rng.normal(0, DEFAULT_PARAMS.wear_sigma, n)),
        home_node=pools.home[h_idx], work_node=pools.work[w_idx],
        poi0=pools.poi[p0], poi1=pools.poi[p1], id_width=max(6, len(str(n))),
    )
