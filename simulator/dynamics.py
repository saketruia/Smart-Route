"""Speed, travel-time and fuel models (vectorised, shared by historical and live modes).

The formulas are synthetic but internally consistent:

    speed  = baseline x traffic_factor(level) x vehicle_pace x style_pace x trip_mult x noise
    time   = distance / speed
    fuel   = distance / rated_kmpl x speed_penalty x stop_and_go x style_penalty x noise

so higher congestion -> lower speed -> longer time -> (generally) more fuel, while
independent noise terms keep it a genuine prediction problem rather than an exact formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import (
    FUEL_HYBRID, STYLE_FUEL_PENALTY, STYLE_SPEED_FACTOR, STYLE_SPEED_SIGMA, STYLE_STOP_GO,
    STATE_CRUISING_RATIO, STATE_SLOW_RATIO,
)


@dataclass(frozen=True)
class DynamicsParams:
    min_speed_kmh: float = 3.0
    max_speed_over_limit: float = 1.30
    traffic_speed_drop: float = 0.80        # speed fraction lost at severe congestion
    traffic_speed_exponent: float = 1.1
    speed_noise_additive_kmh: float = 1.0
    trip_speed_sigma: float = 0.04          # per-trip "mood"/pace multiplier
    trip_fuel_sigma: float = 0.03
    segment_fuel_sigma: float = 0.05
    wear_sigma: float = 0.05                # latent per-vehicle condition factor
    low_speed_ref_kmh: float = 35.0
    low_speed_fuel_coef: float = 0.90
    high_speed_ref_kmh: float = 85.0
    high_speed_fuel_coef: float = 0.012
    stop_go_fuel_coef: float = 0.30
    hybrid_low_speed_relief: float = 0.5
    min_fuel_l: float = 1e-6


DEFAULT_PARAMS = DynamicsParams()


def traffic_speed_factor(level: np.ndarray, p: DynamicsParams = DEFAULT_PARAMS) -> np.ndarray:
    """Fraction of free-flow speed retained at a given congestion level."""
    return 1.0 - p.traffic_speed_drop * np.clip(level, 0.0, 1.0) ** p.traffic_speed_exponent


def effective_speed(baseline_kmh, limit_kmh, level, vehicle_factor, style, trip_mult, noise_z,
                    p: DynamicsParams = DEFAULT_PARAMS) -> np.ndarray:
    """Vehicle speed (km/h) on a segment. ``noise_z`` is standard-normal noise (0 for none)."""
    speed = (baseline_kmh * traffic_speed_factor(level, p) * vehicle_factor * STYLE_SPEED_FACTOR[style]
             * trip_mult * np.exp(STYLE_SPEED_SIGMA[style] * noise_z)
             + p.speed_noise_additive_kmh * noise_z)
    speed = np.nan_to_num(speed, nan=p.min_speed_kmh, posinf=p.min_speed_kmh, neginf=p.min_speed_kmh)
    return np.clip(speed, p.min_speed_kmh, limit_kmh * p.max_speed_over_limit)


def travel_time_min(distance_km, speed_kmh) -> np.ndarray:
    return np.asarray(distance_km) / np.asarray(speed_kmh) * 60.0


def fuel_per_km_factor(speed_kmh, level, style, fuel_type, p: DynamicsParams = DEFAULT_PARAMS) -> np.ndarray:
    """Multiplier on the rated fuel use: low speed / stop-and-go / hard driving cost more."""
    low = np.maximum(0.0, (p.low_speed_ref_kmh - speed_kmh) / p.low_speed_ref_kmh) ** 1.5
    relief = np.where(fuel_type == FUEL_HYBRID, p.hybrid_low_speed_relief, 1.0)
    speed_pen = 1.0 + p.low_speed_fuel_coef * low * relief + p.high_speed_fuel_coef * np.maximum(
        0.0, speed_kmh - p.high_speed_ref_kmh)
    stop_go = 1.0 + p.stop_go_fuel_coef * np.clip(level, 0, 1) ** 2 * STYLE_STOP_GO[style]
    return speed_pen * stop_go * STYLE_FUEL_PENALTY[style]


def fuel_litres(distance_km, kmpl, speed_kmh, level, style, fuel_type, multiplier, noise_z,
                p: DynamicsParams = DEFAULT_PARAMS) -> np.ndarray:
    """Fuel used (litres) over ``distance_km``. ``multiplier`` folds in latent vehicle/trip factors."""
    fuel = (np.asarray(distance_km) / kmpl * fuel_per_km_factor(speed_kmh, level, style, fuel_type, p)
            * multiplier * np.exp(p.segment_fuel_sigma * noise_z))
    fuel = np.nan_to_num(fuel, nan=p.min_fuel_l, posinf=p.min_fuel_l)
    return np.maximum(fuel, p.min_fuel_l)


def vehicle_state_code(speed_kmh, baseline_kmh) -> np.ndarray:
    """0 = CRUISING, 1 = SLOW, 2 = STOP_AND_GO based on speed relative to free flow."""
    ratio = np.asarray(speed_kmh) / np.maximum(baseline_kmh, 1e-9)
    return np.where(ratio >= STATE_CRUISING_RATIO, 0, np.where(ratio >= STATE_SLOW_RATIO, 1, 2)).astype(np.int8)
