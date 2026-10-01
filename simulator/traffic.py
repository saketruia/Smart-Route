"""Time-dependent traffic engine.

``TrafficEngine.level(segment_index, epoch_seconds)`` returns congestion in ``[0, 1]``
(0 free-flowing, 0.25 light, 0.5 moderate, 0.75 heavy, 1 severe).  It is a *pure function*
of ``(seed, segment, time)``: every vehicle sees the same congestion on a segment at the
same moment, in historical and live mode alike.

Level = demand(profile, time-of-day, weekday/weekend)      # predictable, learnable structure
        x segment sensitivity x per-day factor             # road character, "bad days"
        + smooth noise                                     # short-term variation
        + problem-road window                              # recurring severe jams
        + occasional incidents                             # unpredictable disruptions

The synthetic city clock is UTC: hour-of-day is computed from the UTC timestamp.
"""

from __future__ import annotations

import numpy as np

from .config import TrafficConfig
from .models import P_COMMERCIAL, P_EVENING, P_LOW, P_MODERATE, P_MORNING
from .rng import hash_normal, hash_uniform
from .road_network import RoadNetwork

SECONDS_PER_DAY = 86_400
EPOCH_WEEKDAY_OFFSET = 3           # 1970-01-01 was a Thursday (Monday = 0)
RAMP_H = 0.3                       # softness of rush-hour edges (hours)
DAYTIME = (6.0, 22.5)              # general activity window
COMMERCIAL_HOURS = (11.5, 21.0)
NOISE_BUCKET_S = 1800.0
DAY_FACTOR_SIGMA = 0.10
WEEKEND_MORNING_SCALE, WEEKEND_EVENING_SCALE = 0.20, 0.30
WEEKEND_PROBLEM_SCALE = 0.25
INCIDENT_START_H = (6.0, 22.0)
INCIDENT_DURATION_H = (0.5, 2.5)
INCIDENT_SEVERITY = (0.25, 0.60)
SOFT_CAP_START = 0.8

# Coefficients per profile: LOW, MORNING_HEAVY, EVENING_HEAVY, MODERATE, COMMERCIAL
BASE = np.array([0.05, 0.08, 0.08, 0.10, 0.07])
DAY = np.array([0.10, 0.08, 0.08, 0.22, 0.15])
MORNING = np.array([0.10, 0.60, 0.15, 0.25, 0.10])
EVENING = np.array([0.10, 0.15, 0.60, 0.25, 0.20])
MIDDAY = np.array([0.00, 0.00, 0.00, 0.05, 0.30])
WEEKEND_LEISURE = np.array([0.05, 0.08, 0.10, 0.18, 0.22])
assert len(BASE) == max(P_LOW, P_MORNING, P_EVENING, P_MODERATE, P_COMMERCIAL) + 1

# hash stream ids (keep distinct so noise components are independent)
_S_DAY, _S_NOISE, _S_INC_OCC, _S_INC_START, _S_INC_DUR, _S_INC_SEV = range(1, 7)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _plateau(hour: np.ndarray, start: float, end: float, ramp: float = RAMP_H) -> np.ndarray:
    return _sigmoid((hour - start) / ramp) * _sigmoid((end - hour) / ramp)


def _peak(hour: np.ndarray, start: float, end: float) -> np.ndarray:
    """Rush-hour plateau with a slightly higher centre."""
    centre, half = 0.5 * (start + end), 0.5 * (end - start)
    return _plateau(hour, start, end) * (0.75 + 0.25 * np.exp(-(((hour - centre) / (0.6 * half)) ** 2)))


def _soft_cap(x: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] with a smooth shoulder above ``SOFT_CAP_START``."""
    x = np.maximum(x, 0.0)
    return np.where(x < SOFT_CAP_START, x,
                    SOFT_CAP_START + (1 - SOFT_CAP_START) * np.tanh((x - SOFT_CAP_START) / (1 - SOFT_CAP_START)))


class TrafficEngine:
    def __init__(self, network: RoadNetwork, cfg: TrafficConfig, seed: int):
        self.net, self.cfg, self.seed = network, cfg, int(seed)

    @staticmethod
    def time_parts(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(day_index, hour_of_day, is_weekend)`` for epoch seconds."""
        t = np.asarray(t, dtype=np.float64)
        day = np.floor(t / SECONDS_PER_DAY).astype(np.int64)
        hour = (t - day * SECONDS_PER_DAY) / 3600.0
        weekend = ((day + EPOCH_WEEKDAY_OFFSET) % 7) >= 5
        return day, hour, weekend

    def demand(self, seg: np.ndarray, hour: np.ndarray, weekend: np.ndarray) -> np.ndarray:
        """Noise-free demand (before sensitivity)."""
        c, net = self.cfg, self.net
        p = net.seg_profile[seg]
        morning = _peak(hour, c.morning_start, c.morning_end) * np.where(weekend, WEEKEND_MORNING_SCALE, 1.0)
        evening = _peak(hour, c.evening_start, c.evening_end) * np.where(weekend, WEEKEND_EVENING_SCALE, 1.0)
        daytime = _plateau(hour, *DAYTIME, ramp=0.5)
        commercial = _plateau(hour, *COMMERCIAL_HOURS, ramp=0.5)
        d = (BASE[p] + DAY[p] * daytime + MORNING[p] * morning + EVENING[p] * evening
             + MIDDAY[p] * commercial + np.where(weekend, WEEKEND_LEISURE[p] * commercial, 0.0))
        strength = net.seg_problem_strength[seg]
        has_problem = strength > 0
        if has_problem.any():
            start = np.where(has_problem, net.seg_problem_start[seg], 0.0)
            end = np.where(has_problem, net.seg_problem_end[seg], 0.0)
            window = _plateau(hour, start, end, ramp=0.2)
            d = d + strength * window * np.where(weekend, WEEKEND_PROBLEM_SCALE, 1.0) * has_problem
        return d

    def expected_level(self, seg: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Congestion without any random component - the underlying learnable pattern."""
        seg = np.asarray(seg)
        _, hour, weekend = self.time_parts(t)
        return _soft_cap(self.demand(seg, hour, weekend) * self.net.seg_sensitivity[seg])

    def level(self, seg: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Congestion in [0, 1] including noise and incidents (deterministic per seg/time)."""
        seg = np.asarray(seg)
        t = np.asarray(t, dtype=np.float64)
        day, hour, weekend = self.time_parts(t)
        demand = self.demand(seg, hour, weekend)
        day_factor = np.clip(1.0 + DAY_FACTOR_SIGMA * hash_normal(self.seed, _S_DAY, seg, day), 0.75, 1.25)
        x = demand * self.net.seg_sensitivity[seg] * day_factor

        bucket = t / NOISE_BUCKET_S
        k = np.floor(bucket).astype(np.int64)
        frac = bucket - k
        noise = (hash_normal(self.seed, _S_NOISE, seg, k) * (1 - frac)
                 + hash_normal(self.seed, _S_NOISE, seg, k + 1) * frac)
        x = x + self.cfg.noise_sigma * (0.6 + demand) * noise

        rate = self.cfg.incident_rate_per_segment_day
        if rate > 0:
            occurs = hash_uniform(self.seed, _S_INC_OCC, seg, day) < rate
            if occurs.any():
                start = INCIDENT_START_H[0] + (INCIDENT_START_H[1] - INCIDENT_START_H[0]) * hash_uniform(self.seed, _S_INC_START, seg, day)
                dur = INCIDENT_DURATION_H[0] + (INCIDENT_DURATION_H[1] - INCIDENT_DURATION_H[0]) * hash_uniform(self.seed, _S_INC_DUR, seg, day)
                sev = INCIDENT_SEVERITY[0] + (INCIDENT_SEVERITY[1] - INCIDENT_SEVERITY[0]) * hash_uniform(self.seed, _S_INC_SEV, seg, day)
                x = x + occurs * sev * _plateau(hour, start, start + dur, ramp=0.1)
        return _soft_cap(x)
