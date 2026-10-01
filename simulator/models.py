"""Shared constants, lookup tables and record types.

Categorical values are stored as small integer codes inside numpy arrays (cheap for
100K+ vehicles / millions of events) and mapped back to readable strings on output.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone

import numpy as np

# --------------------------------------------------------------------------- roads
ROAD_TYPES = ("residential", "commercial", "industrial", "arterial", "highway")
RT_RESIDENTIAL, RT_COMMERCIAL, RT_INDUSTRIAL, RT_ARTERIAL, RT_HIGHWAY = range(5)


@dataclass(frozen=True)
class RoadTypeSpec:
    speed_limit_kmh: float
    baseline_ratio: float   # typical free-flow speed as a fraction of the limit
    lanes_min: int
    lanes_max: int
    sensitivity: float      # how strongly demand turns into congestion


ROAD_TYPE_SPECS = {
    "residential": RoadTypeSpec(30, 0.85, 1, 1, 0.60),
    "commercial": RoadTypeSpec(40, 0.85, 1, 2, 1.00),
    "industrial": RoadTypeSpec(50, 0.95, 2, 2, 0.70),
    "arterial": RoadTypeSpec(60, 0.90, 2, 3, 1.10),
    "highway": RoadTypeSpec(80, 0.95, 3, 4, 1.15),
}


def _road_table(attr: str) -> np.ndarray:
    return np.array([getattr(ROAD_TYPE_SPECS[t], attr) for t in ROAD_TYPES], dtype=np.float64)


RT_SPEED_LIMIT = _road_table("speed_limit_kmh")
RT_BASELINE_RATIO = _road_table("baseline_ratio")
RT_LANES_MIN = _road_table("lanes_min").astype(int)
RT_LANES_MAX = _road_table("lanes_max").astype(int)
RT_SENSITIVITY = _road_table("sensitivity")

# Node zones (synthetic land use)
ZONE_SUBURB, ZONE_DOWNTOWN, ZONE_INDUSTRIAL = 0, 1, 2
ZONE_NAMES = ("suburb", "downtown", "industrial")

# Recurring congestion profiles
PROFILE_NAMES = ("LOW", "MORNING_HEAVY", "EVENING_HEAVY", "MODERATE", "COMMERCIAL")
P_LOW, P_MORNING, P_EVENING, P_MODERATE, P_COMMERCIAL = range(5)

# ------------------------------------------------------------------------ vehicles
VEHICLE_TYPES = ("sedan", "hatchback", "SUV", "pickup", "van")
FUEL_TYPES = ("petrol", "diesel", "hybrid")
FUEL_PETROL, FUEL_DIESEL, FUEL_HYBRID = range(3)
DRIVING_STYLES = ("calm", "normal", "aggressive")
STYLE_CALM, STYLE_NORMAL, STYLE_AGGRESSIVE = range(3)


@dataclass(frozen=True)
class VehicleTypeSpec:
    share: float
    kmpl_mean: float
    kmpl_sd: float
    kmpl_min: float
    kmpl_max: float
    tank_l: float
    fuel_shares: tuple[float, float, float]  # petrol, diesel, hybrid


VEHICLE_TYPE_SPECS = {
    "sedan": VehicleTypeSpec(0.34, 15.5, 2.0, 12.0, 20.0, 45.0, (0.60, 0.25, 0.15)),
    "hatchback": VehicleTypeSpec(0.28, 19.0, 2.5, 15.0, 24.0, 38.0, (0.72, 0.18, 0.10)),
    "SUV": VehicleTypeSpec(0.22, 11.0, 1.8, 8.0, 15.0, 60.0, (0.35, 0.55, 0.10)),
    "pickup": VehicleTypeSpec(0.06, 9.5, 1.2, 7.0, 13.0, 70.0, (0.20, 0.80, 0.00)),
    "van": VehicleTypeSpec(0.10, 10.5, 1.5, 8.0, 14.0, 65.0, (0.25, 0.75, 0.00)),
}
DIESEL_EFFICIENCY_BONUS = 1.10
HYBRID_EFFICIENCY_BONUS = 1.30

# Driving-style behaviour tables, indexed by style code.
STYLE_SPEED_FACTOR = np.array([0.94, 1.00, 1.07])    # pace relative to a normal driver
STYLE_SPEED_SIGMA = np.array([0.03, 0.05, 0.09])     # speed variability (log-normal sigma)
STYLE_FUEL_PENALTY = np.array([0.93, 1.00, 1.18])    # fuel multiplier
STYLE_STOP_GO = np.array([0.8, 1.0, 1.4])            # stop-and-go fuel penalty multiplier
STYLE_RESPONSE_S = np.array([15.0, 10.0, 6.0])       # live: speed-adjustment time constant

# ---------------------------------------------------------------------------- trips
TRIP_KINDS = ("COMMUTE_TO_WORK", "COMMUTE_HOME", "LEISURE_OUT", "LEISURE_RETURN")
KIND_TO_WORK, KIND_HOME, KIND_LEISURE_OUT, KIND_LEISURE_RETURN = range(4)

# ------------------------------------------------------------------------ telemetry
VEHICLE_STATES = ("CRUISING", "SLOW", "STOP_AND_GO")
EVENT_KINDS = ("SEGMENT_COMPLETE", "PERIODIC")
EVENT_SEGMENT_COMPLETE, EVENT_PERIODIC = range(2)
STATE_CRUISING_RATIO = 0.75   # speed / free-flow baseline thresholds
STATE_SLOW_RATIO = 0.35


@dataclass
class TelemetryEvent:
    """One telemetry event.

    ``timestamp`` is when the measurement happened (event time); ``emitted_at`` is when the
    vehicle / gateway sent it (arrival time).  With out-of-order simulation the two differ
    and ``emitted_at`` may be non-monotonic relative to ``timestamp``.

    ``distance_km``, ``travel_time_min`` and ``fuel_consumed_l`` describe the observation
    window that ends at ``timestamp``:

    * ``SEGMENT_COMPLETE`` (historical data): the full traversal of ``segment_id``.
    * ``PERIODIC`` (live data): the interval since the vehicle's previous event.
    """

    event_id: str
    vehicle_id: str
    timestamp: str
    emitted_at: str | None
    trip_id: str
    segment_id: str
    segment_seq: int
    latitude: float
    longitude: float
    heading_deg: float
    speed_kmh: float
    traffic_level: float
    fuel_consumed_l: float
    distance_km: float
    travel_time_min: float
    fuel_level_pct: float
    vehicle_state: str
    event_kind: str

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


TELEMETRY_FIELDS = tuple(f.name for f in fields(TelemetryEvent))


def iso_utc(epoch_seconds: float) -> str:
    """ISO-8601 UTC string with millisecond precision, e.g. ``2025-01-06T08:15:03.250Z``."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
