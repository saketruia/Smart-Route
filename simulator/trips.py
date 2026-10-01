"""Trip planning: when each vehicle travels, where, and which candidate route it takes.

Each vehicle has habitual corridors (home<->work, home<->favourite places).  Trips fill
(day, daypart) slots: weekday mornings/evenings are mostly commutes, other slots are
leisure.  Route choice mixes a *stable personal habit* (a per vehicle-per-corridor
preference over the candidate routes) with occasional exploration, so a driver ends up
with e.g. "Route A 45%, Route B 35%, Route C 20%" - often not the best route.

Everything is vectorised over a block of vehicles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HistoricalConfig, VehicleConfig
from .models import (
    KIND_HOME, KIND_LEISURE_OUT, KIND_LEISURE_RETURN, KIND_TO_WORK,
)
from .rng import hash_normal, make_rng
from .routes import MAX_ROUTES_PER_OD, RouteCatalog
from .traffic import EPOCH_WEEKDAY_OFFSET, SECONDS_PER_DAY
from .vehicles import VehicleFleet

PART_AM, PART_MIDDAY, PART_PM, PART_EVENING = range(4)
PARTS_PER_DAY = 4
# slot weights per part: weekday / weekend
SLOT_WEIGHTS = np.array([[1.0, 0.25, 1.0, 0.20], [0.20, 0.50, 0.35, 0.25]])
# departure-hour distribution per [weekend][part]: (mean, sd, lo, hi)
START_HOURS = np.array([
    [[8.0, 0.75, 6.5, 9.75], [13.0, 1.2, 11.5, 15.5], [18.0, 0.8, 16.5, 19.75], [21.3, 0.5, 20.25, 22.5]],
    [[9.75, 1.0, 8.0, 11.25], [13.0, 1.2, 11.5, 15.5], [17.5, 1.0, 16.0, 19.5], [21.3, 0.5, 20.25, 22.5]],
])
FAV_POI_1_PROB, FAV_POI_2_PROB = 0.45, 0.25     # remaining 30%: any valid POI
ROUTE_DIST_PENALTY, ROUTE_TIME_PENALTY = 6.0, 4.0
_S_ROUTE_HABIT = 7


@dataclass
class TripPlan:
    """Planned trips for a block of vehicles (arrays of equal length ``n``)."""
    trip_no: np.ndarray        # global 1-based trip number
    vehicle_idx: np.ndarray
    kind: np.ndarray
    origin: np.ndarray
    dest: np.ndarray
    od: np.ndarray
    route: np.ndarray          # global route index in the catalog
    start_s: np.ndarray        # epoch seconds

    def __len__(self) -> int:
        return len(self.trip_no)


def start_epoch_day(start_date: str) -> int:
    return int((np.datetime64(start_date) - np.datetime64("1970-01-01")).astype("timedelta64[D]").astype(int))


def choose_poi(fleet: VehicleFleet, catalog: RouteCatalog, veh: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick a leisure destination node per vehicle (favourites preferred, always a valid corridor)."""
    u = rng.random(len(veh))
    home_idx = np.searchsorted(catalog.pools.home, fleet.home_node[veh])
    any_poi_idx = rng.integers(0, len(catalog.pools.poi), len(veh))
    any_poi = catalog.pools.poi[any_poi_idx]
    ok = catalog.valid_poi[home_idx, any_poi_idx]
    any_poi = np.where(ok, any_poi, fleet.poi0[veh])
    return np.where(u < FAV_POI_1_PROB, fleet.poi0[veh],
                    np.where(u < FAV_POI_1_PROB + FAV_POI_2_PROB, fleet.poi1[veh], any_poi))


def choose_routes(catalog: RouteCatalog, fleet: VehicleFleet, veh: np.ndarray, od: np.ndarray,
                  seed: int, exploration: float, rng: np.random.Generator) -> np.ndarray:
    """Pick a candidate route per trip (habit + exploration). Returns global route indices."""
    cand = catalog.od_route_ids[od]                       # [T, K]
    mask = cand >= 0
    k = np.arange(MAX_ROUTES_PER_OD)[None, :]
    habit = hash_normal(seed, _S_ROUTE_HABIT, veh[:, None], od[:, None], k)
    logits = (-ROUTE_DIST_PENALTY * catalog.od_dist_rel[od] - ROUTE_TIME_PENALTY * catalog.od_time_rel[od]
              + fleet.highway_pref[veh][:, None] * catalog.od_highway[od]
              + fleet.route_sigma[veh][:, None] * habit)
    logits = np.where(mask, logits, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs = (1 - exploration) * probs + exploration * mask / mask.sum(axis=1, keepdims=True)
    choice = (np.cumsum(probs, axis=1) < rng.random(len(veh))[:, None]).sum(axis=1)
    choice = np.minimum(choice, MAX_ROUTES_PER_OD - 1)
    return cand[np.arange(len(veh)), choice]


def plan_trips(fleet: VehicleFleet, v_start: int, v_stop: int, catalog: RouteCatalog,
               hcfg: HistoricalConfig, vcfg: VehicleConfig, seed: int, chunk_id: int) -> TripPlan:
    """Plan ``trips_per_vehicle`` trips for vehicles ``[v_start, v_stop)``."""
    n_days, tpv = hcfg.days, hcfg.trips_per_vehicle
    n_slots = n_days * PARTS_PER_DAY
    if tpv > n_slots:
        raise ValueError(f"trips_per_vehicle={tpv} exceeds available slots ({n_slots}); use more days")
    rng = make_rng(seed, 4, chunk_id)
    n_veh = v_stop - v_start
    day0 = start_epoch_day(hcfg.start_date)

    # Distinct (day, part) slots per vehicle via Gumbel top-k (weighted, no replacement).
    slot = np.arange(n_slots)
    day_of, part_of = slot // PARTS_PER_DAY, slot % PARTS_PER_DAY
    weekend_of = ((day0 + day_of + EPOCH_WEEKDAY_OFFSET) % 7) >= 5
    weights = SLOT_WEIGHTS[weekend_of.astype(int), part_of]
    keys = np.log(weights)[None, :] + rng.gumbel(size=(n_veh, n_slots))
    top = np.argpartition(-keys, tpv - 1, axis=1)[:, :tpv]
    slots = np.sort(top, axis=1).ravel()

    veh = np.repeat(np.arange(v_start, v_stop), tpv)
    trip_no = (veh * tpv + np.tile(np.arange(tpv), n_veh) + 1).astype(np.int64)
    day, part, weekend = day_of[slots], part_of[slots], weekend_of[slots]

    p = START_HOURS[weekend.astype(int), part]
    hour = np.clip(rng.normal(p[:, 0], p[:, 1]), p[:, 2], p[:, 3])
    start_s = (day0 + day) * SECONDS_PER_DAY + hour * 3600.0

    kind = np.where(~weekend & (part == PART_AM), KIND_TO_WORK,
           np.where(~weekend & (part == PART_PM), KIND_HOME,
           np.where(np.isin(part, (PART_AM, PART_MIDDAY)), KIND_LEISURE_OUT, KIND_LEISURE_RETURN))).astype(np.int8)

    home, work = fleet.home_node[veh], fleet.work_node[veh]
    poi = choose_poi(fleet, catalog, veh, rng)
    origin = np.select([kind == KIND_TO_WORK, kind == KIND_HOME, kind == KIND_LEISURE_OUT], [home, work, home], poi)
    dest = np.select([kind == KIND_TO_WORK, kind == KIND_HOME, kind == KIND_LEISURE_OUT], [work, home, poi], home)
    od = catalog.od_lookup[origin, dest]
    if (od < 0).any():
        raise RuntimeError("Planned a trip on a corridor missing from the route catalog")
    route = choose_routes(catalog, fleet, veh, od, seed, vcfg.exploration_rate, rng)
    return TripPlan(trip_no, veh.astype(np.int64), kind, origin.astype(np.int32), dest.astype(np.int32),
                    od.astype(np.int32), route.astype(np.int32), start_s)
