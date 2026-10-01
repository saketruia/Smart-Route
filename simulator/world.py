"""Assemble all static simulator components from one configuration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .config import SimulatorConfig
from .dynamics import DEFAULT_PARAMS, DynamicsParams
from .rng import make_rng
from .road_network import RoadNetwork, generate_road_network
from .routes import RouteCatalog, build_route_catalog
from .traffic import TrafficEngine
from .vehicles import VehicleFleet, generate_fleet

log = logging.getLogger(__name__)


@dataclass
class World:
    cfg: SimulatorConfig
    network: RoadNetwork
    catalog: RouteCatalog
    fleet: VehicleFleet
    traffic: TrafficEngine
    dynamics: DynamicsParams = DEFAULT_PARAMS


def build_world(cfg: SimulatorConfig, n_vehicles: int | None = None) -> World:
    """Road network -> route catalog -> vehicle fleet -> traffic engine (all seeded)."""
    t0 = time.perf_counter()
    network = generate_road_network(cfg.network, cfg.traffic, make_rng(cfg.seed, 1))
    catalog = build_route_catalog(network, cfg.vehicles, cfg.seed)
    fleet = generate_fleet(n_vehicles or cfg.vehicles.count, catalog, cfg.vehicles, cfg.seed)
    traffic = TrafficEngine(network, cfg.traffic, cfg.seed)
    log.info("World built in %.1fs: %d nodes, %d segments, %d corridors, %d routes, %d vehicles",
             time.perf_counter() - t0, network.n_nodes, network.n_segments, catalog.n_od,
             catalog.n_routes, fleet.n)
    return World(cfg, network, catalog, fleet, traffic)
