import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.config import SimulatorConfig  # noqa: E402
from simulator.world import build_world  # noqa: E402

TEST_VEHICLES = 300


@pytest.fixture(scope="session")
def cfg():
    return SimulatorConfig().override(vehicles__count=TEST_VEHICLES, historical__trips_per_vehicle=4,
                                      historical__days=7, historical__chunk_vehicles=150)


@pytest.fixture(scope="session")
def world(cfg):
    return build_world(cfg)
