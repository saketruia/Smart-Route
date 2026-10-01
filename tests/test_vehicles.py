import numpy as np

from simulator.models import VEHICLE_TYPE_SPECS, VEHICLE_TYPES


def test_fleet_size_and_ids(world, cfg):
    f = world.fleet
    assert f.n == cfg.vehicles.count
    ids = f.ids()
    assert len(set(ids)) == f.n and ids[0].startswith("V")


def test_fuel_efficiency_within_type_ranges(world):
    f = world.fleet
    for code, name in enumerate(VEHICLE_TYPES):
        spec = VEHICLE_TYPE_SPECS[name]
        k = f.kmpl[f.vehicle_type == code]
        if len(k):
            assert k.min() >= spec.kmpl_min * 0.99 and k.max() <= spec.kmpl_max * 1.31 * 1.001


def test_attributes_valid(world):
    f = world.fleet
    assert (f.tank_l > 0).all() and (f.kmpl > 0).all()
    assert set(np.unique(f.driving_style)) <= {0, 1, 2}
    assert (f.home_node != f.work_node).all()
    assert f.to_frame(world.network.node_ids).notna().all().all()
