import numpy as np

from simulator.trips import plan_trips


def _plan(world, cfg, chunk=0):
    return plan_trips(world.fleet, 0, 50, world.catalog, cfg.historical, cfg.vehicles, cfg.seed, chunk)


def test_trip_count_and_valid_routes(world, cfg):
    p = _plan(world, cfg)
    assert len(p) == 50 * cfg.historical.trips_per_vehicle
    assert (p.route >= 0).all() and (world.catalog.route_od[p.route] == p.od).all()
    assert (p.origin != p.dest).all()


def test_multiple_routes_per_corridor(world):
    ids = world.catalog.od_route_ids
    assert ((ids >= 0).sum(axis=1) >= 2).all()


def test_routes_connect_origin_to_destination(world):
    c, n = world.catalog, world.network
    for r in np.random.default_rng(0).integers(0, c.n_routes, 40):
        segs = c.segments_of(r)
        assert n.seg_from[segs[0]] == c.od_origin[c.route_od[r]]
        assert n.seg_to[segs[-1]] == c.od_dest[c.route_od[r]]
        assert (n.seg_to[segs[:-1]] == n.seg_from[segs[1:]]).all()


def test_drivers_use_alternative_routes(world, cfg):
    p = _plan(world, cfg)
    assert (p.route != world.catalog.od_best_ff_route[p.od]).mean() > 0.1


def test_plans_are_deterministic(world, cfg):
    a, b = _plan(world, cfg), _plan(world, cfg)
    assert np.array_equal(a.route, b.route) and np.array_equal(a.start_s, b.start_s)
