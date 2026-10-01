import numpy as np

from simulator.models import P_MORNING

DAY = 86400
MONDAY = int(np.datetime64("2025-01-06T00:00:00").astype("datetime64[s]").astype(int))


def _mean_level(world, hour, day=0, mask=None):
    seg = np.arange(world.network.n_segments) if mask is None else np.flatnonzero(mask)
    return world.traffic.level(seg, np.full(len(seg), MONDAY + day * DAY + hour * 3600.0)).mean()


def test_levels_in_range(world):
    seg = np.arange(world.network.n_segments)
    for h in (3, 8, 13, 18, 22):
        lvl = world.traffic.level(seg, np.full(len(seg), MONDAY + h * 3600.0))
        assert np.isfinite(lvl).all() and lvl.min() >= 0 and lvl.max() <= 1


def test_rush_hour_heavier_than_night(world):
    assert _mean_level(world, 18) > _mean_level(world, 3) + 0.1
    assert _mean_level(world, 8.5) > _mean_level(world, 3) + 0.1


def test_weekend_lighter_than_weekday_rush(world):
    assert _mean_level(world, 18, day=5) < _mean_level(world, 18, day=0)


def test_morning_profile_peaks_in_morning(world):
    m = world.network.seg_profile == P_MORNING
    assert m.any()
    assert _mean_level(world, 8.5, mask=m) > _mean_level(world, 18, mask=m)


def test_problem_roads_congested_in_window(world):
    n = world.network
    idx = np.flatnonzero(n.problem_mask)
    hour = (n.seg_problem_start[idx] + n.seg_problem_end[idx]) / 2
    lvl = world.traffic.level(idx, MONDAY + hour * 3600.0)
    assert lvl.mean() > 0.6


def test_traffic_is_deterministic(world):
    seg = np.arange(200)
    t = np.full(200, MONDAY + 8 * 3600.0)
    assert np.array_equal(world.traffic.level(seg, t), world.traffic.level(seg, t))
