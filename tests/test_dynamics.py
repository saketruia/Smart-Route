import numpy as np

from simulator import dynamics as dyn


def _speed(level, style=1):
    return dyn.effective_speed(np.array([50.0]), np.array([60.0]), np.array([level]), np.array([1.0]),
                               np.array([style]), np.array([1.0]), np.array([0.0]))[0]


def test_congestion_reduces_speed():
    assert _speed(0.0) > _speed(0.5) > _speed(1.0) >= dyn.DEFAULT_PARAMS.min_speed_kmh


def test_aggressive_faster_than_calm():
    assert _speed(0.0, 2) > _speed(0.0, 0)


def test_travel_time_formula():
    assert np.isclose(dyn.travel_time_min(30.0, 60.0), 30.0)


def _fuel(speed, level, style=1, fuel_type=0):
    return dyn.fuel_litres(np.array([10.0]), np.array([15.0]), np.array([speed]), np.array([level]),
                           np.array([style]), np.array([fuel_type]), 1.0, np.array([0.0]))[0]


def test_fuel_baseline_and_penalties():
    assert np.isclose(_fuel(50, 0.0), 10 / 15, rtol=1e-6)
    assert _fuel(10, 0.9) > _fuel(50, 0.0)          # stop-and-go costs more
    assert _fuel(50, 0.0, style=2) > _fuel(50, 0.0, style=0)
    assert _fuel(10, 0.9, fuel_type=2) < _fuel(10, 0.9, fuel_type=0)   # hybrid relief


def test_vehicle_state_codes():
    s = dyn.vehicle_state_code(np.array([50.0, 25.0, 5.0]), np.array([50.0, 50.0, 50.0]))
    assert list(s) == [0, 1, 2]
