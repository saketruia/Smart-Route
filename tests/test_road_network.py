import networkx as nx
import numpy as np

from simulator.models import ROAD_TYPES


def test_network_is_strongly_connected(world):
    n = world.network
    g = nx.DiGraph(list(zip(n.seg_from.tolist(), n.seg_to.tolist())))
    assert g.number_of_nodes() == n.n_nodes
    assert nx.is_strongly_connected(g)


def test_segments_have_valid_attributes(world):
    n = world.network
    assert len(set(n.seg_ids)) == n.n_segments
    assert (n.seg_distance_km > 0).all() and (n.seg_speed_limit > 0).all()
    assert (n.seg_baseline_speed <= n.seg_speed_limit).all()
    assert (n.seg_lanes >= 1).all()
    assert set(np.unique(n.seg_road_type)) <= set(range(len(ROAD_TYPES)))
    assert ((n.seg_heading_deg >= 0) & (n.seg_heading_deg < 360)).all()


def test_two_directions_per_road(world):
    n = world.network
    pairs = set(zip(n.seg_from.tolist(), n.seg_to.tolist()))
    assert all((b, a) in pairs for a, b in pairs)


def test_problem_roads_exist(world):
    assert 0 < world.network.problem_mask.sum() < world.network.n_segments * 0.1
