import pytest
import networkx as nx
from typing import Tuple
from framework.utils import clean_osmnx_graph


def create_multigraph(parallel_edges_length: Tuple[float, float]) -> nx.MultiDiGraph:
    test_network = nx.MultiDiGraph()
    test_network.add_nodes_from(["A", "B", "C"])
    test_network.add_edges_from(
        [
            ("A", "B", dict(length=50.0)),
            ("B", "B", None),
            ("B", "C", dict(length=parallel_edges_length[0])),
            ("B", "C", dict(length=parallel_edges_length[1])),
        ]
    )
    return test_network


@pytest.mark.parametrize(
    "network, error, expected_n_edges, expected_length, key",
    [
        (create_multigraph((100.0, 80.0)), None, 2, 80.0, 1),
        (create_multigraph((100.0, 100.1)), None, 2, 100.0, 0),
        (create_multigraph((100.0, 100.0)), AssertionError, int, float, int),
    ],
)
def test_cleaning_of_osmnx_network(
    network, error, expected_n_edges, expected_length, key
) -> None:
    if error is None:
        clean_osmnx_graph(network)
        assert network.number_of_edges() == expected_n_edges
        assert network.edges["B", "C", key]["length"] == expected_length
    else:
        with pytest.raises(error):
            clean_osmnx_graph(network)
