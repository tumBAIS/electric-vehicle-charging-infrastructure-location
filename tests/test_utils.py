import pytest
from framework.utils import *


def test_sort_tuples():
    test_tuple = [(2, 4), (8, 3), (4, 5), (3, 2)]
    expected_result = [(8, 3), (3, 2), (2, 4), (4, 5)]
    assert sort_tuples(test_tuple) == expected_result


def test_filter_tuple_list():
    test_tuple = [(1, 2), (2, 3), (3, 4), (4, 5)]
    expected_result = [(2, 3), (3, 4)]
    expected_result2 = [(1, 2), (2, 3), (3, 4)]
    expected_result3 = [(2, 3), (3, 4), (4, 5)]
    assert filter_tuple_list(test_tuple, 2, 4) == expected_result
    assert filter_tuple_list(test_tuple, 1, 4) == expected_result2
    assert filter_tuple_list(test_tuple, 2, 5) == expected_result3


@pytest.mark.parametrize(
    "from_x, from_y, to_x, to_y, expected_value",
    [
        ("50.32001497940026", "11.897478177718405", "50.31763099746265", "11.924557759863472", "1940"),
        ("707114", "5577173", "707149", "5578634", "1460"),
    ]
)
def test_distance_computation(from_x, from_y, to_x, to_y, expected_value):
    from_ = Coordinate(float(from_x), float(from_y))
    to_ = Coordinate(float(to_x), float(to_y))
    assert pytest.approx(float(expected_value), abs=10) == distance_euclidean(from_, to_)
