import pytest
from framework.intermediate_representation import (
    IntermediateRepresentation,
    Vertex,
    VertexID,
    Charger,
    ChargerID,
    Arc,
    VehicleID,
    Route,
    Stop,
)
from framework.preprocessing import simplify_intermediate_repr, preprocess_time_windows_in_inter_rep
from framework.utils import *


@pytest.fixture
def test_routes() -> Tuple[List[Route], List[Route]]:
    return ([
        Route(
            stop_sequence=[
                Stop(
                    vertex_id=VertexID("0"),
                    stopover_time=0.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=27.0,
                ),
                Stop(
                    vertex_id=VertexID("1"),
                    stopover_time=0.0,
                    earliest_time_of_service=50.0,
                    latest_time_of_service=100.0,
                ),
                Stop(
                    vertex_id=VertexID("4"),
                    stopover_time=0.0,
                    earliest_time_of_service=90.0,
                    latest_time_of_service=200.0,
                ),
                Stop(
                    vertex_id=VertexID("0"),
                    stopover_time=0.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=220.0,
                ),
            ],
            vehicle_id=VehicleID("0"),
        )
    ],
    [
        Route(
            stop_sequence=[
                Stop(
                    vertex_id=VertexID("0"),
                    stopover_time=0.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=27.0,
                ),
                Stop(
                    vertex_id=VertexID("1"),
                    stopover_time=0.0,
                    earliest_time_of_service=50.0,
                    latest_time_of_service=100.0,
                ),
                Stop(
                    vertex_id=VertexID("4"),
                    stopover_time=0.0,
                    earliest_time_of_service=90.0,
                    latest_time_of_service=200.0,
                ),
                Stop(
                    vertex_id=VertexID("0"),
                    stopover_time=0.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=220.0,
                ),
            ],
            vehicle_id=VehicleID("0"),
        ),
        Route(
            stop_sequence=[
                Stop(
                    vertex_id=VertexID("0"),
                    stopover_time=0.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=27.0,
                ),
                Stop(
                    vertex_id=VertexID("1"),
                    stopover_time=0.0,
                    earliest_time_of_service=50.0,
                    latest_time_of_service=100.0,
                ),
                Stop(
                    vertex_id=VertexID("4"),
                    stopover_time=0.0,
                    earliest_time_of_service=90.0,
                    latest_time_of_service=200.0,
                ),
                Stop(
                    vertex_id=VertexID("0"),
                    stopover_time=0.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=220.0,
                ),
            ],
            vehicle_id=VehicleID("1"),
        )
    ])


@pytest.fixture
def test_intermediate_representation(test_routes) -> IntermediateRepresentation:
    c = Coordinate(0, 0)
    star_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                        segment_construction_cost=0)
    dyn_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                       segment_construction_cost=1)
    vertices = [
        Vertex(id=VertexID("0"), is_depot=True, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("1"), is_depot=False, is_stop=True, coordinate=c, constructible_charger={
            star_c
        }),
        Vertex(id=VertexID("2"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("3"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("4"), is_depot=False, is_stop=True, coordinate=c, constructible_charger=set()),
    ]
    arcs = {
        (VertexID("0"), VertexID("1")): Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("0"), VertexID("2")): Arc(
            distance=0.5, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("2"), VertexID("3")): Arc(
            distance=0.5, speed_limit=100, constructible_chargers={dyn_c}
        ),
        (VertexID("3"), VertexID("1")): Arc(
            distance=0.5, speed_limit=150, constructible_chargers=set()
        ),
        (VertexID("1"), VertexID("4")): Arc(
            distance=1.0, speed_limit=35, constructible_chargers=set()
        ),
        (VertexID("1"), VertexID("2")): Arc(
            distance=0.75, speed_limit=35, constructible_chargers=set()
        ),
        (VertexID("2"), VertexID("4")): Arc(
            distance=0.75, speed_limit=45, constructible_chargers=set()
        ),
        (VertexID("4"), VertexID("0")): Arc(
            distance=0.001, speed_limit=45, constructible_chargers=set()
        ),
    }
    return IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=test_routes[0])

@pytest.fixture
def test_intermediate_representation_two_vehicles(test_routes) -> IntermediateRepresentation:
    c = Coordinate(0, 0)
    star_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                        segment_construction_cost=0)
    dyn_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                       segment_construction_cost=1)
    vertices = [
        Vertex(id=VertexID("0"), is_depot=True, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("1"), is_depot=False, is_stop=True, coordinate=c, constructible_charger={
            star_c
        }),
        Vertex(id=VertexID("2"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("3"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("4"), is_depot=False, is_stop=True, coordinate=c, constructible_charger=set()),
    ]
    arcs = {
        (VertexID("0"), VertexID("1")): Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("0"), VertexID("2")): Arc(
            distance=0.5, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("2"), VertexID("3")): Arc(
            distance=0.5, speed_limit=100, constructible_chargers={dyn_c}
        ),
        (VertexID("3"), VertexID("1")): Arc(
            distance=0.5, speed_limit=150, constructible_chargers=set()
        ),
        (VertexID("1"), VertexID("4")): Arc(
            distance=1.0, speed_limit=35, constructible_chargers=set()
        ),
        (VertexID("1"), VertexID("2")): Arc(
            distance=0.75, speed_limit=35, constructible_chargers=set()
        ),
        (VertexID("2"), VertexID("4")): Arc(
            distance=0.75, speed_limit=45, constructible_chargers=set()
        ),
        (VertexID("4"), VertexID("0")): Arc(
            distance=0.001, speed_limit=45, constructible_chargers=set()
        ),
    }
    return IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=test_routes[1])

@pytest.fixture
def expected_result_no_deviation(test_routes) -> IntermediateRepresentation:
    c = Coordinate(0,0)
    star_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                        segment_construction_cost=0)
    dyn_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                       segment_construction_cost=1)
    vertices = [
        Vertex(id=VertexID("0"), is_depot=True, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("1"), is_depot=False, is_stop=True, coordinate=c, constructible_charger={
            star_c
        }),
        Vertex(id=VertexID("2"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("3"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("4"), is_depot=False, is_stop=True, coordinate=c, constructible_charger=set()),
    ]
    arcs = {
        (VertexID("0"), VertexID("2")): Arc(
            distance=0.5, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("2"), VertexID("3")): Arc(
            distance=0.5, speed_limit=100, constructible_chargers={dyn_c}
        ),
        (VertexID("3"), VertexID("1")): Arc(
            distance=0.5, speed_limit=150, constructible_chargers=set()
        ),
        (VertexID("1"), VertexID("4")): Arc(
            distance=1.0, speed_limit=35, constructible_chargers=set()
        ),
        (VertexID("4"), VertexID("0")): Arc(
            distance=0.001, speed_limit=45, constructible_chargers=set()
        ),
        (VertexID("0"), VertexID("1")): Arc(
            distance=1.5, speed_limit=100, constructible_chargers=set()
        )
    }
    return IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=test_routes[0])


@pytest.fixture
def expected_result_deviation(test_routes) -> IntermediateRepresentation:
    c = Coordinate(0,0)
    star_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                        segment_construction_cost=0)
    dyn_c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
                       segment_construction_cost=1)
    vertices = [
        Vertex(id=VertexID("0"), is_depot=True, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("1"), is_depot=False, is_stop=True, coordinate=c, constructible_charger={
            star_c
        }),
        Vertex(id=VertexID("2"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("3"), is_depot=False, is_stop=False, coordinate=c, constructible_charger=set()),
        Vertex(id=VertexID("4"), is_depot=False, is_stop=True, coordinate=c, constructible_charger=set()),
    ]
    arcs = {
        (VertexID("0"), VertexID("2")): Arc(
            distance=0.5, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("2"), VertexID("3")): Arc(
            distance=0.5, speed_limit=100, constructible_chargers={dyn_c}
        ),
        (VertexID("3"), VertexID("1")): Arc(
            distance=0.5, speed_limit=150, constructible_chargers=set()
        ),
        (VertexID("1"), VertexID("4")): Arc(
            distance=1.0, speed_limit=35, constructible_chargers=set()
        ),
        (VertexID("4"), VertexID("0")): Arc(
            distance=0.001, speed_limit=45, constructible_chargers=set()
        ),
        # detour to static charger before continuing route
        # this arc cannot be concatenated because there is dyn segment in the middle
        # (VertexID("4"), VertexID("1")): Arc(
        #     distance=1.0, speed_limit=35, constructible_chargers=set()
        # ),
        (VertexID("1"), VertexID("0")): Arc(
            distance=1.001, speed_limit=35.00999000999001, constructible_chargers=set()
        ),
        # detour to dyn charger before continuing route
        (VertexID("4"), VertexID("2")): Arc(
            distance=0.501, speed_limit=49.99001996007984, constructible_chargers=set()
        ),
        (VertexID("3"), VertexID("0")): Arc(
            distance=1.501, speed_limit=73.31445702864757, constructible_chargers=set()
        ),
        (VertexID("1"), VertexID("2")): Arc(
            distance=0.75, speed_limit=35, constructible_chargers=set()
        ),
        (VertexID("3"), VertexID("4")): Arc(
            distance=1.5, speed_limit=73.33333333333333, constructible_chargers=set()
        ),
        # direct connections (additional to directions via nodes/arcs on shortest path)
        (VertexID("0"), VertexID("1")): Arc(
            distance=1.5, speed_limit=50.0, constructible_chargers=set()
        ),
        # direct connections
        (VertexID("4"), VertexID("1")): Arc(
            distance=1.501, speed_limit=37.511250000000004, constructible_chargers=set()
        ),
    }
    return IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=test_routes[0])


def test_simplify_intermediate_representation_no_deviation(
    test_intermediate_representation, expected_result_no_deviation
):
    result = simplify_intermediate_repr(test_intermediate_representation, allow_path_deviation=False)
    assert result == expected_result_no_deviation


def test_simplify_intermediate_representation_deviation(
    test_intermediate_representation, expected_result_deviation
):
    result = simplify_intermediate_repr(test_intermediate_representation, allow_path_deviation=True)
    assert result == expected_result_deviation


@pytest.mark.parametrize(
    "max_speed,expected_value_end",
    [
        ("150", "27.0"),
        ("101", "27.0"),
        ("90", "27.0"),
        ("70", "19.0"),
        ("0.0001", "None") # should raise a ValueError
    ]
)
def test_time_window_preprocessing(test_intermediate_representation, max_speed, expected_value_end):
    if expected_value_end == "None":
        with pytest.raises(ValueError) as exc_info:
            _ = preprocess_time_windows_in_inter_rep(test_intermediate_representation, float(max_speed))
        assert str(exc_info.value) == "Route 0 is time infeasible"
        return
    test_ir_simplified = simplify_intermediate_repr(test_intermediate_representation,False)
    ir_test = preprocess_time_windows_in_inter_rep(test_ir_simplified, float(max_speed))
    assert ir_test.routes[0].stop_sequence[0].latest_time_of_service == pytest.approx(float(expected_value_end))


def _extend_stop_sequence(original: list[Stop], num_rounds: int, offset_min:int)->list[Stop]:
    cntr = 0
    new = original[:-1]
    while cntr < num_rounds-1:
        cntr+=1
        for s in range(1,len(original)):
            new_stop = Stop(
                vertex_id=original[s].vertex_id,
                stopover_time=original[s].stopover_time,
                earliest_time_of_service=original[s].earliest_time_of_service + original[-1].latest_time_of_service*cntr + offset_min*60,
                latest_time_of_service=original[s].latest_time_of_service + original[-1].latest_time_of_service*cntr + offset_min*60,
            )
            new = new + [new_stop]
    return new