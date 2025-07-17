import pytest

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)

from iterative_local_search.spprc_network import *
from math_programming_model.network import *
from framework.intermediate_representation import *
from framework.preprocessing import *
from iterative_local_search.subproblem import *
import framework.intermediate_representation as ir
from framework.preprocessing import *

time_step = 5
max_run_time = 5
c = Charger(ChargerID("Charger1"), charging_rate=30000, transformer_construction_cost=10,
            segment_construction_cost=1)


@pytest.fixture
def test_routes() -> List[Route]:
    return [
        Route(
            stop_sequence=[
                Stop(
                    vertex_id=VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=1.0,
                ),
                Stop(
                    vertex_id=VertexID("S1"),
                    stopover_time=1.0,
                    earliest_time_of_service=250,
                    latest_time_of_service=300,
                ),
                Stop(
                    vertex_id=VertexID("C1"),
                    stopover_time=1.0,
                    earliest_time_of_service=250,
                    latest_time_of_service=1200,
                ),
                Stop(
                    vertex_id=VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=1000,
                    latest_time_of_service=1200,
                ),
                Stop(
                    vertex_id=VertexID("S3"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=2270,
                ),
                Stop(
                    vertex_id=VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=2200,
                    latest_time_of_service=2700,
                ),
            ],
            vehicle_id=VehicleID("0"),
        )
    ]


@pytest.fixture
def test_intermediate_representation(test_routes) -> IntermediateRepresentation:
    coordinate = util.Coordinate(0.0, 0.0)

    vertices = [
        ir.Vertex(id=VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("C1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment2"), is_stop=False, is_depot=False, coordinate=coordinate),
    ]

    arcs = {
        (VertexID("Depot"), VertexID("S1")): ir.Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("S1"), VertexID("Segment1")): ir.Arc(
            distance=0.000001, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("Segment1"), VertexID("Segment2")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("Segment2"), VertexID("C1")): ir.Arc(
            distance=0.000001, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("C1"), VertexID("S2")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("S2"), VertexID("S3")): ir.Arc(
            distance=6.0, speed_limit=20, constructible_chargers=set()
        ),
        (VertexID("S3"), VertexID("Depot")): ir.Arc(
            distance=1.0, speed_limit=60, constructible_chargers=set()
        )
    }
    return IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=test_routes)


@pytest.fixture
def build_decision_variables_vertex_charger(test_intermediate_representation) -> DecisionVariables:
    intermediate_rep = test_intermediate_representation

    max_speed = 100
    max_consumption = 7

    soc_init = 100
    soc_max = 100
    soc_min = 30

    energy_prices = 1.0
    consumption_cost = 0.0
    intermediate_rep.get_vertex(VertexID("C1")).constructible_charger = {c}
    intermediate_rep_simplified = simplify_intermediate_repr(inter_rep=intermediate_rep, allow_path_deviation=False)
    decision_variables = DecisionVariables(intermediate_rep_simplified, {VertexID("C1"): c}, {}, max_speed, max_consumption,
                                           soc_init, soc_max,
                                           soc_min,
                                           energy_prices, consumption_cost)

    return decision_variables


@pytest.fixture
def build_decision_variables_arc_charger(test_intermediate_representation) -> DecisionVariables:
    intermediate_rep = test_intermediate_representation

    max_speed = 100
    max_consumption = 7

    soc_init = 100
    soc_max = 100
    soc_min = 30

    energy_prices = 1.0
    consumption_cost = 0.0
    intermediate_rep.get_arc(VertexID("Segment1"), VertexID("Segment2")).constructible_chargers = {c}
    intermediate_rep_simplified = simplify_intermediate_repr(inter_rep=intermediate_rep, allow_path_deviation=False)
    decision_variables = DecisionVariables(intermediate_rep_simplified, {}, {(VertexID("Segment1"), VertexID("Segment2")): c},
                                           max_speed, max_consumption, soc_init, soc_max, soc_min, energy_prices,
                                           consumption_cost)

    return decision_variables


def test_feasible_solution(build_decision_variables_vertex_charger):
    decision_variables = build_decision_variables_vertex_charger
    a = decision_spprc_network(
        decision_variables, decision_variables.intermediate_rep.routes[0], time_step
    )
    solution = A_star(a, decision_variables, 0, False, "python")

    lower_bounds = lower_energy_bounds(a, 0, decision_variables)
    for k,v in lower_bounds.items():
        assert round(v,0) in [7.0, 49.0, 63.0, 77.0, 105.0, 0.0]
    assert solution.consumed_energy == pytest.approx(15.0 * 7.0, 1e-3) # consumed energy is positive
    assert solution.consumed_energy == solution.itineraries[0].route[-1].accumulated_consumed_energy

    assert solution.recharged_energy == pytest.approx(41.666666666, 1e-3)
    assert solution.recharged_energy == solution.itineraries[0].route[-1].accumulated_charged_energy

    assert solution.routing_cost == pytest.approx(41.666666666, 1e-3)
    assert decision_variables.soc_init - solution.consumed_energy + solution.recharged_energy == pytest.approx(
        solution.itineraries[0].route[-1].soc, 1e-3
    )
    # Check that accumulated consumed/recharged energy is increasing
    for x, y in pairwise(solution.itineraries[0].route):
        assert x.accumulated_consumed_energy <= y.accumulated_consumed_energy
        assert x.accumulated_charged_energy <= y.accumulated_charged_energy


@pytest.mark.parametrize(
    "charging_rate, expected_value",
    [
        ("1235", "35.2857"), # In this case limit is not reached and we charge what we can get on the arc
    ]
)
def test_feasible_solution_with_arc_charger(build_decision_variables_arc_charger, charging_rate, expected_value):
    decision_variables = build_decision_variables_arc_charger
    for (_, _), v in decision_variables.arc_chargers.items():
        v.charging_rate=float(charging_rate)
    a = decision_spprc_network(
        decision_variables, decision_variables.intermediate_rep.routes[0], time_step
    )
    solution = A_star(a, decision_variables, 0, False, "python")
    lower_bounds = lower_energy_bounds(a, 0, decision_variables)
    for k,v in lower_bounds.items():
        assert round(v,0) in [7.0, 49.0, 63.0, 77.0, 105.0, 0.0]
    assert solution.consumed_energy == pytest.approx(15.0 * 7.0, 1e-3)
    assert solution.consumed_energy == solution.itineraries[0].route[-1].accumulated_consumed_energy

    assert solution.recharged_energy == pytest.approx(float(expected_value), 1e-3)
    assert solution.recharged_energy == solution.itineraries[0].route[-1].accumulated_charged_energy

    assert decision_variables.soc_init - solution.consumed_energy + solution.recharged_energy == pytest.approx(
        solution.itineraries[0].route[-1].soc, 1e-3
    )
    # Check that accumulated consumed/recharged energy is increasing
    for x, y in pairwise(solution.itineraries[0].route):
        assert x.accumulated_consumed_energy <= y.accumulated_consumed_energy
        assert x.accumulated_charged_energy <= y.accumulated_charged_energy


def test_infeasible_solution(build_decision_variables_vertex_charger):
    decision_variables = build_decision_variables_vertex_charger

    # Delete Charger
    decision_variables.vertex_chargers = {}

    a = decision_spprc_network(
        decision_variables, decision_variables.intermediate_rep.routes[0], time_step
    )

    assert len(a.route.stops) == 6

    cpp = False
    if cpp:
        with pytest.raises(ValueError) as exc_info:
            _ = A_star(a, decision_variables, 0, False, "cpp")
    else:
        with pytest.raises(ValueError) as exc_info:
            _ = A_star(a, decision_variables, 0, False, "python")

    assert str(exc_info.value) == "No feasible solution found"
