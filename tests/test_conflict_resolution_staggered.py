import pytest

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)

from iterative_local_search.spprc_network import *
from math_programming_model.network import *
from iterative_local_search.subproblem import A_star
from framework.intermediate_representation import *
from framework.staggered_conflict_resolution import resolve_conflict
from iterative_local_search.conflict import is_conflict_free
from framework.preprocessing import *

time_step = 5
max_run_time = 5
c = Charger(ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
            segment_construction_cost=0)
dc = Charger(ChargerID("D1"), charging_rate=30, transformer_construction_cost=30, segment_construction_cost=1)


@pytest.fixture
def test_routes() -> Tuple[List[Route]]:
    two_vehicles = [
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
                    vertex_id=VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=765,
                    latest_time_of_service=775,
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
            vehicle_id=VehicleID("1"),
        ),
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
                    vertex_id=VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=770,
                    latest_time_of_service=785,
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
            vehicle_id=VehicleID("2"),
        )
    ]
    third_route = Route(
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
                    vertex_id=VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=1001,
                    latest_time_of_service=1201,
                ),
                Stop(
                    vertex_id=VertexID("S3"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=2270,
                ),
                Stop(
                    vertex_id=VertexID("S1"),
                    stopover_time=1.0,
                    earliest_time_of_service=2500,
                    latest_time_of_service=3000,
                ),
                Stop(
                    vertex_id=VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=2500,
                    latest_time_of_service=3000,
                ),
                Stop(
                    vertex_id=VertexID("S3"),
                    stopover_time=1.0,
                    earliest_time_of_service=2800,
                    latest_time_of_service=3200,
                ),
                Stop(
                    vertex_id=VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=3100,
                    latest_time_of_service=3400,
                ),
            ],
            vehicle_id=VehicleID("3"),
        )
    three_vehicles = two_vehicles.copy()
    three_vehicles.append(third_route)
    return two_vehicles, three_vehicles


@pytest.fixture
def test_intermediate_representation(test_routes) -> Tuple[
    IntermediateRepresentation, IntermediateRepresentation, IntermediateRepresentation
]:
    coordinate = util.Coordinate(0.0, 0.0)
    two_vehicles = test_routes[0]
    three_vehicles = test_routes[1]
    vertices_one = [
        ir.Vertex(id=VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("D1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("D2"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("C1"), is_stop=False, is_depot=False, coordinate=coordinate)
    ]

    vertices_two = [
        ir.Vertex(id=VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("D1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("D2"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("C1"), is_stop=False, is_depot=False, coordinate=coordinate)
    ]

    vertices_three = [
        ir.Vertex(id=VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("D1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("D2"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("C1"), is_stop=False, is_depot=False, coordinate=coordinate)
    ]

    arcs = {
        (VertexID("Depot"), VertexID("S1")): Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("S1"), VertexID("S2")): Arc(
            distance=7.0, speed_limit=30, constructible_chargers=set()
        ),
        (VertexID("S1"), VertexID("S2")): Arc(
            distance=2.0, speed_limit=15, constructible_chargers=set()
        ),
        (VertexID("S1"), VertexID("C1")): Arc(
            distance=1.0, speed_limit=15, constructible_chargers=set()
        ),
        (VertexID("C1"), VertexID("S2")): Arc(
            distance=1.0, speed_limit=15, constructible_chargers=set()
        ),
        (VertexID("S2"), VertexID("S3")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("S3"), VertexID("D1")): ir.Arc(
            distance=0.1, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("D1"), VertexID("D2")): ir.Arc(
            distance=0.8, speed_limit=10, constructible_chargers={dc}
        ),
        (VertexID("D2"), VertexID("S1")): ir.Arc(
            distance=0.1, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("S3"), VertexID("Depot")): Arc(
            distance=1.0, speed_limit=60, constructible_chargers=set()
        )
    }
    # same arcs but copies of vertices
    return IntermediateRepresentation(vertices=vertices_one, arcs=arcs, routes=two_vehicles), \
           IntermediateRepresentation(vertices=vertices_two, arcs=arcs, routes=three_vehicles), \
           IntermediateRepresentation(vertices=vertices_three, arcs=arcs, routes=three_vehicles)


@pytest.fixture
def build_decision_variables(test_intermediate_representation) -> Tuple[DecisionVariables, DecisionVariables, DecisionVariables]:

    ir_rep_two = test_intermediate_representation[0]
    ir_rep_three = test_intermediate_representation[1]
    ir_rep_four = test_intermediate_representation[2]

    ir_rep_two.get_vertex("S2").constructible_charger = {c}
    ir_rep_three.get_vertex("S2").constructible_charger = {c}
    ir_rep_four.get_vertex("C1").constructible_charger = {c}

    intermediate_rep_two_simplified = simplify_intermediate_repr(ir_rep_two, False)
    intermediate_rep_three_simplified = simplify_intermediate_repr(ir_rep_three, False)
    intermediate_rep_four_simplified = simplify_intermediate_repr(ir_rep_four, True)

    max_speed = 100
    max_consumption = 7

    soc_init = 100
    soc_max = 100
    soc_min = 30

    energy_prices = 1.0
    consumption_cost = 0.15

    decision_variables_two = DecisionVariables(intermediate_rep_two_simplified, {VertexID("S2"): c}, {}, 71, max_consumption,
                                           soc_init, soc_max,
                                           soc_min,
                                           energy_prices, consumption_cost)
    decision_variables_three = DecisionVariables(intermediate_rep_three_simplified, {VertexID("S2"): c}, {(VertexID("D1"), VertexID("D2")): dc}, 70, max_consumption,
                                           soc_init, soc_max,
                                           soc_min,
                                           energy_prices, consumption_cost)
    decision_variables_three_standalone = DecisionVariables(intermediate_rep_four_simplified, {VertexID("C1"): c}, {(VertexID("D1"), VertexID("D2")): dc}, 69, max_consumption,
                                           soc_init, soc_max,
                                           soc_min,
                                           energy_prices, consumption_cost)

    return decision_variables_two, decision_variables_three, decision_variables_three_standalone


def test_pair_conflict(build_decision_variables):
    decision_variables = build_decision_variables[0]
    decision_variables.soc_init = 90
    solution = SolutionRepresentation(
        [], decision_variables.construct_dynamic_invest(), decision_variables.construct_static_invest(), 0.0, 0.0, 0.0
    )
    for i in range(2):
        a = decision_spprc_network(
            decision_variables, decision_variables.intermediate_rep.routes[i], time_step
        )
        solution += A_star(a, decision_variables, i, False, "python")

    # should not be conflict free
    assert not is_conflict_free(decision_variables, time_step, solution)[0]

    # and with a solution passed
    sol = resolve_conflict(solution,decision_variables.intermediate_rep, decision_variables.vehicle_max_speed)
    assert isinstance(sol, SolutionRepresentation)


def test_multi_conflict(build_decision_variables):
    decision_variables = build_decision_variables[1]
    decision_variables.soc_init = 90
    solution = SolutionRepresentation(
        [], decision_variables.construct_dynamic_invest(), decision_variables.construct_static_invest(), 0.0, 0.0, 0.0
    )
    for i in range(3):
        a = decision_spprc_network(
            decision_variables, decision_variables.intermediate_rep.routes[i], time_step
        )
        solution += A_star(a, decision_variables, i, False, "python")

    # should not be conflict free
    assert not is_conflict_free(decision_variables, time_step, solution)[0]

    # and with a solution passed
    sol = resolve_conflict(solution, decision_variables.intermediate_rep, decision_variables.vehicle_max_speed)
    assert isinstance(sol, SolutionRepresentation)


def test_multi_conflict_with_standalone_charger(build_decision_variables):
    decision_variables = build_decision_variables[2]
    decision_variables.soc_init = 90
    solution = SolutionRepresentation(
        [], decision_variables.construct_dynamic_invest(), decision_variables.construct_static_invest(), 0.0, 0.0, 0.0
    )
    for i in range(3):
        a = decision_spprc_network(
            decision_variables, decision_variables.intermediate_rep.routes[i], time_step
        )
        solution += A_star(a, decision_variables, i, False, "python")

    # should not be conflict free
    assert not is_conflict_free(decision_variables, time_step, solution)[0]

    # and with a solution passed
    sol = resolve_conflict(solution, decision_variables.intermediate_rep, decision_variables.vehicle_max_speed)
    assert isinstance(sol, SolutionRepresentation)

