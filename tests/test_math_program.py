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
from math_programming_model.math_programming_solver import MathProgrammingParameters, MathProgrammingSolver
from framework.instance_parameter import InstanceParameters

import framework.intermediate_representation as ir

time_step = 1
max_run_time = 5
stat_c = Charger(ChargerID("Charger1"), charging_rate=30000, transformer_construction_cost=10,
            segment_construction_cost=0)
dyn_c = Charger(ChargerID("Charger2"), charging_rate=30000, transformer_construction_cost=9,
            segment_construction_cost=1e-3)


@pytest.fixture
def test_routes() -> List[Route]:
    circles = [
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
                    vertex_id=VertexID("C1"),
                    stopover_time=1.0,
                    earliest_time_of_service=250,
                    latest_time_of_service=1200,
                ),
                Stop(
                    vertex_id=VertexID("S20"),
                    stopover_time=1.0,
                    earliest_time_of_service=1000,
                    latest_time_of_service=1200,
                ),
                Stop(
                    vertex_id=VertexID("S30"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=2270,
                ),
                Stop(
                    vertex_id=VertexID("S1"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=2300,
                ),
                Stop(
                    vertex_id=VertexID("C1"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=3200,
                ),
                Stop(
                    vertex_id=VertexID("S20"),
                    stopover_time=1.0,
                    earliest_time_of_service=2750,
                    latest_time_of_service=3200,
                ),
                Stop(
                    vertex_id=VertexID("S30"),
                    stopover_time=1.0,
                    earliest_time_of_service=3750,
                    latest_time_of_service=4270,
                ),
                Stop(
                    vertex_id=VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=4200,
                    latest_time_of_service=4700,
                ),
            ],
            vehicle_id=VehicleID("1"),
        )]
    return circles


@pytest.fixture
def test_intermediate_representation(test_routes) -> Tuple[IntermediateRepresentation, IntermediateRepresentation]:
    coordinate = util.Coordinate(0.0, 0.0)
    circles = test_routes
    vertices = [
        ir.Vertex(id=VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S20"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S30"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("C1"), is_stop=True, is_depot=False, coordinate=coordinate, constructible_charger={stat_c}),
        ir.Vertex(id=VertexID("Segment1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment2"), is_stop=False, is_depot=False, coordinate=coordinate),
    ]

    arcs = {
        (VertexID("Depot"), VertexID("S1")): ir.Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("S1"), VertexID("Segment1")): ir.Arc(
            distance=0.0001, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("Segment1"), VertexID("Segment2")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers={dyn_c}
        ),
        (VertexID("Segment2"), VertexID("C1")): ir.Arc(
            distance=0.0001, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("C1"), VertexID("S2")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("C1"), VertexID("S20")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("S2"), VertexID("S3")): ir.Arc(
            distance=6.0, speed_limit=20, constructible_chargers=set()
        ),
        (VertexID("S20"), VertexID("S30")): ir.Arc(
            distance=6.0, speed_limit=20, constructible_chargers=set()
        ),
        (VertexID("S30"), VertexID("S1")): ir.Arc(
            distance=0.000001, speed_limit=20, constructible_chargers=set()
        ),
        (VertexID("S3"), VertexID("Depot")): ir.Arc(
            distance=1.0, speed_limit=60, constructible_chargers=set()
        ),
        (VertexID("S30"), VertexID("Depot")): ir.Arc(
            distance=1.0, speed_limit=60, constructible_chargers=set()
        )
    }

    vs_with_second_segment = [
        ir.Vertex(id=VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S20"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S30"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("C1"), is_stop=True, is_depot=False, coordinate=coordinate,
                  constructible_charger={stat_c}),
        ir.Vertex(id=VertexID("Segment1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("SegmentMiddle"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment2"), is_stop=False, is_depot=False, coordinate=coordinate),
    ]

    as_with_second_segment = {
        (VertexID("Depot"), VertexID("S1")): ir.Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("S1"), VertexID("Segment1")): ir.Arc(
            distance=0.0001, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("Segment1"), VertexID("SegmentMiddle")): ir.Arc(
            distance=1.0, speed_limit=70, constructible_chargers={dyn_c}
        ),
        (VertexID("SegmentMiddle"), VertexID("Segment2")): ir.Arc(
            distance=1.0, speed_limit=70, constructible_chargers={dyn_c}
        ),
        (VertexID("Segment2"), VertexID("C1")): ir.Arc(
            distance=0.0001, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("C1"), VertexID("S2")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("C1"), VertexID("S20")): ir.Arc(
            distance=2.0, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("S2"), VertexID("S3")): ir.Arc(
            distance=6.0, speed_limit=20, constructible_chargers=set()
        ),
        (VertexID("S20"), VertexID("S30")): ir.Arc(
            distance=6.0, speed_limit=20, constructible_chargers=set()
        ),
        (VertexID("S30"), VertexID("S1")): ir.Arc(
            distance=0.000001, speed_limit=20, constructible_chargers=set()
        ),
        (VertexID("S3"), VertexID("Depot")): ir.Arc(
            distance=1.0, speed_limit=60, constructible_chargers=set()
        ),
        (VertexID("S30"), VertexID("Depot")): ir.Arc(
            distance=1.0, speed_limit=60, constructible_chargers=set()
        )
    }

    return IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=circles),\
           IntermediateRepresentation(vertices=vs_with_second_segment, arcs=as_with_second_segment, routes=circles)

@pytest.mark.parametrize(
    "battery_capacity, stat_charging_rate, dyn_charging_rate, stat_cost, dyn_fix_cost, dyn_var_cost, expected_value",
    [
        ("120", "1000", "30", "10", "2", "5", "122.22222"),
        ("100", "1000", "30", "10", "2", "5", "5152.4285714"),
        ("150", "10", "1235", "10", "2", "0.0001", "72.20419999999"),
        ("170", "10", "1235", "10", "2", "0.0001", " 58.202799999999"),
    ]
)
def test_math_program_with_circles(test_intermediate_representation, battery_capacity, stat_charging_rate, dyn_charging_rate,
                                 stat_cost, dyn_fix_cost, dyn_var_cost, expected_value):
    # get intermediate representation and update with test parameters
    inter_rep = test_intermediate_representation[1]
    for k1, k2, arc in inter_rep.charger_edges:
        for c in arc.constructible_chargers:
            c.charging_rate=float(dyn_charging_rate)
            c.transformer_construction_cost = float(dyn_fix_cost)
            c.segment_construction_cost = float(dyn_var_cost)
    for v in inter_rep.charger_nodes:
        for c in v.constructible_charger:
            c.charging_rate=float(stat_charging_rate)
            c.transformer_construction_cost = float(stat_cost)

    instance_parameters = InstanceParameters(velocity=100,consumption=7,soc_init=float(battery_capacity),
                                             soc_max=float(battery_capacity),soc_min=0.3*float(battery_capacity),
                                             energy_prices=1.0,consumption_cost=0.0, allow_path_deviations=True)
    solver_parameters = MathProgrammingParameters(None, time_step, max_run_time, num_replicas=0)
    solver = MathProgrammingSolver(instance_parameters, solver_parameters)
    sol_rep = solver.solve(inter_rep)

    assert sol_rep.recharged_energy == pytest.approx(sum(itinerary.route[-1].accumulated_charged_energy for itinerary in sol_rep.itineraries))
    assert sol_rep.consumed_energy == pytest.approx(sum(itinerary.route[-1].accumulated_consumed_energy for itinerary in sol_rep.itineraries))

    # if infeasible instance
    if sol_rep is None:
        assert str(expected_value)=="None"
    else:
        assert sol_rep.global_cost == pytest.approx(float(expected_value), 1e-3)


@pytest.mark.parametrize(
    "battery_capacity, stat_charging_rate, dyn_charging_rate, stat_cost, dyn_fix_cost, dyn_var_cost, expected_value",
    [
        ("120", "1000", "30", "10", "2", "5", "122.2222222"),
        ("100", "1000", "30", "10", "2", "5", "5152.428571"),
        ("150", "10", "1235", "10", "2", "0.0001", "72.20419999999"),
        ("170", "10", "1235", "10", "2", "0.0001", "58.20279999999"),
        ("170", "10", "1500", "10", "2", "0.0001", "58.202799999999"),
        ("150", "100", "1235", "0.001", "2", "1", "1072.0052"),
        ("150", "1000", "1235", "0.001", "2", "1", "70.5565"),
    ]
)
def test_math_program_with_transformers(test_intermediate_representation, battery_capacity, stat_charging_rate, dyn_charging_rate,
                                 stat_cost, dyn_fix_cost, dyn_var_cost, expected_value):
    # get intermediate representation and update with test parameters
    inter_rep = test_intermediate_representation[1]
    for k1, k2, arc in inter_rep.charger_edges:
        for c in arc.constructible_chargers:
            c.charging_rate=float(dyn_charging_rate)
            c.transformer_construction_cost = float(dyn_fix_cost)
            c.segment_construction_cost = float(dyn_var_cost)
    for v in inter_rep.charger_nodes:
        for c in v.constructible_charger:
            c.charging_rate=float(stat_charging_rate)
            c.transformer_construction_cost = float(stat_cost)

    instance_parameters = InstanceParameters(velocity=100,consumption=7,soc_init=float(battery_capacity),
                                             soc_max=float(battery_capacity),soc_min=0.3*float(battery_capacity),
                                             energy_prices=1.0,consumption_cost=0.0, allow_path_deviations=False)
    solver_parameters = MathProgrammingParameters(None, time_step, max_run_time, num_replicas=1)
    solver = MathProgrammingSolver(instance_parameters, solver_parameters)
    sol_rep = solver.solve(inter_rep)

    # if infeasible instance
    if sol_rep is None:
        assert str(expected_value)=="None"
    else:
        # else convert to solution representation and check
        assert sol_rep.global_cost == pytest.approx(float(expected_value), 1e-3)

