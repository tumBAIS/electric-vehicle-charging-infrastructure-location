import pytest

import framework.preprocessing
from framework.utils import *
import framework.intermediate_representation as ir
from framework.preprocessing import simplify_intermediate_repr
from iterative_local_search.decision_variables import DecisionVariables
from iterative_local_search.spprc_network import decision_spprc_network
from iterative_local_search.subproblem import lower_energy_bounds

star_c = ir.Charger(ir.ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
            segment_construction_cost=0)
dyn_c = ir.Charger(ir.ChargerID("C1"), charging_rate=30000, transformer_construction_cost=10,
            segment_construction_cost=1)

@pytest.fixture
def test_routes() -> Tuple[List[ir.Route], List[ir.Route]]:
    return ([
        ir.Route(
            stop_sequence=[
                ir.Stop(
                    vertex_id=ir.VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=1.0,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S1"),
                    stopover_time=1.0,
                    earliest_time_of_service=250,
                    latest_time_of_service=300,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=1000,
                    latest_time_of_service=1200,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S3"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=2270,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=2200,
                    latest_time_of_service=2700,
                ),
            ],
            vehicle_id=ir.VehicleID("1"),
        )
    ],
    [
        ir.Route(
            stop_sequence=[
                ir.Stop(
                    vertex_id=ir.VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=1.0,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S1"),
                    stopover_time=1.0,
                    earliest_time_of_service=250,
                    latest_time_of_service=300,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=1000,
                    latest_time_of_service=1200,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S3"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=2270,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S1"),
                    stopover_time=1.0,
                    earliest_time_of_service=2200,
                    latest_time_of_service=2300,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=2950,
                    latest_time_of_service=3150,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("S3"),
                    stopover_time=1.0,
                    earliest_time_of_service=3950,
                    latest_time_of_service=4220,
                ),
                ir.Stop(
                    vertex_id=ir.VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=4150,
                    latest_time_of_service=4650,
                ),
            ],
            vehicle_id=ir.VehicleID("1"),
            )
    ],
)


@pytest.fixture
def test_intermediate_representation(test_routes) -> Tuple[ir.IntermediateRepresentation, ir.IntermediateRepresentation]:
    coordinate = Coordinate(0.0, 0.0)

    vertices = [
        ir.Vertex(id=ir.VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=ir.VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=ir.VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=ir.VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=ir.VertexID("Segment1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=ir.VertexID("Segment2"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=ir.VertexID("Segment3"), is_stop=False, is_depot=False, coordinate=coordinate),
    ]

    arcs = {
        (ir.VertexID("Depot"), ir.VertexID("S1")): ir.Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (ir.VertexID("S1"), ir.VertexID("S2")): ir.Arc(
            distance=2.0, speed_limit=30, constructible_chargers=set()
        ),
        (ir.VertexID("S2"), ir.VertexID("Segment1")): ir.Arc(
            distance=0.7, speed_limit=70, constructible_chargers=set()
        ),
        (ir.VertexID("Segment1"), ir.VertexID("Segment2")): ir.Arc(
            distance=0.1, speed_limit=70, constructible_chargers=set()
        ),
        (ir.VertexID("Segment2"), ir.VertexID("Segment3")): ir.Arc(
            distance=1.0, speed_limit=70, constructible_chargers=set()
        ),
        (ir.VertexID("Segment3"), ir.VertexID("S3")): ir.Arc(
            distance=0.2, speed_limit=70, constructible_chargers=set()
        ),
        (ir.VertexID("S3"), ir.VertexID("Depot")): ir.Arc(
            distance=1.0, speed_limit=60, constructible_chargers=set()
        ),
    }

    arcs_circular = arcs
    arcs_circular[(ir.VertexID("S3"), ir.VertexID("S1"))] = ir.Arc(
            distance=0.01, speed_limit=60, constructible_chargers=set()
        )
    return ir.IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=test_routes[0]), \
           ir.IntermediateRepresentation(vertices=vertices, arcs=arcs_circular, routes=test_routes[1]),


@pytest.mark.parametrize(
    "consumption,factor",
    [
        ("7", "1"),
        ("14", "2"),
    ]
)
def test_lower_bounds(test_intermediate_representation, consumption, factor):
    inter_rep = test_intermediate_representation[0]
    inter_rep.get_vertex(ir.VertexID("S2")).constructible_charger = {star_c}
    inter_rep.get_arc("Segment1", "Segment2").constructible_chargers = {dyn_c}
    inter_rep = simplify_intermediate_repr(inter_rep, False)
    decision_variables = DecisionVariables(
        inter_rep, vertex_chargers={ir.VertexID("S3"): star_c}, arc_chargers={
            (ir.VertexID("Segment1"), ir.VertexID("Segment2")): dyn_c,
            (ir.VertexID("Segment2"), ir.VertexID("Segment3")): dyn_c,
        },
        vehicle_max_speed=100, vehicle_consumption=float(consumption), soc_init=10*float(factor), max_soc=100, min_soc=30,
        energy_prices=1.0, consumption_cost=0.15)
    network = decision_spprc_network(decision_variables,inter_rep.routes[0],5)
    baseline = {
        'S3': 7.0, 'S2': 21.0, 'S1': 35.0, 'Depot': 63.0, 'Depot-end': 0, 'S3-0': 7.0, 'S3-1': 7.0,
        'S2-Segment1-S3': 16.1, 'S2-Segment2-S3': 15.4, 'S2-Segment3-S3': 8.4
    }
    for b in lower_energy_bounds(network, 0, decision_variables).values():
        assert (b / int(factor)) in baseline.values()



# @pytest.mark.parametrize(
#     "consumption,soc_init,dyn_charging_rate,stat_charging_rate,expected_value",
#     [
#         ("7", "46.8", "30000", "30000", "42.8571428"),
#         ("7", "47.0", "485", "1000", "None"), # dyn charging rate is lower than consumption --> inf
#         ("7", "60", "600", "100", "5.55555555"), # dyn charging rate equates consumption --> you make it to static charger
#         ("7", "47.0", "600", "1e-9", "None"),
#     ]
# )
# def test_upper_bound(test_intermediate_representation, consumption, soc_init, dyn_charging_rate, stat_charging_rate,
#                      expected_value):
#     inter_rep = test_intermediate_representation[0]
#     star_c.charging_rate = float(stat_charging_rate)
#     dyn_c.charging_rate = float(dyn_charging_rate)
#     inter_rep.get_vertex(ir.VertexID("S2")).constructible_charger = {star_c}
#     inter_rep.get_arc("Segment1", "Segment2").constructible_chargers = {dyn_c}
#     inter_rep = simplify_intermediate_repr(inter_rep, False)
#     decision_variables = DecisionVariables(
#         inter_rep, vertex_chargers={ir.VertexID("S3"): star_c}, arc_chargers={
#             (ir.VertexID("Segment1"), ir.VertexID("Segment2")): dyn_c,
#             (ir.VertexID("Segment2"), ir.VertexID("Segment3")): dyn_c,
#         },
#         vehicle_max_speed=100, vehicle_consumption=float(consumption), soc_init=float(soc_init), max_soc=1e6,
#         min_soc=0, energy_prices=1.0, consumption_cost=0)
#     network = decision_spprc_network(decision_variables, inter_rep.routes[0], 100)
#     ub = get_upper_bound(network, soc_init=float(soc_init), min_soc=0, max_soc=1e6, lower_bounds=lower_energy_bounds(network, 0, decision_variables),
#                          restrict_arrival_time=False)
#     if expected_value=="None":
#         assert ub == math.inf
#     else:
#         assert ub == pytest.approx(float(expected_value), 1e-3)
