import pytest

import framework.preprocessing
from framework.utils import *
import framework.intermediate_representation as ir
from iterative_local_search.decision_variables import DecisionVariables
from iterative_local_search.spprc_network import decision_spprc_network
from framework.preprocessing import simplify_intermediate_repr

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
    "soc_init,charging_rate,expected_num_vertices,expected_num_arcs",
    [
        ("100", "30000", "5", "4"),  # no chargers
        ("90", "30000", "7", "8"),
        ("90", "15000", "7", "9") # half the charging rate
    ]
)
def test_network(test_intermediate_representation, charging_rate, soc_init, expected_num_vertices, expected_num_arcs):
    inter_rep = test_intermediate_representation[0]
    inter_rep.get_vertex(ir.VertexID("S2")).constructible_charger = {star_c}
    inter_rep.get_arc("Segment1", "Segment2").constructible_chargers = {dyn_c}
    star_c.charging_rate = float(charging_rate)
    inter_rep = simplify_intermediate_repr(inter_rep, False)
    decision_variables = DecisionVariables(
        inter_rep, {ir.VertexID("S2"): star_c}, {(ir.VertexID("Segment1"), ir.VertexID("Segment2")): dyn_c},
        100, 7,float(soc_init), 100, 30,1.0, 0.15
    )
    network = decision_spprc_network(decision_variables,inter_rep.routes[0],5)
    assert len(network.vertices.keys())==int(expected_num_vertices)
    assert len(list(itertools.chain(*network.arcs.values()))) == int(expected_num_arcs)


@pytest.mark.parametrize(
    "soc_init,charging_rate,expected_num_vertices,expected_num_arcs",
    [
        ("100", "30000", "5", "4"),  # This should still work because dyn segments are only added if needed
        ("90", "30000", "7", "8"), # This should work
    ]
)
def test_network_with_dyn_segment_to_stationary_charger(test_intermediate_representation, charging_rate, soc_init, expected_num_vertices, expected_num_arcs):
    inter_rep = test_intermediate_representation[0]
    star_c.charging_rate = float(charging_rate)
    inter_rep.get_vertex(ir.VertexID("S2")).constructible_charger = {star_c}
    inter_rep.get_arc("Segment1", "Segment2").constructible_chargers = {dyn_c}
    inter_rep = simplify_intermediate_repr(inter_rep, False)
    decision_variables = DecisionVariables(
        inter_rep, {ir.VertexID("S3"): star_c},
        {(ir.VertexID("Segment1"), ir.VertexID("Segment2")): dyn_c, (ir.VertexID("Segment2"), ir.VertexID("Segment3")): dyn_c},
        100, 7,float(soc_init), 100, 30,1.0, 0.15
    )
    network = decision_spprc_network(decision_variables,inter_rep.routes[0],5)
    assert len(network.vertices.keys())==int(expected_num_vertices)
    assert len(list(itertools.chain(*network.arcs.values()))) == int(expected_num_arcs)


@pytest.mark.parametrize(
    "soc_init,charging_rate,expected_num_vertices,expected_num_arcs,allow_deviations",
    [
        ("200", "30000", "8", "7", "False"),  # Circular route (x2) but also double soc init --> no chargers needed
        ("90", "30000", "12", "15", "False"),  # Manually checked --> makes sense
        ("90", "30000", "12", "20", "True"),  # Manually checked --> makes sense
    ]
)
def test_circular_network_with_dyn_segment_to_stationary_charger(test_intermediate_representation, charging_rate, soc_init, expected_num_vertices, expected_num_arcs, allow_deviations):
    if allow_deviations == "False":
        allow_deviations=False
    else:
        allow_deviations=True
    inter_rep = test_intermediate_representation[1]
    inter_rep.get_arc(origin=ir.VertexID("Segment1"), target=ir.VertexID("Segment2")).constructible_chargers = {
        dyn_c
    }
    inter_rep.get_arc(origin=ir.VertexID("Segment2"), target=ir.VertexID("Segment3")).constructible_chargers = {
        dyn_c
    }
    inter_rep.get_vertex(ir.VertexID("S3")).constructible_chargers = {
        star_c
    }
    star_c.charging_rate = float(charging_rate)
    inter_rep = simplify_intermediate_repr(inter_rep, allow_path_deviation=allow_deviations)
    decision_variables = DecisionVariables(
        inter_rep, {ir.VertexID("S3"): star_c},
        {(ir.VertexID("Segment1"), ir.VertexID("Segment2")): dyn_c, (ir.VertexID("Segment2"), ir.VertexID("Segment3")): dyn_c},
        100, 7,float(soc_init), 200, 30,1.0, 0.15
    )
    network = decision_spprc_network(decision_variables,inter_rep.routes[0], 5)
    assert len(network.vertices.keys())==int(expected_num_vertices)
    assert len(list(itertools.chain(*network.arcs.values()))) == int(expected_num_arcs)

