import pytest

import os
import sys

from typing import List

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)

import framework.intermediate_representation as ir
import framework.utils as util
from framework.preprocessing import simplify_intermediate_repr
from framework.intermediate_representation import VertexID
from iterative_local_search.decision_variables import DecisionVariables
from iterative_local_search.iterative_local_search import BinaryTightener


time_step = 5
max_run_time = 5
c = ir.Charger(ir.ChargerID("Charger1"), charging_rate=30000, transformer_construction_cost=10,
            segment_construction_cost=1)


@pytest.fixture
def test_routes() -> List[ir.Route]:
    return [
        ir.Route(
            stop_sequence=[
                ir.Stop(
                    vertex_id=VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=0.0,
                    latest_time_of_service=1.0,
                ),
                ir.Stop(
                    vertex_id=VertexID("S1"),
                    stopover_time=1.0,
                    earliest_time_of_service=250,
                    latest_time_of_service=300,
                ),
                ir.Stop(
                    vertex_id=VertexID("C1"),
                    stopover_time=1.0,
                    earliest_time_of_service=250,
                    latest_time_of_service=1200,
                ),
                ir.Stop(
                    vertex_id=VertexID("S2"),
                    stopover_time=1.0,
                    earliest_time_of_service=1000,
                    latest_time_of_service=1200,
                ),
                ir.Stop(
                    vertex_id=VertexID("S3"),
                    stopover_time=1.0,
                    earliest_time_of_service=2000,
                    latest_time_of_service=2270,
                ),
                ir.Stop(
                    vertex_id=VertexID("Depot"),
                    stopover_time=1.0,
                    earliest_time_of_service=2200,
                    latest_time_of_service=2700,
                ),
            ],
            vehicle_id=ir.VehicleID("0"),
        )
    ]

@pytest.fixture
def test_intermediate_representation(test_routes) -> ir.IntermediateRepresentation:
    coordinate = util.Coordinate(0.0, 0.0)

    vertices = [
        ir.Vertex(id=VertexID("Depot"), is_stop=False, is_depot=True, coordinate=coordinate),
        ir.Vertex(id=VertexID("S1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S2"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("S3"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("C1"), is_stop=True, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment1"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment2"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment3"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment4"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment5"), is_stop=False, is_depot=False, coordinate=coordinate),
        ir.Vertex(id=VertexID("Segment6"), is_stop=False, is_depot=False, coordinate=coordinate),
    ]

    arcs = {
        (VertexID("Depot"), VertexID("S1")): ir.Arc(
            distance=4.0, speed_limit=50, constructible_chargers=set()
        ),
        (VertexID("S1"), VertexID("Segment1")): ir.Arc(
            distance=0.000001, speed_limit=70, constructible_chargers=set()
        ),
        (VertexID("Segment1"), VertexID("Segment2")): ir.Arc(
            distance=0.3, speed_limit=70, constructible_chargers={c}
        ),
        (VertexID("Segment2"), VertexID("Segment3")): ir.Arc(
            distance=0.3, speed_limit=70, constructible_chargers={c}
        ),
        (VertexID("Segment3"), VertexID("Segment4")): ir.Arc(
            distance=0.3, speed_limit=70, constructible_chargers={c}
        ),
        (VertexID("Segment4"), VertexID("Segment5")): ir.Arc(
            distance=0.3, speed_limit=70, constructible_chargers={c}
        ),
        (VertexID("Segment5"), VertexID("Segment6")): ir.Arc(
            distance=0.8, speed_limit=70, constructible_chargers={c}
        ),
        (VertexID("Segment6"), VertexID("C1")): ir.Arc(
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
    return ir.IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=test_routes)



def build_decision_variables_vertex_charger(intermediate_rep: ir.IntermediateRepresentation) -> DecisionVariables:
    max_speed = 100
    max_consumption = 7

    soc_init = 100
    soc_max = 100
    soc_min = 30

    energy_prices = 1.0
    consumption_cost = 0.0

    return DecisionVariables(
        intermediate_rep, {}, {}, max_speed, max_consumption, soc_init, soc_max, soc_min, energy_prices, consumption_cost
    )


@pytest.mark.parametrize(
    "feasible1, feasible2, feasible3, num_segments, upper_index_0, upper_index_1, upper_index_2, upper_index_3, final_upper_index",
    [
        ("True", "True", "False", "5",  "5", "2", "1", "0", "1"),
        ("True", "False", "None", "5",  "5", "2", "1", "None", "2"),
        ("False", "True", "False", "5", "5", "2", "3", "None", "3"),
        ("False", "False", "True", "5", "5", "2", "3", "4", "4"),
        ("True", "True", "True", "4", "4", "2", "1", "0", "0"),
        ("True", "False", "None", "4", "4", "2", "1", "None", "2"),
        ("False", "True", "None", "4", "4", "2", "3", "None", "3"),
        ("False", "False", "True", "4", "4", "2", "3", "4", "4"),
    ]
)
def test_binary_tightener(
        test_intermediate_representation, feasible1, feasible2, feasible3, num_segments, upper_index_0, upper_index_1,
        upper_index_2, upper_index_3, final_upper_index):

    segment_list = [
        (VertexID("Segment1"), VertexID("Segment2")),
        (VertexID("Segment2"), VertexID("Segment3")),
        (VertexID("Segment3"), VertexID("Segment4")),
        (VertexID("Segment4"), VertexID("Segment5")),
    ]

    if num_segments=="5":
        segment_list.append((VertexID("Segment5"), VertexID("Segment6")))

    for i in range(len(segment_list)):
        test_intermediate_representation.get_arc(*segment_list[i]).constructible_charger = {c}

    intermediate_rep_simplified = simplify_intermediate_repr(
        inter_rep=test_intermediate_representation, allow_path_deviation=True
    )

    # this adds the full charger (i.e., all segments)
    decision_variables = build_decision_variables_vertex_charger(intermediate_rep_simplified)
    decision_variables.add_charging_station((VertexID("Segment1"), VertexID("Segment2"), c))

    # initialise the deconstruction class
    test_class = BinaryTightener(decision_variables, segments=segment_list, current_solution=0.0, charger=c)
    assert test_class.current_index == int(upper_index_0)

    # conduct first cut (initial configuration has to be feasible)
    test_class.cut_segments(True)

    def str_to_bool(s):
        if s == "True":
            return True
        elif s == "False":
            return False
        else:
            raise ValueError(f"Cannot parse '{s}' as a boolean")

    for iter in range(1,4):
        if test_class.has_converged():
            break
        assert test_class.current_index == int(eval(f"upper_index_{iter}"))
        test_class.cut_segments(str_to_bool(eval(f"feasible{iter}")))
    assert int(eval(final_upper_index)) == test_class.upper_bound_index