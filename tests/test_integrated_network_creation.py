import math
import pytest
import itertools
from framework.utils import Coordinate, distance_euclidean
from framework.preprocessing import simplify_intermediate_repr
import framework.intermediate_representation as ir
from math_programming_model.math_programming_solver import MathProgrammingParameters, MathProgrammingSolver
from framework.instance_parameter import InstanceParameters
from iterative_local_search.decision_variables import DecisionVariables
from iterative_local_search.subproblem import A_star
from iterative_local_search.spprc_network import decision_spprc_network



def distance(f, t):
    return math.sqrt((f.lat - t.lat) ** 2 + (f.lon - t.lon) ** 2)

# Coordinates
coords = dict({
    "Depot": (0,0),
    "s1": (0, 100),
    "s2": (100, 100),
    "c1": (100, 0),
    "c2": (50, 50),
    "d1": (45, 200),
    "d2": (55, 200),
    "d3": (0, 45),
    "d4": (0, 55),
})

# Chargers
stat_chargers = {
    ir.VertexID("c1"): {
        ir.Charger(ir.ChargerID("c1"), charging_rate=1,
    transformer_construction_cost=1,
                   segment_construction_cost=0)}, # not at spp
    ir.VertexID("c2"): {
        ir.Charger(ir.ChargerID("c2"), charging_rate=1, transformer_construction_cost=100,
                   segment_construction_cost=0)}, # not at stop but on spp
    ir.VertexID("s2"): {
        ir.Charger(ir.ChargerID("s2"), charging_rate=1, transformer_construction_cost=100,
                   segment_construction_cost=0)} # at stop
}

dyn_chargers = {
    (ir.VertexID("d1"), ir.VertexID("d2")): {
        ir.Charger(ir.ChargerID("d12"), charging_rate=1, transformer_construction_cost=1,
                   segment_construction_cost=1e-3)}, # not on spp
    (ir.VertexID("d3"), ir.VertexID("d4")): {
        ir.Charger(ir.ChargerID("d34"), charging_rate=1, transformer_construction_cost=1,
                   segment_construction_cost=1e-3)}, # on spp
}

route1 = ir.Route(
    stop_sequence=[
        ir.Stop(ir.VertexID("Depot"), 0.0, 0.0, 1.0),
        ir.Stop(ir.VertexID("s1"), 0.0, 0.0, 400),
        ir.Stop(ir.VertexID("s2"), 0.0, 300, 1200),
        ir.Stop(ir.VertexID("Depot"), 0.0, 600, 1600),
    ],
    vehicle_id=ir.VehicleID("Vehicle1")
)

route2 = ir.Route(
    stop_sequence=[
        ir.Stop(ir.VertexID("Depot"), 0.0, 0.0, 1.0),
        ir.Stop(ir.VertexID("s1"), 0.0, 0.0, 400),
        ir.Stop(ir.VertexID("s2"), 0.0, 300, 1400),
        ir.Stop(ir.VertexID("s1"), 0.0, 800, 1600),
        ir.Stop(ir.VertexID("Depot"), 0.0, 1200, 1900),
    ],
    vehicle_id=ir.VehicleID("Vehicle2")
)

vertices = []
for l,coord in coords.items():
    vertices.append(
        ir.Vertex(ir.VertexID(l), "Depot" in l, "s" in l, Coordinate(*coord), constructible_charger=stat_chargers[ir.VertexID(l)] if ir.VertexID(l) in stat_chargers else set())
    )

arcs = dict()
for (v1, v2) in itertools.permutations(vertices, 2):
    arcs[(v1.id, v2.id)] = ir.Arc(
        distance(v1.coordinate, v2.coordinate), speed_limit=10000, constructible_chargers=dyn_chargers[(v1.id, v2.id)] if (v1.id, v2.id) in dyn_chargers else set()
    )

intermediate_rep_original = ir.IntermediateRepresentation(
    vertices=vertices, arcs=arcs, routes=[route1, route2]
)

ir_s_no_dev = simplify_intermediate_repr(intermediate_rep_original, allow_path_deviation=False)

ir_s_with_dev = simplify_intermediate_repr(intermediate_rep_original, allow_path_deviation=True)

inter_rep = ir_s_no_dev
battery_capacity = 192
stat_charging_rate = 3600
dyn_charging_rate = 3600


for k1, k2, arc in inter_rep.charger_edges:
    for c in arc.constructible_chargers:
        c.charging_rate=float(dyn_charging_rate)
for v in inter_rep.charger_nodes:
    for c in v.constructible_charger:
        c.charging_rate=float(stat_charging_rate)


@pytest.mark.parametrize(
    "deviation,capacity,fix_cost_dyn,fix_cost_stat,dyn_charging_rate,expected_value",
    [
        ("False", "192", "1", "100", "3600", "809.842712474"),
        ("False", "300", "1", "100", "3600", "683.42135623"),
        ("False", "300", "1", "100", "10000", "593.842712474"),
    ]
)
def test_integrated_network_creation(deviation, capacity, fix_cost_dyn, fix_cost_stat, dyn_charging_rate, expected_value):
    if deviation=="False":
        inter_rep = ir_s_no_dev
    else:
        inter_rep = ir_s_with_dev

    battery_capacity = int(capacity)
    stat_charging_rate = 3600
    dyn_charging_rate = float(dyn_charging_rate)

    for k1, k2, arc in inter_rep.charger_edges:
        for c in arc.constructible_chargers:
            c.charging_rate = float(dyn_charging_rate)
            c.transformer_construction_cost = float(fix_cost_dyn)
            # c.segment_construction_cost = float(dyn_var_cost)*1000
    for v in inter_rep.charger_nodes:
        for c in v.constructible_charger:
            c.charging_rate = float(stat_charging_rate)
            c.transformer_construction_cost = float(fix_cost_stat)

    instance_parameters = InstanceParameters(
        velocity=1000,
        consumption=lambda x: 1.0 if x==ir.VehicleID("Vehicle1") else 0.5,
        soc_init=float(battery_capacity), soc_max=float(battery_capacity),soc_min=0.0, energy_prices=1.0,
        consumption_cost=1.0, allow_path_deviations=False
    )
    solver_parameters = MathProgrammingParameters(None, 1, 1, num_replicas=1)
    solver = MathProgrammingSolver(instance_parameters, solver_parameters)
    sol_rep = solver.solve(inter_rep)
    assert sol_rep.global_cost == pytest.approx(float(expected_value), rel=1e-3)


@pytest.mark.parametrize(
    "fix_cost_dyn_spp,fix_cost_dyn_not_spp,fix_cost_stat_spp_no_stop,fix_cost_stat_not_spp,fix_cost_stat_spp_stop,expected_value",
    [
        ("1", "10000", "10000", "10000", "10000", "643.84271247"),
        ("10000", "3", "10000", "10000", "10000", "904.4769564638"),
        ("10000", "10000", "2", "10000", "10000", "862.620489777"),
        ("10000", "10000", "10000", "1", "10000", "961.62048977"),
        ("10000", "10000", "10000", "10000", "4", "823.199133"),
    ]
)
def test_integrated_network_creation_special_cases(fix_cost_dyn_spp,fix_cost_dyn_not_spp,fix_cost_stat_spp_no_stop,fix_cost_stat_not_spp,fix_cost_stat_spp_stop,expected_value):
    """The idea is to set the charging rate of all chargers so high that one charger suffices and optimal decision
    only depends on cost structure"""
    inter_rep = ir_s_with_dev
    battery_capacity = 1000
    stat_charging_rate = 1e6
    dyn_charging_rate = 1e6

    for k1, k2, arc in inter_rep.charger_edges:
        if "d1" in k1:
            for c in arc.constructible_chargers:
                c.charging_rate = float(dyn_charging_rate)
                c.transformer_construction_cost = float(fix_cost_dyn_not_spp)
        else:
            for c in arc.constructible_chargers:
                c.charging_rate = float(dyn_charging_rate)
                c.transformer_construction_cost = float(fix_cost_dyn_spp)
            # c.segment_construction_cost = float(dyn_var_cost)*1000

    for v in inter_rep.charger_nodes:
        if "c1" in v.id:
            for c in v.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_not_spp)
        elif "s2" in v.id:
            for c in v.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_spp_stop)
        elif "c2" in v.id:
            for c in v.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_spp_no_stop)

    instance_parameters = InstanceParameters(
        velocity=2000,
        consumption=lambda x: 1.0 if x=="Vehicle1" else 0.5,
        soc_init=0.25*float(battery_capacity),soc_max=float(battery_capacity),soc_min=0.0,
        energy_prices=1.0,consumption_cost=1.0, allow_path_deviations=True
    )
    solver_parameters = MathProgrammingParameters(None, 1, 60, num_replicas=1)
    solver = MathProgrammingSolver(instance_parameters, solver_parameters)
    sol_rep = solver.solve(inter_rep)
    assert sol_rep.global_cost == pytest.approx(float(expected_value), rel=1e-3)


@pytest.mark.parametrize(
    "fix_cost_dyn_spp,fix_cost_dyn_not_spp,fix_cost_stat_spp_no_stop,fix_cost_stat_not_spp,fix_cost_stat_spp_stop,expected_value",
    [
        ("10000", "10000", "10000", "1", "10000", "11070.84271"),
        ("6", "10000", "10000", "10000", "10000", "1076.8427"),
    ]
)
def test_integrated_network_multiple_replicas(fix_cost_dyn_spp,fix_cost_dyn_not_spp,fix_cost_stat_spp_no_stop,fix_cost_stat_not_spp,fix_cost_stat_spp_stop,expected_value):
    """With massive speed such that a vehicle can visit the same charger multiple times"""
    inter_rep = ir_s_with_dev
    battery_capacity = 330
    stat_charging_rate = 1e6
    dyn_charging_rate = 1e6

    for k1, k2, arc in inter_rep.charger_edges:
        if "d1" in k1:
            for c in arc.constructible_chargers:
                c.charging_rate = float(dyn_charging_rate)
                c.transformer_construction_cost = float(fix_cost_dyn_not_spp)
        else:
            for c in arc.constructible_chargers:
                c.charging_rate = float(dyn_charging_rate)
                c.transformer_construction_cost = float(fix_cost_dyn_spp)

    for v in inter_rep.charger_nodes:
        if "c1" in v.id:
            for c in v.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_not_spp)
        elif "s2" in v.id:
            for c in v.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_spp_stop)
        elif "c2" in v.id:
            for c in v.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_spp_no_stop)

    instance_parameters = InstanceParameters(velocity=10000,consumption=1,soc_init=0.7*float(battery_capacity),
                                             soc_max=float(battery_capacity),soc_min=0.0,
                                             energy_prices=1.0,consumption_cost=1.0, allow_path_deviations=True)
    solver_parameters = MathProgrammingParameters(None, 1, 60, num_replicas=1)
    solver = MathProgrammingSolver(instance_parameters, solver_parameters)
    sol_rep = solver.solve(inter_rep)

    assert sol_rep.global_cost == pytest.approx(float(expected_value), rel=1e-3)


@pytest.mark.parametrize(
    "fix_cost_dyn_spp,fix_cost_dyn_not_spp,fix_cost_stat_spp_no_stop,fix_cost_stat_not_spp,fix_cost_stat_spp_stop,u,v,expected_value",
    [
        ("1", "10000", "10000", "10000", "10000", "d3", "d4", "1052.4213562373095"), # -> d3-d4 should be solution
        ("10000", "3", "10000", "10000", "10000", "d1", "d2", "1183.7384782319225"), # d1-d2
        ("10000", "10000", "2", "10000", "10000", "c2", "c2", "862.6204902523968"), # c2
        ("10000", "10000", "10000", "1", "10000", "c1", "c1", "961.6204902523968"), # c1
        ("10000", "10000", "10000", "10000", "4", "s2", "s2", "823.1991340150873"), # s2
    ]
)
def test_solver_results(fix_cost_dyn_spp,fix_cost_dyn_not_spp,fix_cost_stat_spp_no_stop,fix_cost_stat_not_spp,fix_cost_stat_spp_stop,u,v,expected_value):
    """The idea is to set the charging rate of all chargers so high that one charger suffices and optimal decision
    only depends on cost structure"""
    # In General: Dyn not necessary same result as above but for static this should hold
    inter_rep = ir_s_with_dev
    battery_capacity = 1000
    stat_charging_rate = 1e6
    dyn_charging_rate = 1e5
    arc_chargers={}
    vertex_chargers={}

    for k1, k2, arc in inter_rep.charger_edges:
        if "d1" in k1:
            for c in arc.constructible_chargers:
                c.charging_rate = float(dyn_charging_rate)
                c.transformer_construction_cost = float(fix_cost_dyn_not_spp)
        else:
            for c in arc.constructible_chargers:
                c.charging_rate = float(dyn_charging_rate)
                c.transformer_construction_cost = float(fix_cost_dyn_spp)

    for vertex in inter_rep.charger_nodes:
        if "c1" in vertex.id:
            for c in vertex.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_not_spp)
        elif "s2" in vertex.id:
            for c in vertex.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_spp_stop)
        elif "c2" in vertex.id:
            for c in vertex.constructible_charger:
                c.charging_rate = float(stat_charging_rate)
                c.transformer_construction_cost = float(fix_cost_stat_spp_no_stop)

    if u==v:
        arc_chargers = {}
        vertex_chargers = {ir.VertexID(u): next(iter(inter_rep.get_vertex(ir.VertexID(u)).constructible_charger))}
    else:
        vertex_chargers = {}
        arc_chargers = {
            (ir.VertexID(u), ir.VertexID(v)): next(iter(inter_rep.get_arc(ir.VertexID(u), ir.VertexID(v)).constructible_chargers))
        }

    decision_variables_1 = DecisionVariables(inter_rep, vertex_chargers, arc_chargers, 2000, 1, 250, 1000, 0, 1.0, 1.0)
    decision_variables_2 = DecisionVariables(inter_rep, vertex_chargers, arc_chargers, 2000, 0.5, 250, 1000, 0, 1.0, 1.0)
    spprc_network_1 = decision_spprc_network(decision_variables_1, decision_variables_1.intermediate_rep.routes[0], 1)
    spprc_network_2 = decision_spprc_network(decision_variables_2, decision_variables_2.intermediate_rep.routes[1], 1)
    solution_subproblem_1_p = A_star(spprc_network_1, decision_variables_1, 0, False, "python")
    solution_subproblem_2_p = A_star(spprc_network_2, decision_variables_2, 1, False, "python")
    solution_subproblem_1_cpp = A_star(spprc_network_1, decision_variables_1, 0, False, "cpp")
    solution_subproblem_2_cpp = A_star(spprc_network_2, decision_variables_2, 1, False, "cpp")
    solution_subproblem_p = solution_subproblem_1_p+solution_subproblem_2_p
    solution_subproblem_cpp = solution_subproblem_1_cpp + solution_subproblem_2_cpp
    assert solution_subproblem_p.global_cost == pytest.approx(float(expected_value), rel=1e-3)
    assert solution_subproblem_cpp.global_cost == pytest.approx(float(expected_value), rel=1e-3)