import math
import copy
import random
from itertools import permutations, pairwise
from collections import defaultdict
from typing import Dict, Tuple, Set, List
from framework.intermediate_representation import Charger, ChargerID, Vertex, VertexID, Arc, ArcID
import framework.intermediate_representation as ir
from framework.utils import Coordinate
from pathlib import Path


segment_construction_cost = 1e-4
transformer_construction_cost_stat = (15.0, 25.0)
transformer_construction_cost_dyn = (1.5,2.5)
num_dyn_charger = [2, 3, 4]
speed_limit = math.inf


def parse_sequences(solution_path: Path, instance_name: str):
    with open(solution_path) as instance_stream:
        instance_stream.readline()

        next_line = instance_stream.readline()
        while next_line.split()[0] != instance_name:
            next_line = instance_stream.readline()

        configuration = next_line.split()[-1]
        configuration = configuration.removeprefix('D0;')
        configuration = configuration.removesuffix(';D0')
        sequences = configuration.split(';D0;D0;')
    return sequences


def create_routes(solution_path: Path, instance_name: str, stops: dict[str, ir.Stop], max_routes=None):
    sequences = parse_sequences(solution_path, instance_name)
    max_routes = max_routes if max_routes is not None else len(sequences)
    routes = []
    solution_chargers = set()
    for index, sequence in enumerate(sequences[:max_routes:]):
        stop_sequence = []
        vehicle_id = 'Route' + str(index)
        stop_sequence.append(copy.copy(stops['Depot']))
        for element in sequence.split(";"):
            if element[0] != 'S':
                stop_sequence.append(copy.copy(stops[element]))
            else:
                solution_chargers.add(element)
        stop_sequence.append(copy.copy(stops['Depot']))
        routes.append(ir.Route(stop_sequence=stop_sequence, vehicle_id=ir.VehicleID(vehicle_id)))
    return routes, stops, solution_chargers


def create_vertices(instance_path: Path, vertices: Set[Vertex]):
    vertices = {v.id: v for v in vertices}
    stops_dict = {}
    parameters = {}
    with open(instance_path, "r") as file:
        lines = file.readlines()

    for line in lines[1:]:
        values = line.split()

        if len(values)==0:
            continue

        if len(values) >= 3 and "/" in values[-1]:  # Detect parameter lines
            key = values[0]  # The first element is the key
            value = float(values[-1].strip("/"))  # Extract numerical value
            parameters[key] = value
            continue

        if values[1] == 'f':
            continue

        # in this case we need to add the stop
        if values[1] in ["c", "d"]:
            is_depot = values[1] == 'd'
            id = 'Depot' if is_depot else values[0]
            is_stop = not is_depot
            stops_dict[id] = ir.Stop(vertex_id=VertexID(id), stopover_time=0.,
                                     earliest_time_of_service=float(values[5])*3600,
                                     latest_time_of_service=float(values[6])*3600)

        if values[0] in vertices:
            continue

        assert values[1] in ["c", "d"], f"{values[1]}"

        coordinate = Coordinate(lat=float(values[2]), lon=float(values[3]))
        vertices[id] = ir.Vertex(id=VertexID(id), is_depot=is_depot, is_stop=is_stop, coordinate=coordinate,
                                     constructible_charger=set(), name=id)

    return vertices, stops_dict, parameters


def euclidean(u: ir.Vertex, v: ir.Vertex) -> float:
    return math.sqrt((u.coordinate.lon - v.coordinate.lon) ** 2 + (u.coordinate.lat - v.coordinate.lat) ** 2)


def add_arc(
        u: ir.Vertex,
        v: ir.Vertex,
        arc_speed_limit: float,
        arcs: Dict[Tuple[ir.VertexID, ir.VertexID], ir.Arc],
        arc_charger_mapping: Dict[Tuple[VertexID, VertexID], ChargerID],
        chargers: Dict[ChargerID, ir.Charger]
):
    """Works in place"""
    try:
        c_id = arc_charger_mapping[(u.id, v.id)]
        chs = {chargers[c_id]}
    except KeyError:
        chs = set()
    arcs[u.id, v.id] = ir.Arc(distance=euclidean(u, v),
                              speed_limit=arc_speed_limit,
                              constructible_chargers=chs)


def create_arcs(vertices: dict[str, ir.Vertex], routes, arc_speed_limit, arc_charger_mapping, chargers):
    arcs = {}
    consec_stops = [(s1.vertex_id, s2.vertex_id) for r in routes for (s1,s2) in pairwise(r.stop_sequence)]

    # map auxiliary nodes to chargers
    node_charger_mapping = {}
    for (v1, v2), charger_id in arc_charger_mapping.items():
        node_charger_mapping[v1] = charger_id
        node_charger_mapping[v2] = charger_id

    charger_arc_mapping = defaultdict(set)
    for (u,v), charger in arc_charger_mapping.items():
        charger_arc_mapping["_".join(u.split("_")[:-1])].add(int(u.split("_")[-1]))
        charger_arc_mapping["_".join(v.split("_")[:-1])].add(int(v.split("_")[-1]))

    for vertex_from, vertex_to in permutations(vertices.values(), 2):

        # These exclusions are valid because the possible connections are not allowed in the methods either
        # Concrete: In the spprc networks and model networks, charger to charger connections are not reflected
        # exclude charger to charger where both chargers are not stops
        if vertex_from.is_standalone_charger and vertex_to.is_standalone_charger:
            continue

        # exclude static charger to auxiliary and vice versa
        if vertex_from.is_standalone_charger and vertex_to.is_auxiliary:
            continue

        # the vice versa case
        if vertex_to.is_standalone_charger and vertex_from.is_auxiliary:
            continue

        # exclude stop/depot to stop/depot where stops are not consecutive on route
        if vertex_from.only_stop_or_depot and vertex_to.only_stop_or_depot and not (vertex_from.id, vertex_to.id) in consec_stops:
            continue

        # two auxiliary nodes must belong to the same charger to be connected
        # we accept the unnecessary arcs within a single charger, preprocessing will take care of them
        if vertex_from.is_auxiliary and vertex_to.is_auxiliary and \
                not (node_charger_mapping[vertex_from.id] == node_charger_mapping[vertex_to.id]):
            continue

        # do not connect any node except one to first or last segment
        if vertex_from.is_auxiliary and vertex_from.id.split("_")[-1]=="0" and not (vertex_from.id,vertex_to.id) in arc_charger_mapping:
            continue

        if vertex_to.is_auxiliary and int(vertex_to.id.split("_")[-1])==max(charger_arc_mapping["_".join(vertex_to.id.split("_")[:-1])]) and not (vertex_from.id,vertex_to.id) in arc_charger_mapping:
            continue

        add_arc(vertex_from, vertex_to, arc_speed_limit, arcs, arc_charger_mapping, chargers)
    return arcs


def parse_chargers(charger_path) -> Dict[ChargerID, Charger]:
    charger_dict = {}
    with open(charger_path, "r") as file:
        lines = file.readlines()

    # Parse each line after the header
    for line in lines[1:]:
        values = line.split()
        charger_dict[ChargerID(values[0])] = Charger(
            id=ChargerID(values[0]), segment_construction_cost=float(values[2]), transformer_construction_cost=float(values[1]),
            charging_rate=float(values[3])
        )
    return charger_dict


def parse_locations(location_path, chargers) -> Tuple[Set[Vertex], Dict[Tuple[VertexID, VertexID], ChargerID]]:
    vertices = set()

    with open(location_path, "r") as file:
        lines = file.readlines()

    # Parse stationary chargers
    for line in lines[1:]:
        values = line.split()
        if values[1]=="x":
            continue
        is_stop = False
        if values[1] == "ic":
            is_stop=True
        try:
            chs = {chargers[values[4]]}
        except KeyError:
            chs = set()
        vertices.add(
            Vertex(
                VertexID(values[0]), False, is_stop, Coordinate(lat=float(values[2]), lon=float(values[3])),
                constructible_charger=chs
            )
        )
    # Parse segment nodes
    charger_arc_mapping = defaultdict(list)
    for line in lines[1:]:
        values = line.split()
        if values[1]!="x":
            continue
        charger_arc_mapping[values[4]].append((values[0], int(values[5])))
        vertices.add(
            Vertex(
                VertexID(values[0]), False, False, Coordinate(lat=float(values[2]), lon=float(values[3])),
                constructible_charger=set()
            )
        )

    # Sort each list of tuples by the second element
    for key in charger_arc_mapping:
        charger_arc_mapping[key] = sorted(charger_arc_mapping[key], key=lambda x: x[1])

    # convert to arc -> charger ID mapping
    arc_charger_mapping = {}
    for charger, v_list in charger_arc_mapping.items():
        for t1, t2 in pairwise(v_list):
            arc_charger_mapping[(t1[0], t2[0])] = charger
    return vertices, arc_charger_mapping


def read_instance(instance_name: str, recharge_degree: str, max_routes=None):
    instance_path = Path('./Instances/evrptw_instances/' + instance_name + '.txt')
    charger_info = Path('./Instances/evrptw_instances/' + instance_name + '_AddInfraStruct.txt')
    location_info = Path('./Instances/evrptw_instances/' + instance_name + '_AddLoc.txt')
    solution_path = Path('./Instances/EVRPTW' + recharge_degree + '_Distance_details.txt')

    chargers = parse_chargers(charger_info)
    vertices, arc_charger_mapping = parse_locations(location_info, chargers)

    vertices, stops_dict, parameters = create_vertices(instance_path, vertices)
    routes, stops, solution_chargers = create_routes(solution_path, instance_name, stops_dict, max_routes=max_routes)
    velocity = float(parameters['v'])
    arcs = create_arcs(vertices, routes, velocity, arc_charger_mapping, chargers)

    return ir.IntermediateRepresentation(vertices.values(), arcs, routes), parameters


if __name__ == "__main__":
    instance_name = "C102"
    ir, paras = read_instance(instance_name,"PR",None)

