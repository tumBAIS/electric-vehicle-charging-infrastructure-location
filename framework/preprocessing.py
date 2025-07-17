# coding: utf-8
#import copy
import logging
import math
import random
import warnings
import networkx as nx
from typing import Tuple, List, Dict, Set
from itertools import pairwise, product

from framework.intermediate_representation import (
    Arc,
    ArcID,
    Vertex,
    VertexID,
    IntermediateRepresentation,
)
from framework.utils import sort_tuples

random.seed(9002)

# some type aliases
OSMNetwork = nx.MultiDiGraph
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def _identify_dynamic_segment_borders(
        inter_rep: IntermediateRepresentation,
) -> Tuple[List[Vertex], List[Vertex]]:
    """
    Identify the vertices that were added to the network as "borders" of dyn. charging segments
    :param inter_rep: Graph in intermediate representation
    :return: List with all vertices that are connected to dyn. charger segment (outgoing, ingoing)
    """
    out_vertices = []
    in_vertices = []
    charger_edges_by_charger = {}
    for u, v, a in inter_rep.charger_edges:
        c = next(iter(a.constructible_chargers))
        if c in charger_edges_by_charger:
            charger_edges_by_charger[c].append((u, v))
        else:
            charger_edges_by_charger[c] = [(u, v)]
    for c in charger_edges_by_charger:
        charger_edges_by_charger[c] = sort_tuples(charger_edges_by_charger[c])
    for c, arc_list in charger_edges_by_charger.items():
        out_vertices.append(inter_rep.get_vertex(arc_list[0][0]))
        in_vertices.append(inter_rep.get_vertex(arc_list[-1][1]))
    return out_vertices, in_vertices


def _identify_candidate_vertices_deviation(
        inter_rep: IntermediateRepresentation,
) -> List[Tuple[List[Vertex], List[Vertex]]]:
    """
    Identify all (origin, target) combinations of vertices that could potentially be part of the optimal vehicle routes
    :param inter_rep: Graph in intermediate representation
    :return: Source and Targets to evaluate with shortest path method to identify valid vertices
    """
    return_list = []
    for route in inter_rep.routes:
        for s1, s2 in pairwise(route.stop_sequence):
            return_list.append(([inter_rep.get_vertex(s1.vertex_id)], [inter_rep.get_vertex(s2.vertex_id)]))

    dyn_charger_out, dyn_charger_in = _identify_dynamic_segment_borders(inter_rep)

    # case 2
    return_list.append(
        (
            [v for v in inter_rep.vertices if (v.is_stop or v.is_depot)],
            [
                *dyn_charger_out,
                *inter_rep.charger_nodes,
            ],
        ),
    )

    # case 3
    return_list.append(
        (
            [
                *dyn_charger_in,
                *inter_rep.charger_nodes,
            ],
            [v for v in inter_rep.vertices if (v.is_stop or v.is_depot)],
        ),
    )

    # case 4
    for (v1, v2, _) in inter_rep.charger_edges:
        return_list.append(([inter_rep.get_vertex(v1)], [inter_rep.get_vertex(v2)]))

    return_list = [(x, y) for x, y in return_list if len(x) > 0 and len(y) > 0]

    # assert all combinations exist at least once
    for t1, t2 in return_list:
        assert (len(t1) > 0 and len(t2) > 0)

    return return_list


def _identify_candidate_vertices_no_deviation(
        inter_rep: IntermediateRepresentation,
) -> List[Tuple[List[Vertex], List[Vertex]]]:
    """
    Identify all (origin, target) combinations of vertices that could potentially be part of the optimal vehicle routes
    :param inter_rep: Graph in intermediate representation
    :return: Source and Targets to evaluate with shortest path method to identify valid vertices
    """
    return_list = []
    for route in inter_rep.routes:
        for s1, s2 in pairwise(route.stop_sequence):
            return_list.append(([inter_rep.get_vertex(s1.vertex_id)], [inter_rep.get_vertex(s2.vertex_id)]))

    return_list = [(x, y) for x, y in return_list if len(x) > 0 and len(y) > 0]

    # assert all combinations exist at least once
    for t1, t2 in return_list:
        assert (len(t1) > 0 and len(t2) > 0)

    return return_list


def _identify_candidate_vertices(
        inter_rep: IntermediateRepresentation,
        allow_path_deviation: bool,
) -> List[Tuple[List[Vertex], List[Vertex]]]:
    """
    Identify all (origin, target) combinations of vertices that could potentially be part of the optimal vehicle routes
    :param inter_rep: Graph in intermediate representation
    :param allow_path_deviation: Boolean indicating if deviation from the shortest path is allowed
    :return: Source and Targets to evaluate with shortest path method to identify valid vertices
    """
    if allow_path_deviation:
        return _identify_candidate_vertices_deviation(inter_rep)
    return _identify_candidate_vertices_no_deviation(inter_rep)


def _arcs_between_vertices(
        inter_rep: IntermediateRepresentation, vertices: List[VertexID]
) -> List[Tuple[VertexID, VertexID, Arc]]:
    """
    Extract list of arcs on a path (i.e., sequence of nodes) from a given intermediate representation.
    :param inter_rep: Intermediate representation containing path specified by parameter 'vertices'
    :param vertices: Path trough intermediate network representation
    :return: List of arcs in specified path
    """
    arcs = []
    for v1, v2 in pairwise(vertices):
        arcs.append((v1, v2, inter_rep.get_arc(v1, v2)))
    return arcs


def _consolidate_arcs(l: List[Tuple[str, str, Arc]]) -> Tuple[float, float]:
    """return consolidated distance, speed limit pair"""
    d = sum(a.distance for u,v,a in l)
    if d==0:
        warnings.warn(f"We consolidated to a zero distance arc! {l}")
        return l[0][2].distance, l[0][2].speed_limit
    sl = sum(a.distance*a.speed_limit for u,v,a in l)/d
    return d, sl


def reduce_ir_network(ir: IntermediateRepresentation) -> IntermediateRepresentation:
    """Remove all superfluous nodes from the network of the given IR (particularly relevant when no deviations are
    allowed - only to use when a separate IR is utilised for every vehicle (model)"""
    assert len(ir.routes)==1
    superfluous_vertices = set()
    segment_vertices = {node for u, v, _ in ir.charger_edges for node in (u, v)}
    route_vertices = {v.id for v in ir.vertices if v.is_depot or v.is_stop}
    pairwise_route_vertices = [(u.vertex_id,v.vertex_id) for u,v in pairwise(ir.routes[0].stop_sequence)]
    for v in ir.vertices:
        if v.is_stop or v.is_depot:
            continue
        elif v.id not in segment_vertices:
            # check if any predecessor / successor is a stop or depot
            if len(set(ir.get_predecessors(v)) & route_vertices)==0 or len(set(ir.get_successors(v)) & route_vertices)==0:
                superfluous_vertices.add(v)

    # this removes all former stop nodes (i.e. stops of other routes and other auxiliary nodes)
    ir.remove_vertices(superfluous_vertices)

    # there might be some dead ends left --> recollect them
    while True:
        superfluous_vertices = set()
        found = False
        for v in ir.vertices:
            if len(ir.get_adjacent_arcs(v)) <= 1:
                superfluous_vertices.add(v)
                found = True
        ir.remove_vertices(superfluous_vertices)
        if not found:
            break

    # also remove vertices that have been added because they were on the shortest path
    superfluous_vertices = set()
    for vertex in ir.vertices:
        if (not vertex.can_construct_charger) or vertex.is_stop or vertex.is_depot:
            continue
        valid_charger = False
        for (u,v) in product(ir.get_predecessors(vertex), ir.get_successors(vertex)):
            if (u,v) in pairwise_route_vertices:
                valid_charger = True
                break
        if not valid_charger:
            superfluous_vertices.add(vertex)

    # this removes all former stop nodes (i.e. stops of other routes and other auxiliary nodes)
    ir.remove_vertices(superfluous_vertices)

    # remaining superfluous nodes are segment nodes which are not weakly connected to the route network anymore
    ir_route_network = ir.get_weakly_connected_nodes(ir.routes[0].stop_sequence[1].vertex_id)

    # redefine set of add. superfluous vertices
    superfluous_vertices = set(ir.vertices) - {ir.get_vertex(v) for v in ir_route_network}

    # this removes all former stop nodes (i.e. stops of other routes and other auxiliary nodes
    ir.remove_vertices(superfluous_vertices)

    # resort graph
    ir.sort_graph_representation()
    return ir


def _consolidate_subset_of_intermediate_repr(
        inter_rep: IntermediateRepresentation,
        vertices_from: List[Vertex],
        vertices_to: List[Vertex],
        current_arcs: Dict[Tuple[VertexID,VertexID],Arc],
) -> Tuple[Set[Vertex], Dict[ArcID, Arc]]:
    """
    Create new simplified intermediate representation
    The entire functions suffers a bit from the fact that we allowed zero distance arcs.
    """
    vertices = {*vertices_to, *vertices_from}
    arcs = {}
    for vertice_from in vertices_from:
        for vertice_to in vertices_to:
            if (vertice_from.id, vertice_to.id) in current_arcs:
                continue
            if vertice_from == vertice_to:
                continue
            else:
                path = inter_rep.calc_shortest_path(
                    origin=vertice_from,
                    target=vertice_to,
                )
                arcs_on_path = _arcs_between_vertices(inter_rep, path)

                # extract mapping between charger objects and arcs
                arcs_by_chargers = {}
                for idx, (u,v,arc) in enumerate(arcs_on_path):
                    if arc.can_construct_charger:
                        charger = next(iter(arc.constructible_chargers))
                        if charger in arcs_by_chargers:
                            arcs_by_chargers[charger].append((u,v,arc,idx))
                        else:
                            arcs_by_chargers[charger] = [(u,v,arc,idx)]

                # loop over all vertices on path and connect accordingly
                for idx, (u,v,arc) in enumerate(arcs_on_path[0:-1]):
                    vertex=inter_rep.get_vertex(v)
                    if vertex.can_construct_charger:
                        vertices.add(vertex)
                        if vertice_from.is_stop or vertice_from.is_depot:
                            distance, speed_limit = _consolidate_arcs(arcs_on_path[:idx + 1])
                            arcs[(vertice_from.id, v)] = Arc(
                                distance=distance,
                                speed_limit=speed_limit,
                                constructible_chargers=set()
                            )
                        if vertice_to.is_stop or vertice_to.is_depot:
                            distance, speed_limit = _consolidate_arcs(arcs_on_path[idx+1:])
                            arcs[(v, vertice_to.id)] = Arc(
                                distance=distance,
                                speed_limit=speed_limit,
                                constructible_chargers=set()
                            )

                # now do the arc chargers
                for charger, arc_list in arcs_by_chargers.items():
                    for u,v,arc,_ in arc_list:
                        arcs[(u,v)] = arc
                        vertices.add(inter_rep.get_vertex(u))
                        vertices.add(inter_rep.get_vertex(v))

                    # if deviations are allowed, all segments are passed as (vertice_from, vertice_to) pair individually
                    if not vertice_from.id==arc_list[0][0] and (vertice_from.is_stop or vertice_from.is_depot):
                        distance, speed_limit = _consolidate_arcs(arcs_on_path[:arc_list[0][3]])
                        arcs[(vertice_from.id, arc_list[0][0])] = Arc(
                            distance=distance,
                            speed_limit=speed_limit,
                            constructible_chargers=set()
                        )
                    if not vertice_to.id==arc_list[-1][1] and (vertice_to.is_stop or vertice_to.is_depot):
                        distance, speed_limit = _consolidate_arcs(arcs_on_path[arc_list[-1][3]+1:])
                        arcs[(arc_list[-1][1], vertice_to.id)] = Arc(
                            distance=distance,
                            speed_limit=speed_limit,
                            constructible_chargers=set()
                        )

                # direct connection always needed (but only add if not yet happended)
                if (vertice_from.id, vertice_to.id) in arcs:
                    continue

                distance, speed_limit = _consolidate_arcs(arcs_on_path)
                arcs[(arcs_on_path[0][0], arcs_on_path[-1][1])] = Arc(
                    distance=distance,
                    speed_limit=speed_limit,
                    constructible_chargers=set(),
                )
    return vertices, arcs


def get_vertex(vertices: set[Vertex], vertex_id: str):
    for v in vertices:
        if v.id == vertex_id:
            return v
    raise ValueError("No vertex with vertex id " + vertex_id + " found in vertices set!")


def simplify_intermediate_repr(inter_rep: IntermediateRepresentation, allow_path_deviation: bool) -> IntermediateRepresentation:
    """
    Simplify intermediate representation by consolidating shortest paths to single arcs and disregarding
    all vertices not on any shortest path.
    @param inter_rep: Intermediate representation to be simplified
    @param allow_path_deviation: boolean indicating if deviations from the shortest path should be allowed for charging
    @return: Simplified intermediate representation
    """

    vertices = set()
    arcs = dict()
    inter_rep.sort_graph_representation()
    candidates = _identify_candidate_vertices(inter_rep, allow_path_deviation)
    for _from, _to in candidates:
        v_temp, a_temp = _consolidate_subset_of_intermediate_repr(inter_rep, _from, _to, arcs)
        vertices.update(v_temp)
        arcs.update(a_temp)
    result = IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=inter_rep.routes)

    # some checks
    for u,v,arc in result.arcs:
        u_vertex = result.get_vertex(u)
        v_vertex = result.get_vertex(v)
        # one of the two must be stop/depot or arc charger in the middle
        if not arc.can_construct_charger:
            assert len([vertex for vertex in {u_vertex, v_vertex} if vertex.is_stop or vertex.is_depot]) >= 1
    return result


def preprocess_time_windows_in_inter_rep(
        inter_rep: IntermediateRepresentation, vehicle_maxspeed: float
) -> IntermediateRepresentation:
    """
    Backpropagate latest departure time to retain feasibility across all routes.
    @param inter_rep: IntermediateRepresentation
    @param vehicle_maxspeed: Homogenous vehicle maxspeed
    @return: inter_rep: IntermediateRepresentation with smaller timewindows (unchanged earliest time of service)
    """
    logging.info("--Preprocess time windows--")
    changed = False
    for route in inter_rep.routes:
        for to_, from_ in pairwise(reversed(route.stop_sequence)):
            # calculate shortest path from source to target
            sp = inter_rep.get_arc(from_.vertex_id, to_.vertex_id)
            # multiply with speed (minimum from speed profile or max speed from osmnx or vehicle_maxspeed
            tt = sp.get_travel_time_seconds(vehicle_maxspeed)
            if to_.latest_time_of_service - from_.latest_time_of_service < tt:
                logging.warning(f"Cut latest departure time at {from_.vertex_id} from {from_.latest_time_of_service} "
                                f"to {math.ceil(to_.latest_time_of_service-tt)}")
                from_.latest_time_of_service = math.floor(max(to_.latest_time_of_service - tt,0))
                if from_.earliest_time_of_service > to_.latest_time_of_service - tt:
                    raise ValueError(f"Route {route.vehicle_id} is time infeasible")
                changed = True
    if changed:
        inter_rep.update_routes()
        inter_rep.revalidate()
        logging.info("--Finished Preprocessing by call to validator of intermediate representation again--")
    return inter_rep

