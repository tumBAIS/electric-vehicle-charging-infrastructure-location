# coding=utf-8
import warnings
import random
import networkx as nx
import itertools
import copy
import math
import hashlib
from typing import Optional, Union, Iterable, Tuple, Dict, List
from itertools import pairwise
import framework.intermediate_representation as ir
from docplex.mp.solution import SolveSolution
from framework.intermediate_representation import IntermediateRepresentation, split_into_vehicle_networks, VertexID
from math_programming_model.model import DynamicChargerLRPParameters, DynamicChargerLRP
import math_programming_model.model as mdl
from framework.preprocessing import reduce_ir_network, simplify_intermediate_repr
import math_programming_model.network as mn
from math_programming_model.solution import Solution, converting_solution_object
from dataclasses import dataclass
from framework.solver import SolverParameters, Solver
from framework.instance_parameter import InstanceParameters
from framework.solution_representation import SolutionRepresentation

random.seed(9002)


def str_to_int_deterministic(s):
    """Return deterministic 8 digit int"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 10 ** 8  # Use md5, sha256, etc.

@dataclass(frozen=True)
class MathProgrammingParameters(SolverParameters):
    num_replicas: int


def _convert_ir_charger_to_mn_charger(ir_c: ir.Charger, is_static: bool) -> Union[mn.StaticCharger, mn.DynamicCharger]:
    id_ = str_to_int_deterministic(ir_c.id)
    if is_static:
        return mn.StaticCharger(id=id_,construction_cost=ir_c.transformer_construction_cost+ir_c.segment_construction_cost,
                                charging_rate_kwh_per_h=ir_c.charging_rate, irID=ir_c.id)
    return mn.DynamicCharger(id=id_, construction_cost_per_km=ir_c.segment_construction_cost*1000,
                             transformer_construction_cost=ir_c.transformer_construction_cost,
                             charging_rate_kwh_per_h=ir_c.charging_rate, irID=ir_c.id)


def _derive_vertex_type(vertex: ir.Vertex) -> mn.VertexType:
    """
    We encode different vertices in the problem as follows:
    STOP --> all stop representations (may be dummy nodes, i.e., a dummy of another node)
    CHARGER --> all stationary charging nodes (may be stop as well, never auxiliary or depot)
    DEPOT --> only depot
    AUXILIARY --> segment start / end nodes
    """
    v_type = mn.VertexType.AUXILIARY
    if vertex.is_stop:
        v_type |= mn.VertexType.STOP
    if vertex.is_depot:
        v_type |= mn.VertexType.DEPOT
    if vertex.can_construct_charger:
        v_type |= mn.VertexType.CHARGER

    # Remove auxiliary type if any other type is set
    if v_type != mn.VertexType.AUXILIARY:
        v_type &= ~mn.VertexType.AUXILIARY
    return v_type


class MathProgrammingSolver(Solver):
    solver_parameters: MathProgrammingParameters
    warmstart_solution: Optional[str]

    def __init__(
            self,
            instance_parameters: InstanceParameters,
            solver_parameters: MathProgrammingParameters,
            warmstart_solution: Optional[str] = None
    ):
        super().__init__(instance_parameters, solver_parameters)
        self.warmstart_solution = warmstart_solution

    def solve(self, intermediate_representation: IntermediateRepresentation) -> Optional[SolutionRepresentation]:
        return self._solve(intermediate_representation)[0]

    def _solve(
            self, intermediate_representation: IntermediateRepresentation
    ) -> Tuple[Optional[SolutionRepresentation], Optional[SolveSolution]]:
        vehicles = {
            r.vehicle_id: ir.Vehicle(
                vehicle_id=r.vehicle_id,
                max_speed=self.instance_parameters.max_speed(r.vehicle_id),
                consumption=self.instance_parameters.max_consumption(r.vehicle_id)
            ) for r in intermediate_representation.routes
        }
        solution, cplex_solution = self.solve_problem_with_mip(intermediate_representation,vehicles)
        if solution is None:
            return None, None
        solution_representation = SolutionRepresentation(
            *converting_solution_object(solution, self.solver_parameters.time_step_in_seconds)
        )
        return solution_representation, cplex_solution

    def _create_model_network(
            self,
            ir_vehicle_network: IntermediateRepresentation,
            vehicle: ir.Vehicle,
            charger_map: dict[ir.Charger, mn.DynamicCharger],
    ) -> mn.VehicleNetwork:
        assert len(ir_vehicle_network.routes) == 1
        route = ir_vehicle_network.routes[0]
        vehicle_network = nx.DiGraph()

        # this mapping retains MN Vertex ID <--> Vertex Object (1:1)
        id_to_mn_vertex: dict[mn.VertexID, mn.Vertex] = {}

        # this mapping retains MN Vertex ID <--> IR Vertex Object (1:1)
        id_to_ir_vertex: dict[mn.VertexID, ir.Vertex] = {}

        # dummy_mapping: maps IR Vertex IDs to all MN Representations
        dummy_mapping: dict[ir.VertexID, list[mn.Vertex]] = {ir_v.id: [] for ir_v in ir_vehicle_network.vertices}

        # Add start depot vertices
        start_depot_id = ir_vehicle_network.depot.id + '_START'
        start_depot_vertex = mn.Vertex(
            id=mn.VertexID(start_depot_id),
            type=_derive_vertex_type(ir_vehicle_network.depot),
            time_window_begin=route.earliest_departure_time,
            time_window_end=route.latest_arrival_time,
            charger=charger_map.get(next(iter(ir_vehicle_network.depot.constructible_charger)))
            if ir_vehicle_network.depot.can_construct_charger else None,
            name=ir_vehicle_network.depot.name  # Name does not need suffix
        )
        vehicle_network.add_node(start_depot_vertex)
        id_to_mn_vertex[mn.VertexID(start_depot_id)] = start_depot_vertex
        dummy_mapping[ir_vehicle_network.depot.id].append(start_depot_vertex)
        id_to_ir_vertex[mn.VertexID(start_depot_id)] = ir_vehicle_network.depot

        # Create sequence of vertices representing the stops on the vehicles route
        stop_counter ={s.vertex_id: 0 for s in ir_vehicle_network.routes[0].stop_sequence}
        for s1 in ir_vehicle_network.routes[0].stop_sequence[1:-1]:
            if stop_counter[s1.vertex_id] > 0:
                new_vertex_id = mn.VertexID(f"{s1.vertex_id}_{vehicle.vehicle_id}_{stop_counter[s1.vertex_id]}")
            else:
                new_vertex_id = mn.VertexID(s1.vertex_id)
            new_vertex = mn.Vertex(
                    id = new_vertex_id,
                    type = _derive_vertex_type(ir_vehicle_network.get_vertex(s1.vertex_id)),
                    time_window_begin = s1.earliest_time_of_service,
                    time_window_end = s1.latest_time_of_service,
                    charger=charger_map.get(
                        next(iter(ir_vehicle_network.get_vertex(s1.vertex_id).constructible_charger))
                    ) if ir_vehicle_network.get_vertex(s1.vertex_id).can_construct_charger else None,
                    name=str(new_vertex_id),
                    dummy_of=id_to_mn_vertex[mn.VertexID(s1.vertex_id)] \
                        if stop_counter[s1.vertex_id]>0 else None,
                    replica_level=0,
                )
            id_to_mn_vertex[new_vertex_id] = new_vertex
            dummy_mapping[s1.vertex_id].append(new_vertex)
            id_to_ir_vertex[new_vertex_id] = ir_vehicle_network.get_vertex(s1.vertex_id)
            vehicle_network.add_node(new_vertex)
            stop_counter[s1.vertex_id] += 1

        # Add end depot vertices
        end_depot_id = ir_vehicle_network.depot.id + '_END'
        end_depot_vertex = mn.Vertex(
            id=mn.VertexID(end_depot_id),
            type=_derive_vertex_type(ir_vehicle_network.depot),
            time_window_begin=route.earliest_departure_time,
            time_window_end=route.latest_arrival_time,
            charger=charger_map.get(next(iter(ir_vehicle_network.depot.constructible_charger)))
            if ir_vehicle_network.depot.can_construct_charger else None,
            name=ir_vehicle_network.depot.name  # Name does not need suffix
        )
        vehicle_network.add_node(end_depot_vertex)
        id_to_mn_vertex[mn.VertexID(end_depot_id)] = end_depot_vertex
        dummy_mapping[ir_vehicle_network.depot.id].append(end_depot_vertex)
        id_to_ir_vertex[mn.VertexID(end_depot_id)] = ir_vehicle_network.depot

        # save id_to_mn_vertex keys as route in model network namespace
        mn_route = list(id_to_mn_vertex.values())

        # add remaining nodes (either static chargers or start/end nodes of segments)
        for v in ir_vehicle_network.vertices:
            if v.is_depot:
                continue
            if v.id in vehicle_network:
                continue
            new_vertex = mn.Vertex(
                id=mn.VertexID(v.id),
                type=_derive_vertex_type(v),
                time_window_begin=0,
                time_window_end=1e9, #math.inf,
                charger=charger_map.get(next(iter(v.constructible_charger)))
                if v.can_construct_charger else None,
                name=v.name,
                replica_level=0,
            )
            id_to_mn_vertex[mn.VertexID(v.id)] = new_vertex
            id_to_ir_vertex[mn.VertexID(v.id)] = v
            dummy_mapping[v.id].append(new_vertex)
            vehicle_network.add_node(new_vertex)

        # 2. Iterate over all IR edges and add them to the Vehicle Network (including such where one or two components
        # is duplicated in the Vehicle Network
        for u,v,ir_arc in ir_vehicle_network.arcs:

            for mn_vertex_u in dummy_mapping[u]:
                # no outgoing arc from depot end representation
                if mn_vertex_u.is_depot and "_END" in mn_vertex_u.id:
                    continue
                for mn_vertex_v in dummy_mapping[v]:

                    # arc exists already
                    if vehicle_network.has_edge(mn_vertex_u, mn_vertex_v):
                        continue

                    # no incoming arc in depot start representation
                    if mn_vertex_v.is_depot and "_START" in mn_vertex_v.id:
                        continue

                    new_arc = mn.Arc(
                        origin=mn_vertex_u,
                        target=mn_vertex_v,
                        max_travel_time=ir_arc.get_travel_time_hours(vehicle.max_speed),
                        distance=ir_arc.distance,
                        min_consumption=ir_arc.get_consumption(vehicle.consumption),
                        max_consumption=ir_arc.get_consumption(vehicle.consumption),
                        charger=charger_map.get(next(iter(ir_arc.constructible_chargers)))
                        if ir_arc.can_construct_charger else None,
                    )
                    vehicle_network.add_edge(mn_vertex_u, mn_vertex_v, arc=new_arc)

        assert nx.is_weakly_connected(vehicle_network)

        # 3. Connect depot representations (see patricks code); such that all nodes have degree >= 2
        arc = mn.Arc(
            origin=start_depot_vertex,
            target=end_depot_vertex,
            max_travel_time=0.0,
            distance=0.0,
            min_consumption=0.0,
            max_consumption=0.0,
            charger=None
        )
        vehicle_network.add_edge(
            start_depot_vertex,
            end_depot_vertex,
            arc=arc
        )

        # 4. Replicate chargers (is also needed in the no_deviation case)
        min_required_replicas = max(stop_counter.values())-1
        self._duplicate_static_chargers(
            vehicle_network=vehicle_network,
            n_repl=max(min_required_replicas, self.solver_parameters.num_replicas),
        )
        dyn_charger_to_replica = self._duplicate_dynamic_chargers(
            ir_vehicle_network,
            vehicle_network=vehicle_network,
            n_repl=max(min_required_replicas, self.solver_parameters.num_replicas)
        )

        # up to here the graph has to be weakly connected
        assert nx.is_weakly_connected(vehicle_network)
        if not self.instance_parameters.allow_path_deviations:
            print(f"-------- Model network for {ir_vehicle_network.routes[0].vehicle_id} created --------")
            self._streamline_vehicle_network(vehicle_network, mn_route, ir_vehicle_network, dyn_charger_to_replica)

        return mn.VehicleNetwork(vehicle=mn.Vehicle(route.vehicle_id), network=vehicle_network)

    def _create_model(
            self,
            vehicle_networks: dict[mn.Vehicle, mn.VehicleNetwork],
            vehicle_routes: dict[mn.Vehicle, ir.Route],
        ) -> DynamicChargerLRP:
        model_param = DynamicChargerLRPParameters(
            battery_capacity_in_kwh=self.instance_parameters.soc_max,
            min_soc_as_share=self.instance_parameters.soc_min/self.instance_parameters.soc_max,
            max_soc_as_share=self.instance_parameters.soc_max/self.instance_parameters.soc_max,
            initial_soc_as_share=self.instance_parameters.soc_init/self.instance_parameters.soc_max,
            energy_price_per_kwh=self.instance_parameters.energy_prices,
            time_step_in_seconds=self.solver_parameters.time_step_in_seconds,
            consumption_cost_per_kwh=self.instance_parameters.consumption_cost,
        )

        model_vehicle_routes: dict[mn.Vehicle, mdl.Route] = {
            vehicle_id: _parse_route(vehicle_networks[vehicle_id], route)
            for vehicle_id, route in vehicle_routes.items()
        }

        return DynamicChargerLRP(vehicle_networks=vehicle_networks, routes=model_vehicle_routes, parameters=model_param)

    def solve_problem_with_mip(
            self,
            intermediate_representation: IntermediateRepresentation,
            vehicles: dict[ir.VehicleID, ir.Vehicle],
        ) -> Tuple[Optional[Solution], Optional[SolveSolution]]:
        """
        Solve the MIP math_programming_model related to the given intermediate representation, with the given initial conditions. If
        vertex_chargers or arc_chargers is given, attempt a warm start.
        @param intermediate_representation: the pre-processed intermediate representation of the instance to solve
        @param vehicles: a map between the considered vehicles and their ids
        @return: A solution object and the SolveSolution object if found
        """
        if intermediate_representation.depot.is_stop:
            # Add non-depot stop to handle this case
            raise NotImplementedError("Solver currently assumes that the depot is not a stop.")

        # Create vehicle networks
        ir_vehicle_networks = split_into_vehicle_networks(intermediate_representation)
        for veh_id, intermediate_vehicle_network in ir_vehicle_networks.items():
            simplified_ir = reduce_ir_network(
                intermediate_vehicle_network
            )
            simplified_ir = simplify_intermediate_repr(
                simplified_ir, allow_path_deviation=self.instance_parameters.allow_path_deviations
            )
            ir_vehicle_networks[veh_id] = simplified_ir

        # vehicle networks creation
        model_vehicle_networks: dict[mn.Vehicle, mn.VehicleNetwork] = {}
        charger_map: dict[ir.Charger, Union[mn.StaticCharger, mn.DynamicCharger]] = {
            c[2]: _convert_ir_charger_to_mn_charger(c[2], is_static=c[0]==c[1])
            for c in intermediate_representation.list_of_all_potential_chargers
        }
        for vehicle_id, vehicle_ir in ir_vehicle_networks.items():
            model_vehicle_networks[mn.Vehicle(vehicle_id)] = self._create_model_network(
                vehicle_ir, vehicles[vehicle_id], charger_map
            )
            # assert that non-dummy vertices in mdl equal ir chargers
            # every charger has only one / maximum one non-dummy representation
            assert len(vehicle_ir.charger_edges) >= len([
                id(a) for a in model_vehicle_networks[mn.Vehicle(vehicle_id)].arcs
                if ((not a.is_dummy) and a.can_construct_charger)]
            )
            assert len(vehicle_ir.charger_nodes) >= len(
                [id(v) for v in model_vehicle_networks[mn.Vehicle(vehicle_id)].vertices
                 if ((not v.is_dummy) and v.can_construct_charger)]
            )

        # build math_programming_model based on decomposed vehicle networks
        model = self._create_model(model_vehicle_networks,
            {veh_id: ir_vehicle_networks[veh_id].routes[0] for veh_id in ir_vehicle_networks.keys()}
        )

        sol = model.optimize(
            time_limit=self.solver_parameters.run_time_in_seconds,
            results_file=self.solver_parameters.get_full_solver_path,
            warmstart=self.warmstart_solution
        )

        if sol is not None:
            sol.report(time_step=self.solver_parameters.time_step_in_seconds)

            for veh_id, network in model_vehicle_networks.items():
                # network.plot(hide_non_traversed_arcs=True)
                network.validate_solution(time_step=self.solver_parameters.time_step_in_seconds)

        return sol, model._model.solution

    def _streamline_vehicle_network(
            self,
            vehicle_network: nx.DiGraph,
            mn_route: List[mn.Vertex],
            ir: IntermediateRepresentation,
            dyn_charger_to_arc_mapping: Dict[ir.ChargerID, Dict[int, List[Tuple[mn.Vertex, mn.Vertex]]]]
    ):
        """
        Every replica charger node can only be visited once and if deviations are not allowed, they also have to
        be visited if the are connected in the graph that is passed to this function
        """
        # warnings.warn("This function relies on matching between Vertex IDs in IR and Model Network")
        assert not self.instance_parameters.allow_path_deviations
        assert nx.is_weakly_connected(vehicle_network)

        def map_to_ir_id(node: mn.Vertex) -> VertexID:
            if node.is_depot:
                return VertexID("Depot")
            return VertexID(node.dummy_of.id if node.is_dummy else node.id)

        used_node_set = set()
        used_arc_set = set()
        remove_edges = set()
        for p,s in itertools.pairwise(mn_route):

            ir_p = ir.get_vertex(map_to_ir_id(p))
            ir_s = ir.get_vertex(map_to_ir_id(s))

            # if path goes directly: remove all connections that are not to direct successor (s)
            via_spp, path = ir.spp_via_charger(ir_p, ir_s)

            if via_spp:
                # contains all potential connections sorted by replica level in ascending order
                connection_candidates_to_charger = sorted(
                    [
                        e for e in vehicle_network.out_edges(p)
                        if map_to_ir_id(e[1])==path[1] and e[1] not in used_node_set and (e[1].replica_level>0 or not e[1].is_stop)
                    ],
                    key=lambda x: x[1].replica_level, reverse=False
                )
                assert len(connection_candidates_to_charger)>0, f"No candidates left yields infeasible solution"

                connection_candidates_from_charger = sorted(
                    [
                        e for e in vehicle_network.in_edges(s)
                        if map_to_ir_id(e[0])==path[-2] and e[0] not in used_node_set and (e[0].replica_level>0 or not e[0].is_stop)
                    ],
                    key=lambda x: x[0].replica_level, reverse=False
                )
                assert len(connection_candidates_from_charger)>0, f"No candidates left yields infeasible solution"

                def weight_func(a, b, data):
                    """We need to make sure that the path is free (i.e, avoid infeasibility)"""
                    # We take care of not visiting nodes twice in the while loop below
                    return data["arc"].distance

                find_free_path = True
                while find_free_path:
                    first_segment = connection_candidates_to_charger[0][1]
                    last_segment = connection_candidates_from_charger[0][0]
                    mn_shortest_path = nx.shortest_path(
                        vehicle_network,
                        first_segment,
                        last_segment,
                        weight=weight_func
                    )
                    find_free_path = False
                    for n in mn_shortest_path:
                        if n in used_node_set:
                            if first_segment.replica_level < last_segment.replica_level:
                                connection_candidates_to_charger = connection_candidates_to_charger[1:]
                            elif first_segment.replica_level > last_segment.replica_level:
                                connection_candidates_from_charger = connection_candidates_from_charger[1:]
                            else:
                                connection_candidates_to_charger = connection_candidates_to_charger[1:]
                                connection_candidates_from_charger = connection_candidates_from_charger[1:]
                            # leave for loop
                            find_free_path = True
                            break

            # idea is that there should be only one path between each pair
            for _,v in vehicle_network.out_edges(p):

                if not via_spp and v!=s and not v.is_depot:
                    remove_edges.add((p,v))

                if not via_spp and v==s:
                    used_node_set.add(s)
                    used_arc_set.add((p,s))

                if via_spp and v!=first_segment and not v.is_depot:
                    remove_edges.add((p,v))

                if via_spp and v==first_segment:
                    used_node_set.add(v)
                    used_arc_set.add((p, v))

            for u,_ in vehicle_network.in_edges(s):

                if not via_spp and u!=p and not u.is_depot:
                    remove_edges.add((u,s))

                if not via_spp and u==p:
                    used_node_set.add(p)
                    used_arc_set.add((p, s))

                if via_spp and u!=last_segment and not u.is_depot:
                    remove_edges.add((u,s))

                if via_spp and u==last_segment:
                    used_node_set.add(u)
                    used_arc_set.add((u, s))

            # what is the shortest path in mn network between the two connections - we "clean" this path
            if via_spp:
                assert not (p in mn_shortest_path or s in mn_shortest_path)
                for (node_one, node_two) in pairwise(mn_shortest_path + [s]):
                    used_node_set.add(node_one)
                    for edge in vehicle_network.in_edges(node_one):
                        if edge not in used_arc_set:
                            remove_edges.add(edge)
                            continue
                        used_arc_set.add(edge)

                    for edge in vehicle_network.out_edges(node_one):
                        if edge != (node_one, node_two):
                            remove_edges.add(edge)
                            continue
                        used_arc_set.add(edge)

                    # mark all other segments belonging to the same dynamic charger and replica level as used
                    arc = vehicle_network.get_edge_data(node_one, node_two)["arc"]
                    if arc.can_construct_charger:
                        for edge in dyn_charger_to_arc_mapping[arc.charger.irID][arc.replica_level]:
                            used_node_set.add(edge[0])
                            used_node_set.add(edge[1])

        # remove
        remaining_edges = {(a,b) for (a,b) in vehicle_network.edges} - remove_edges
        remaining_arc_dummies = {vehicle_network.get_edge_data(u,v)["arc"].dummy_of for u,v in remaining_edges} - {None}
        remove_edges = {
            e for e in remove_edges
            if vehicle_network.has_edge(*e)
            and e not in used_arc_set
            and vehicle_network.get_edge_data(*e)["arc"] not in remaining_arc_dummies
        }
        vehicle_network.remove_edges_from(remove_edges)

        additional_remove_nodes = []
        weakly_connected_components = nx.weakly_connected_components(vehicle_network)
        for component in weakly_connected_components:
            comp = set(component)
            if set(mn_route).issubset(comp):
                continue
            for n in comp:
                assert n.can_construct_charger or n.type==mn.VertexType.AUXILIARY, f"{n}, {n.is_stop}"
                if n.is_dummy:
                    additional_remove_nodes.append(n)
        vehicle_network.remove_nodes_from(additional_remove_nodes)
        return None

    def _duplicate_static_chargers(
            self, vehicle_network: nx.DiGraph, n_repl: int
    ):
        node: mn.Vertex
        replica_to_node = {}
        for node in vehicle_network:
            # Skip non-station nodes
            if not node.can_construct_charger:
                continue
            # skip dummies (i.e., stops that are dummies and can charge)
            if node.is_dummy:
                continue
            for replica_id in range(1, 1 + n_repl):
                # Is it standalone or at a stop?
                replica = copy.copy(node)
                replica.id += f'_R{replica_id}'
                # replica.name += f'_R{replica_id}'
                replica.type = mn.VertexType.CHARGER
                replica.time_window_begin = 0
                replica.time_window_end = math.inf
                replica.dummy_of = node if node.dummy_of is None else node.dummy_of
                replica.replica_level = node.replica_level + replica_id

                assert replica not in vehicle_network and replica not in replica_to_node

                replica_to_node[replica] = node

        vehicle_network.add_nodes_from(replica_to_node.keys())

        # Add connections to all nodes
        for origin_node, target_node in itertools.product(vehicle_network, repeat=2):
            # If neither node is a replica we can skip as no new arc is introduced
            if not origin_node in replica_to_node and not target_node in replica_to_node:
                continue

            # if both nodes are chargers exclusively, skip
            if origin_node.type == mn.VertexType.CHARGER and target_node.type == mn.VertexType.CHARGER:
                continue

            # Otherwise we want to add an arc if the original nodes are connected
            original_origin_node = replica_to_node.get(origin_node, origin_node)
            original_target_node = replica_to_node.get(target_node, target_node)

            if (edge := vehicle_network.edges.get((original_origin_node, original_target_node))) is None:
                continue

            # Copy the edge, adjust it's origin/target and add it to the network
            new_edge: mn.Arc = copy.copy(edge['arc'])
            assert new_edge is not None
            new_edge.origin = origin_node
            new_edge.target = target_node
            # print(f'Adding edge {origin_node}-{target_node}')
            vehicle_network.add_edge(origin_node, target_node, arc=new_edge)

        # Test correctness
        for replica, original_node in replica_to_node.items():
            for edge in vehicle_network.in_edges(original_node):
                if edge[0].type != mn.VertexType.CHARGER and edge[0].type != mn.VertexType.AUXILIARY:
                    assert vehicle_network.has_edge(edge[0], replica)
            for edge in vehicle_network.out_edges(original_node):
                if edge[1].type != mn.VertexType.CHARGER and edge[1].type != mn.VertexType.AUXILIARY:
                    assert vehicle_network.has_edge(replica, edge[1])

    def _duplicate_dynamic_chargers(
            self,
            ir: IntermediateRepresentation,
            vehicle_network: nx.DiGraph,
            n_repl: int,
    ) -> Dict[ir.ChargerID, Dict[int, List[Tuple[mn.Vertex, mn.Vertex]]]]:
        arc: mn.Arc
        replica_to_arc = {}
        replica_to_node = {}
        charger_to_mn_replica = {c.id: {r_id: [] for r_id in range(0, 1 + n_repl)} for c in ir.dyn_chargers}

        def _get_existing_copy(l: Iterable[mn.Vertex], match: str) -> Optional[mn.Vertex]:
            for v in l:
                if v.id == match:
                    return v
            return None

        for u, v, data in vehicle_network.edges(data=True):
            # Skip non-station nodes
            if not data['arc'].can_construct_charger:
                continue
            # add original to mapping
            charger_to_mn_replica[data["arc"].charger.irID][0].append((u, v))

            # add as many replicas as required
            for replica_id in range(1, 1 + n_repl):
                # identify vertices in original structure
                origin = data['arc'].origin
                destination = data['arc'].target

                # make copies (only if not being done in previous iteration)
                if u.id + f"_R{replica_id}" in replica_to_node.keys():
                    replica_origin = _get_existing_copy(replica_to_node.keys(), u.id + f"_R{replica_id}")
                else:
                    replica_origin = copy.copy(u)
                    replica_origin.id += f'_R{replica_id}'
                    # replica_origin.name += f'_R{replica_id}'
                    replica_origin.type = mn.VertexType.AUXILIARY
                    replica_origin.dummy_of = origin if origin.dummy_of is None else origin.dummy_of
                    replica_origin.replica_level += replica_id
                    replica_to_node[replica_origin] = origin

                if v.id + f"_R{replica_id}" in replica_to_node.keys():
                    replica_destination = _get_existing_copy(replica_to_node.keys(), v.id + f"_R{replica_id}")
                else:
                    replica_destination = copy.copy(v)
                    replica_destination.id += f'_R{replica_id}'
                    # replica_destination.name += f'_R{replica_id}'
                    replica_destination.type = mn.VertexType.AUXILIARY
                    replica_destination.dummy_of = destination if destination.dummy_of is None else destination.dummy_of
                    replica_destination.replica_level += replica_id
                    replica_to_node[replica_destination] = destination

                # arc can always be copied because we iterate over edges, attributes need change
                replica_arc = copy.copy(data['arc'])
                replica_arc.origin = replica_origin
                replica_arc.target = replica_destination
                replica_arc.dummy_of = data['arc'] if data["arc"].dummy_of is None else data["arc"].dummy_of
                replica_arc.replica_level += replica_origin.replica_level
                replica_to_arc[replica_arc] = data['arc']
                charger_to_mn_replica[replica_arc.charger.irID][replica_id].append(
                    (replica_arc.origin, replica_arc.target)
                )

        vehicle_network.add_nodes_from(replica_to_node.keys())
        vehicle_network.add_edges_from(
            [(replica_arc.origin, replica_arc.target, {'arc': replica_arc}) for replica_arc in replica_to_arc.keys()]
        )

        # Add connections to all nodes
        for origin_node, destination_node in itertools.product(vehicle_network, repeat=2):
            # If neither node is a replica we can skip as no new arc is introduced
            if not origin_node in replica_to_node and not destination_node in replica_to_node:
                continue

            # If the replicated arc is a charger arc, we have already added
            if (origin_node, destination_node) in vehicle_network.edges:
                assert vehicle_network.get_edge_data(origin_node, destination_node)['arc'].charger is not None
                continue

            # if the nodes are both replicas and belong to different duplication levels or orginals, continue
            if origin_node in replica_to_node and destination_node in replica_to_node and origin_node.replica_level != destination_node.replica_level:
                continue

            # Otherwise we want to add an arc if the original nodes are connected
            original_origin_node = replica_to_node.get(origin_node, origin_node)
            original_target_node = replica_to_node.get(destination_node, destination_node)
            if (edge := vehicle_network.edges.get((original_origin_node, original_target_node))) is None:
                continue

            # there is no reason to switch between different replica levels
            if original_origin_node.type == mn.VertexType.AUXILIARY and original_target_node.type == mn.VertexType.AUXILIARY and (
                    original_origin_node in replica_to_node.values() or original_target_node in replica_to_node.values()
            ):
                continue

            # Copy the edge, adjust it's origin/target and add it to the network
            new_arc: mn.Arc = copy.copy(edge['arc'])
            assert new_arc is not None
            new_arc.origin = origin_node
            new_arc.target = destination_node
            vehicle_network.add_edge(origin_node, destination_node, arc=new_arc)

        # Test correctness
        for replica, original_arc in replica_to_arc.items():
            assert vehicle_network.has_edge(replica.origin, replica.target)
            assert vehicle_network.has_edge(original_arc.origin, original_arc.target)

        return charger_to_mn_replica


def _parse_route(vehicle_network: mn.VehicleNetwork, route: ir.Route) -> list[mn.Vertex]:
    stop_counter = {s.vertex_id: 0 for s in route.stop_sequence}
    parsed_route = [vehicle_network.start_depot]
    for stop in route.stop_sequence[1:-1]:
        # Stops will always have the same ID
        if stop_counter[stop.vertex_id] > 0:
            next_mn_id = f"{stop.vertex_id}_{route.vehicle_id}_{stop_counter[stop.vertex_id]}"
        else:
            next_mn_id = mn.VertexID(stop.vertex_id)
        parsed_route.append(vehicle_network.get_vertex(next_mn_id))
        stop_counter[stop.vertex_id] += 1
    parsed_route.append(vehicle_network.end_depot)

    return parsed_route
