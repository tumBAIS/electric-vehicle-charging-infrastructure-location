import math
import sys
import shortuuid

import pytz
import enum
from itertools import pairwise
from collections import defaultdict
import logging
from dataclasses import dataclass, field, astuple
from rich import print as rprint
from rich.table import Table
from io import StringIO
from typing import Dict, Union, List, Tuple, Optional, Set
import pandas as pd

from framework.utils import ValidationError
import framework.intermediate_representation as ir
from framework.utils import filter_tuple_list
from framework.solution_representation import SolutionRepresentation, Point, Itinerary, SolCharger
from iterative_local_search.decision_variables import DecisionVariables

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def nested_dict_factory():
    return defaultdict(list)


class SPPRCVertexType(enum.IntFlag):
    DEPOT = enum.auto()
    STOP = enum.auto()
    CHARGER = enum.auto()
    SEGMENT_START = enum.auto()
    SEGMENT_END = enum.auto()
    FIRST_CHARGER_NODE = enum.auto()


@dataclass(slots=True)
class Vertex:
    id: str
    irID: ir.VertexID
    departure_time_window: tuple[float, float]
    type: SPPRCVertexType
    arrival_time_window: tuple[float, float] = (0, math.inf)
    predecessor_stop: Optional[ir.VertexID] = None
    successor_stop: Optional[ir.VertexID] = None

    def __eq__(self, other):
        return self.id == other.id

    @property
    def is_stop(self):
        return bool(self.type & SPPRCVertexType.STOP)

    @property
    def is_depot(self):
        return bool(self.type & SPPRCVertexType.DEPOT)

    @property
    def is_stop_or_depot(self):
        return self.is_stop or self.is_depot

    @property
    def is_charger(self):
        return bool(self.type & SPPRCVertexType.CHARGER)

    @property
    def is_segment_start(self):
        return bool(self.type & SPPRCVertexType.SEGMENT_START)

    @property
    def is_segment_end(self):
        return bool(self.type & SPPRCVertexType.SEGMENT_END)

    @property
    def is_first_charger_node(self):
        return bool(self.type & SPPRCVertexType.FIRST_CHARGER_NODE)


@dataclass(slots=True)
class Arc:
    start: str
    end: str
    time: Union[float, int]
    consumed_energy: float
    recharged_energy: float
    time_to_dyn_charger: int = 0
    consumption_sequence: List[float] = field(default_factory=lambda: [0])
    encoded_dyn_charger: Optional[ir.ChargerID] = None

    def __post_init__(self):
        self.time = int(self.time)  # comes in seconds
        self.consumed_energy = round(self.consumed_energy, 3)  # comes in kwh
        self.recharged_energy = round(self.recharged_energy, 3)  # comes in kwh

    def __eq__(self, other):
        return self.recharged_energy == other.recharged_energy

    def __hash__(self):
        return hash(astuple(self))


@dataclass(slots=True)
class Route:
    stops: list[Vertex]

    @property
    def depot(self) -> Vertex:
        return self.stops[0]

    @property
    def latest_arrival_time(self) -> float:
        return self.stops[-1].departure_time_window[1]

    def __len__(self):
        return len(self.stops)

    def __getitem__(self, item: int):
        return self.stops[item]


@dataclass(slots=True)
class Network:
    """
    Expanded network to apply labeling algorithm. It contains all the energy and time constraints for the given route.
    """
    vertices: dict[str, Vertex]
    arcs: dict[tuple[str, str], dict[int, Arc]]
    route: Route
    energy_prices: Union[float, Dict[int, float]]
    consumption_cost: float
    max_soc_in_kwh: float
    min_soc_in_kwh: float
    outgoing_arcs: dict[str, list[Tuple[str, int, Arc]]] = field(default_factory=nested_dict_factory)
    incoming_arcs: dict[str, list[Tuple[str, int, Arc]]] = field(default_factory=nested_dict_factory)
    _mapping: Dict[ir.VertexID, int] = field(default_factory=dict)

    def __post_init__(self):
        # we could implement a network validator here; but keep in mind that this constructor is called a lot and
        # calling the validator may impact computation time
        pass

    def __add__(self, other: "Network") -> "Network":
        raise ValidationError(f"Sum not defined on class 'Network'")

    def get_plain_representation(self) -> Tuple[List, List[Tuple], List[Tuple], List[float], Dict[str, int], Dict[str, int]]:
        id2index = {vertex.id: i for i, vertex in enumerate(list(self.vertices.values()))}
        index2id = {value: key for key, value in id2index.items()}

        # dyn stations (None is mapped to any integer from the enumeration
        charger2index = {
            c: i for i,c in enumerate({v.encoded_dyn_charger for d in self.arcs.values() for v in d.values()})
        }
        index2charger = {value: key for key, value in charger2index.items()} | {-1:None}

        route = [id2index[vertex.id] for vertex in self.route.stops]
        vertices = [
            (id2index[vertex.id], vertex.departure_time_window[0], vertex.departure_time_window[1], vertex.arrival_time_window[0], vertex.arrival_time_window[1])
            for vertex in list(self.vertices.values())]
        # # successors is list and sorting is done on python side such that arcs are sorted ascending in first key and
        arcs = []
        for idx in range(len(index2id)):
            for outgoing_arc in self.outgoing_arcs[index2id[idx]]:
                arc = outgoing_arc[2]
                arcs.append(
                    (idx, id2index[outgoing_arc[0]], arc.time, arc.consumed_energy, arc.recharged_energy,
                     charger2index[arc.encoded_dyn_charger], arc.consumption_sequence, arc.time_to_dyn_charger)
                )
        # determine energy prices
        if type(self.energy_prices) == float:
            energy_prices = [self.energy_prices] * 24
        else:
            energy_prices = [self.energy_prices[i] for i in range(24)]
        return route, vertices, arcs, energy_prices, id2index, index2charger

    def validate(self):
        for v_id, v in self.vertices.items():
            # a stop can never be the start of a dynamic segment
            assert not (v.is_segment_start or v.is_segment_end) if (v.is_stop or v.is_depot or v.is_charger) else True, f"{v}"
            # the set of stops and (stationary) charger nodes are disjunct
            assert not v.is_charger if v.is_stop else True, f"{v}"
            # first charger node can only be if also charger
            assert v.is_charger if v.is_first_charger_node else True, f"{v}"

        for (key1, key2), arc_dict in self.arcs.items():
            for key3, arc in arc_dict.items():
                # pointers are correctly set
                assert key2 in self.successors(key1)
                # arcs that charge stationary must have exactly to successors
                if arc.time == 0 and arc.recharged_energy > 0:
                    assert self.vertices[key2].is_charger
                # arcs that charge dynamically must also consume, and origin must be labelled as segment start
                if arc.recharged_energy > 0 and arc.time == 0:
                    assert arc.consumed_energy > 0, f"{arc}"
                    assert self.vertices[key1].is_segment_start, f"{arc}"

    def successors(self, vertex_id: str) -> Set[str]:
        return {out_arc[0] for out_arc in self.outgoing_arcs[vertex_id]}

    def get_vertex(self, id: str) -> Vertex:
        return self.vertices[id]

    def get_stop(self, item: int) -> Vertex:
        return self.route[item]

    def get_arc(self, start_id: str, end_id: str, key: int) -> Arc:
        return self.arcs[(start_id, end_id)][key]

    def add_or_update_vertex(self, vertex: Vertex) -> None:
        """
        If the vertex exists in the network, update it.
        Else, merely add it to the vertices
        @param vertex: vertex to add
        """
        if vertex.id in self.vertices.keys():
            # in this case only update successors and type info
            # successors = self.vertices[vertex.id].successors
            vertex.type |= self.vertices[vertex.id].type
            self.vertices[vertex.id] = vertex
        else:
            # in this case update counter and add stop / depot to route
            if vertex.is_stop or vertex.is_depot:
                self.update_stop_counter(vertex.irID)
                self.route.stops.append(vertex)
            self.vertices[vertex.id] = vertex

    def update_stop_counter(self, id_):
        try:
            self._mapping[id_] += 1
        except KeyError:
            self._mapping[id_] = 1

    def create_vertex_name(self, ir_id: ir.VertexID, route_id: str):
        if ir_id == "Depot" and ir_id not in self._mapping:
            return "Depot"
        elif ir_id == "Depot":
            return "Depot-end"
        elif ir_id not in self._mapping:
            return str(ir_id)
        return f"{ir_id}_{route_id}_{self._mapping[ir_id]}"

    def add_stop(self, vertex: Vertex) -> None:
        self.route.stops.append(vertex)

    def add_or_update_arc(self, arc: Arc, key: int) -> None:
        """
        Add arc to network.
        @param arc: arc to add
        @param key: key of the arc in a multigraph
        """
        if (arc.start, arc.end) not in self.arcs.keys():
            self.arcs[(arc.start, arc.end)] = {}
        if arc in self.arcs[(arc.start, arc.end)].values():
            return None
        self.arcs[(arc.start, arc.end)][key] = arc
        self.outgoing_arcs[arc.start].append((arc.end, key, arc))
        self.incoming_arcs[arc.end].append((arc.start, key, arc))

    def construct_sr_paths(self, path: list[tuple[str, float, float, float, bool]], soc_init: float) -> tuple[
        list[Point], dict[Point, tuple[float, float]]]:
        """
        Given a path of vertices, recreate the path in terms of Points from the SolutionRepresentation file.
        The result is still an expanded path (vertex chargers for examples are still duplicated).
        @param path: list of (VertexID, arrival_time, departure_time, SOC)
        @param soc_init: initial soc (kwh)
        @return: the list of Points, and a dictionary linking each Point with its arrival and departure SoCs
        """

        def interpolate(a,b):
            return round((a+b)/2,3)

        solution_path = []
        soc_path = {}
        current_point = Point(
            id=ir.VertexID('Depot'),
            arrival_time=int(path[0][1]),
            departure_time=int(path[0][2]),
            soc=soc_init,
            is_depot=True,
            is_stop=False,
            accumulated_consumed_energy=0,
            accumulated_charged_energy=0,
            is_static_charger=False,
        )
        solution_path += [current_point]
        soc_path[current_point] = (soc_init, soc_init)
        for (
                    current_vertex, current_arrival_time, current_departure_time, current_soc, current_consumed_energy,
                    current_recharged_energy, current_static_standalone_charger, _
            ), \
            (
                    next_vertex, next_arrival_time, next_departure_time, next_soc, consumed_energy, recharged_energy,
                    next_static_standalone_charger, next_crossed_segment
            ) in pairwise(path):
            # next crossed segment is None if no segment was crossed between current and next
            if not next_crossed_segment is None:
                synthetic_time = interpolate(current_departure_time, next_arrival_time)
                synthetic_soc = interpolate(current_soc, next_soc)
                synthetic_point = Point(
                    id=next_crossed_segment,
                    arrival_time=int(synthetic_time),
                    departure_time=int(synthetic_time),
                    soc=round(synthetic_soc, 3),
                    is_depot=False,
                    is_stop=False,
                    accumulated_consumed_energy=interpolate(abs(current_consumed_energy),abs(consumed_energy)),
                    accumulated_charged_energy=interpolate(current_recharged_energy,recharged_energy),
                    is_static_charger=False,
                    is_synthetic_dyn_charger_representation=True
                )
                soc_path[synthetic_point] = (synthetic_soc, synthetic_soc)
                solution_path += [synthetic_point]
            next_vertex_with_data = self.get_vertex(next_vertex)
            next_vertex_id = next_vertex_with_data.irID
            next_point = Point(
                id=next_vertex_id,
                arrival_time=int(next_arrival_time),
                departure_time=int(next_departure_time),
                soc=round(next_soc, 3),
                is_depot=not next_static_standalone_charger and next_vertex_with_data.is_depot,
                is_stop=not next_static_standalone_charger and next_vertex_with_data.is_stop,
                accumulated_consumed_energy=abs(round(consumed_energy, 3)),
                accumulated_charged_energy=abs(round(recharged_energy, 3)),
                is_static_charger=next_vertex_with_data.is_charger,
            )

            soc_path[next_point] = (next_soc, next_soc)
            solution_path += [next_point]
        return solution_path, soc_path

    def to_solution_representation(
        self, path: list[tuple[str, float, float, float, bool]], soc_init: float, vehicle_id: ir.VehicleID,
        static_invest: dict[ir.VertexID, SolCharger], dynamic_invest: dict[tuple[ir.VertexID, ir.VertexID], SolCharger],
        total_cost: float, verbose=False, total_consumption: float = None, total_recharge: float = None,
    ) -> SolutionRepresentation:
        """
        Given a path of vertices, compute all elements needed to construct the SolutionRepresentation of the path
        @param path: list of (VertexID, arrival_time, departure_time, soc) forming a feasible route
        @param total_cost: total cost on propagated optimal path / route in monetary unit
        @param total_consumption: total consumption on propagated optimal path / route in kwh
        @param total_recharge: total recharge on propagated optimal path / route in kwh
        @param soc_init: initial SOC (kwh)
        @param vehicle_id: vehicle id of the vehicle that we are building the solution for
        @param static_invest: stationary charging station configuration
        @param dynamic_invest: dynamic charging station configuration
        @param verbose: (False by default)
        @return: SolutionRepresentation
        """
        # Extract expanded path in terms of Points
        solution_path, soc_path = self.construct_sr_paths(path, soc_init=soc_init)
        # Filter the expanded path to eliminate copies
        filtered_solution_path, filtered_soc_path = filter_path(solution_path, soc_path)

        # Print the solution data table
        if verbose:
            report(filtered_solution_path, filtered_soc_path, vehicle_id)

        # Construct the SolutionRepresentation of the solution
        solution_itinerary = Itinerary(vehicle_id, filtered_solution_path)

        if total_consumption is None and total_recharge is None:
            total_consumption = filtered_solution_path[-1].accumulated_consumed_energy
            total_recharge = filtered_solution_path[-1].accumulated_charged_energy

        return SolutionRepresentation(
            [solution_itinerary],
            dynamic_invest,
            static_invest,
            routing_cost=total_cost,
            consumed_energy=abs(total_consumption),
            recharged_energy=total_recharge,
        )


def filter_path(solution_path: list[Point], soc_path: dict[Point, tuple[float, float]]) -> tuple[
    list[Point], list[tuple[float, float]]]:
    """
    Given an expanded path (with copies of vertex chargers) and the corresponding arrival and departure socs at each
    Point, construct the final filtered Point path, with differentiation between stop vertices and charger vertices
    For example, if we recharge at a stop, two vertices are stored, one for the stop and one for the charging, but the
    latter is not time expanded anymore.
    @param solution_path: list[Point]
    @param soc_path: dict[Point, tuple[float, float]]
    @return: the filtered solution path and the corresponding arrival and departure socs
    """
    filtered = [solution_path[0]]
    filtered_soc_path = [soc_path[solution_path[0]]]
    for u in solution_path[1:]:
        if u.id == filtered[-1].id: # and u.is_static_charger and filtered[-1].is_static_charger:
            filtered_soc_path[-1] = (filtered_soc_path[-1][0], soc_path[u][1])
            filtered[-1] = u + filtered[-1]
        else:
            filtered += [u]
            filtered_soc_path += [soc_path[u]]
    return filtered, filtered_soc_path


def report(solution_itinerary: list[Point], soc_path: list[tuple[float, float]], vehicle_id: str) -> None:
    """
    Given a solution itinerary and the corresponding arrival and departure socs at each point, print the data table
    describing each vertex of the path
    @param solution_itinerary
    @param soc_path
    @param vehicle_id
    @return:
    """
    table = Table(title=f"Route of vehicle {vehicle_id}")
    table.add_column("Vertex")
    table.add_column("Arrival time")
    table.add_column("Departure time")
    table.add_column("Arrival SoC")
    table.add_column("Delta time")
    table.add_column("Delta SoC")

    args_list = []

    current_point = solution_itinerary[0]
    current_arrival_time = 0
    current_departure_time = current_point.departure_time.replace(tzinfo=pytz.utc).timestamp()
    current_delta_time = 0
    current_arrival_soc, current_departure_soc = soc_path[0]
    current_delta_soc = current_departure_soc - current_arrival_soc
    args_list += [[current_point.id.split('_')[0], current_arrival_time, current_departure_time,
                   round(current_arrival_soc, 5), current_delta_time, current_delta_soc]]

    for i in range(1, len(solution_itinerary)):
        # In case we recharge at the current (or next) stop : recall that a vertex is stored twice if it corresponds to
        # a stop where we recharge, however we only want to see it once in the table
        if current_point.id == solution_itinerary[i].id:
            current_point = solution_itinerary[i]
            current_arrival_soc, current_departure_soc = soc_path[i]
            current_departure_time = solution_itinerary[i].departure_time.replace(tzinfo=pytz.utc).timestamp()
            if not current_point.is_stop:
                current_delta_time += current_departure_time - current_point.arrival_time.replace(
                    tzinfo=pytz.utc).timestamp()
            current_delta_soc = round(current_delta_soc + soc_path[i][1] - soc_path[i][0], 5)
            args_list[-1] = [current_point.id, current_arrival_time, current_departure_time,
                             round(current_arrival_soc, 5),
                             current_delta_time, current_delta_soc]

        else:
            current_point = solution_itinerary[i]
            current_arrival_time = current_point.arrival_time.replace(tzinfo=pytz.utc).timestamp()
            current_departure_time = current_point.departure_time.replace(tzinfo=pytz.utc).timestamp()
            current_delta_time = 0 if current_point.is_stop else current_departure_time - current_arrival_time
            current_arrival_soc, current_departure_soc = soc_path[i]
            current_delta_soc = round(current_departure_soc - current_arrival_soc, 5)
            args = [current_point.id.split('_')[0], current_arrival_time, current_departure_time,
                    round(current_arrival_soc, 5), current_delta_time, current_delta_soc]
            args_list += [args]

    for args in args_list:
        table.add_row(*map(str, args))
    buf = StringIO("")
    rprint(table, file=buf)
    print(buf.getvalue(), file=sys.stdout)

    pd.DataFrame(args_list, columns=['Vertex', 'Arrival time', 'Departure time', 'Arrival SoC', 'Delta time',
                                     'Delta SoC']).to_csv('sol.csv', index=False)


def _create_network(
        decision: DecisionVariables, route: ir.Route, time_step: int
) -> Network:
    """Sequential network building - this function calls the subroutines, order of calls matters"""
    logging.debug("Building SPPRC network started")
    charging_needed = decision.energy_to_complete_route(0, route) + decision.min_soc > decision.soc_init
    route_index = decision.intermediate_rep.routes.index(route)

    # create network (empty)
    spprc_network = Network(
        vertices={}, arcs={}, route=Route(list()), energy_prices=decision.energy_prices,
        consumption_cost=decision.consumption_cost, max_soc_in_kwh=decision.max_soc, min_soc_in_kwh=decision.min_soc
    )

    # go step-wise from stop (u) to stop (v)
    for i in range(0, len(route.stop_sequence)-1):
        _stop_to_stop(
            spprc_network, current_stop=route[i], next_stop=route[i+1], decision=decision, route_index=route_index
        )

    if charging_needed:
        # if route can be operated w/o charging, prune network and keep network trivial
        for i in range(0, len(route.stop_sequence) - 1):
            _stop_to_vertex_charger_to_stop(
                spprc_network, current_vertex=spprc_network.route.stops[i], next_vertex=spprc_network.route.stops[i+1],
                decision=decision, upper_limit=decision.energy_to_complete_route(i, route), time_step=time_step,
                route_index=route_index,
            )
            _stop_to_arc_charger_to_stop(
                spprc_network, current_vertex=spprc_network.route.stops[i], next_vertex=spprc_network.route.stops[i+1],
                decision=decision
            )
    logging.debug("Building SPPRC network finished")
    return spprc_network


def _stop_to_stop(
        spprc_network: Network, current_stop: ir.Stop, next_stop: ir.Stop, decision: DecisionVariables, route_index: int
) -> Network:
    """
    Create a direct connection between two stops u,v (with no chargers)
    @param spprc_network: instance of Network (changed in place)
    @param current_stop: stop u (as IR object)
    @param next_stop: stop v (as IR object)
    @param decision: DecisionVariables object
    @param route_index: Index of respective route in IR attribute 'routes'
    @return: Pointer to spprc_network (changed in place)
    """
    # condition on IR (the direct connection should always be there, but..)
    if not decision.intermediate_rep.has_arc(current_stop.vertex_id, next_stop.vertex_id):
        return spprc_network

    # check if we are handling the connection depot - first stop or last stop - depot
    cs = decision.intermediate_rep.get_vertex(current_stop.vertex_id)
    ns = decision.intermediate_rep.get_vertex(next_stop.vertex_id)
    start_vertex = Vertex(
        id=spprc_network.create_vertex_name(ir_id=current_stop.vertex_id, route_id=str(route_index)),
        irID=current_stop.vertex_id,
        departure_time_window=(current_stop.earliest_time_of_service,current_stop.latest_time_of_service),
        type=SPPRCVertexType.STOP if cs.is_stop else SPPRCVertexType.DEPOT,
    )
    if cs.can_construct_charger:
        start_vertex.type |= SPPRCVertexType.CHARGER

    # add current stop node
    spprc_network.add_or_update_vertex(start_vertex)

    if not ns.is_depot:
        end_vertex_id = spprc_network.create_vertex_name(ir_id=next_stop.vertex_id, route_id=str(route_index))
    else:
        end_vertex = Vertex(
            id=spprc_network.create_vertex_name(ir_id=next_stop.vertex_id, route_id=str(route_index)),
            irID=next_stop.vertex_id,
            departure_time_window=(next_stop.earliest_time_of_service, next_stop.latest_time_of_service),
            type=SPPRCVertexType.DEPOT,
        )
        if ns.can_construct_charger:
            end_vertex.type |= SPPRCVertexType.CHARGER
        spprc_network.add_or_update_vertex(end_vertex)
        end_vertex_id = end_vertex.id

    # add edge between the stops
    sts_secs, sts_kwh = decision.get_arc_properties(current_stop.vertex_id, next_stop.vertex_id)
    spprc_network.add_or_update_arc(Arc(start_vertex.id, end_vertex_id, sts_secs, sts_kwh, 0.0), 0)

    return spprc_network


def _stop_to_arc_charger_to_stop(spprc_network: Network, current_vertex: Vertex, next_vertex: Vertex,
                                 decision: DecisionVariables) -> Network:
    """
    Create an elementary expanded network between the current stop and the net stop with regard to the arc chargers
    of the DecisionVariables
    @param spprc_network: Current network (changed in place)
    @param current_vertex: Current stop (SPPRC object)
    @param next_vertex: Next stop (SPPRC object)
    @param decision: DecisionVariables object
    @param route_index: Index of the respective route in ir.routes
    @return: network: Pointer to spprc_network (changed in place)
    """
    # we need a mapping between ir segement vertices and their SPPRC counterpart
    counter = 1

    def min_ignore_none(*args):
        return min((x for x in args if x is not None))

    # Iterate over all dyn chargers
    for charger, edges in decision.intermediate_rep.charger_edges_by_charger.items():

        # if stop is not connected to dyn charger segment no connection needed
        if not decision.intermediate_rep.stop_charger_connections[current_vertex.irID][charger.id]:
            continue

        # If no segment belonging to charger opened --> continue
        cont = [1 if edge in decision.arc_chargers else 0 for edge in edges]
        if sum(cont) == 0:
            continue

        direct_consumption = decision.get_arc_properties(current_vertex.irID, next_vertex.irID)[1]
        charger_start=edges[0][0]
        charger_end=edges[-1][1]

        # it does not make sense to go backwards on the charger
        if charger_start == charger_end:
            continue #break

        # we can leave this loop if there is no connection to the respective charger start
        time_to_start, soc_to_start = decision.get_arc_properties(current_vertex.irID, charger_start)
        if time_to_start is None:
            continue #break

        # we can continue with next end if we cannot leave this end point
        time_from_end, soc_from_end = decision.get_arc_properties(charger_end, next_vertex.irID)
        if time_from_end is None:
            continue #break

        # if we are still in this loop we need to add a contracted arc
        time_charger, soc_charger, recharged_energy = time_to_start+time_from_end, soc_to_start+soc_from_end, 0
        filtered_edges = filter_tuple_list(edges, charger_start, charger_end)
        consumption_pattern = [soc_to_start]
        for u,v in filtered_edges:
            is_active = 1 if (u,v) in decision.arc_chargers else 0
            time_on_segment = decision.get_arc_properties(u, v)[0]
            consumption_on_segment = decision.get_arc_properties(u, v)[1]
            time_charger += time_on_segment
            soc_charger += consumption_on_segment
            recharged_energy += time_on_segment * is_active * charger.charging_rate / 3600
            consumption_pattern.append(soc_charger-recharged_energy)

        consumption_pattern.append(soc_from_end)

        # no need to add arcs whose recharged energy is zero
        if recharged_energy<=soc_charger-direct_consumption:
            continue

        # Note: this is the only place where we add parallel edges (we start at counter = 1 and increase)
        spprc_network.add_or_update_arc(
            Arc(current_vertex.id, next_vertex.id,time_charger, soc_charger, recharged_energy, int(time_to_start),
                consumption_pattern, charger.id), counter
        )
        counter += 1

    return spprc_network


def _stop_to_vertex_charger_to_stop(
        spprc_network: Network, current_vertex: Vertex, next_vertex: Vertex, decision: DecisionVariables,
        upper_limit: float, time_step: int, route_index: int
) -> Network:
    """
    This function adds connections between two stops u,v via stationary chargers at vertices
    @param spprc_network: object of class network which is changed in place
    @param current_vertex: stop u (SPPRC object)
    @param next_vertex: stop v (SPPRC object)
    @param decision: DecisionVariables object
    @param upper_limit: upper limit of recharging
    @param time_step: time step of the time expanded network
    @param route_index: index of route in ir.routes
    @return: Pointer to spprc_network (mutated)
    """

    # add the corresponding nodes and edges for each charger vertex
    for charger_vertex_id, charger in decision.vertex_chargers.items():

        # if the charger is at a stop, time windows should reflect time windows from the stop and charging
        # after the stop is not possible because departure time must be respected
        if charger_vertex_id==current_vertex.irID:
            continue

        ctv_secs, ctv_kwh = decision.get_arc_properties(current_vertex.irID, charger_vertex_id)
        vtn_secs, vtn_kwh = decision.get_arc_properties(charger_vertex_id, next_vertex.irID)

        # condition connection on intermediate representation connection
        if ctv_secs is None or vtn_secs is None:
            continue

        # condition connection to charger on recharge requirement and battery space
        delta_soc = _max_rechargeable_soc(
            current_vertex, next_vertex, decision, charger, upper_limit, time_step, ctv_secs, ctv_kwh, vtn_secs, vtn_kwh)
        
        if delta_soc == 0:
            continue

        time_window = (current_vertex.departure_time_window[0] + ctv_secs, next_vertex.departure_time_window[1]-vtn_secs)
        charger_id_prefix = shortuuid.ShortUUID().random(length=10)
        charger_vertex = Vertex(f"{charger_vertex_id}-" + str(charger_id_prefix) + "-0", charger_vertex_id, time_window,
                                SPPRCVertexType.CHARGER | SPPRCVertexType.FIRST_CHARGER_NODE,
                                predecessor_stop=current_vertex.irID, successor_stop=next_vertex.irID)

        # add charger node copy and connect it
        spprc_network.add_or_update_vertex(charger_vertex)
        spprc_network.add_or_update_arc(Arc(current_vertex.id, charger_vertex.id, ctv_secs, ctv_kwh, 0.0), 0)
        #spprc_network.add_or_update_arc(Arc(charger_vertex.id, next_vertex.id, vtn_secs, vtn_kwh, 0.0), 0)

        # vertex charger time expansion
        for index in range(delta_soc):
            _add_charger_copy(spprc_network, charger_id_prefix, charger_vertex_id, charger, index, current_vertex,
                              next_vertex, ctv_secs, vtn_secs, vtn_kwh, time_step)

    return spprc_network


def _add_charger_copy(spprc_network: Network, charger_id_prefix: str, charger_vertex_id: ir.VertexID,
                      charger: ir.Charger, current_index: int, current_vertex: Vertex, next_vertex: Vertex,
                      ctv_secs: float, vtn_secs: float, vtn_kwh: float, time_step: int) -> None:
    """
    Add a charger copy between the current and the next stop. Recall that each copy should have a unique id!
    @param spprc_network: the expanded network where to add the vertex copies (in place)
    @param charger_id_prefix: Prefix for naming of this charger series
    @param charger_vertex_id: the id in the intermediate representation of the charger vertex
    @param charger: the charger constructed on the charger vertex
    @param current_index: the index of the last create copy of the charger vertex
    @param current_vertex: current stop (SPPRC object)
    @param next_vertex: next Stop (SPPRC object)
    @param ctv_secs: time in seconds to go from current stop to charger
    @param vtn_secs: time in seconds to go from charger to next stop
    @param vtn_kwh: time in seconds to go from charger to next stop
    @param time_step: time step of the time expanded network
    @return:
    """
    # New version
    id_str = f"{charger_vertex_id}-" + charger_id_prefix + "-1"
    if current_index==0:
        charger_vertex_j = Vertex(
            id=id_str,
            irID=charger_vertex_id,
            departure_time_window=(current_vertex.departure_time_window[0] + ctv_secs,next_vertex.departure_time_window[1]-vtn_secs),
            type=SPPRCVertexType.CHARGER,
            predecessor_stop=current_vertex.irID,
            successor_stop=next_vertex.irID,
        )
        spprc_network.add_or_update_vertex(charger_vertex_j)

    # compute energy "consumption" when charging for one unit (one second) of time and add edge
    time = (current_index+1)*time_step
    recharged_energy = time * charger.charging_rate / 3600
    arc = Arc(
        start=f"{charger_vertex_id}-" + charger_id_prefix + "-0", # + str(current_index),
        end=id_str,
        time=time,
        consumed_energy=0.0,
        recharged_energy=recharged_energy
    )
    spprc_network.add_or_update_arc(arc, current_index)

    # add edge between the charger copy and the next stop
    if current_index==0:
        connection_arc = Arc(
            start=id_str,
            end=next_vertex.id,
            time=vtn_secs,
            consumed_energy=vtn_kwh,
            recharged_energy=0.0
        )
        spprc_network.add_or_update_arc(connection_arc, 0)


def decision_spprc_network(
        decision: DecisionVariables, route: ir.Route, time_step: int
) -> Network:
    """
    This function's only purpose is to divide the further logic into two path: 1) build network that allows deviations
    from the shortest path, or 2) build network that does not
    @param decision: DecisionVariables - contains the intermediate representation, the vertex chargers, the arc chargers,
    the maximum and minimum socs and the initial soc of the vehicle
    @param route: the vehicle route
    @param time_step: the time step in seconds of our time expanded network
    @return: the expanded network as instance of 'Network'
    """
    return _create_network(decision, route, time_step)


def _max_rechargeable_soc(current_stop: Vertex, next_stop: Vertex, decision: DecisionVariables,
                          charger: ir.Charger, upper_limit: float, time_step: int,
                          ctv_secs: float, ctv_kwh: float, vtn_secs: float, vtn_kwh: float) -> int:
    """
    Compute the maximum recharging time steps between two consecutive stops at a given charger. This maximum is with regard
    to time windows at the next stop, and the needed travel energy (if it is not worth the travel, returns 0)
    The max rechargeable soc is exactly the number of vertex copies to create for the given charger
    @param current_stop: source stop (IR object)
    @param next_stop: target stop (IR object)
    @param decision: DecisionVariables
    @param charger: Charger object
    @param upper_limit: Upper bound on return value (in kwh)
    @param time_step: time step of the time expanded network
    @param ctv_secs: time in seconds to go from current stop to charger
    @param ctv_kwh: kwh to go from current stop to charger
    @param vtn_secs: time in seconds to go from charger to next stop
    @param vtn_kwh: time in seconds to go from charger to next stop
    @return: number of units of time that can be recharged between the considered stops at the considered charger
    """
    # time, energy and cost between vertices
    ctn_kwh = decision.get_arc_properties(current_stop.irID, next_stop.irID)[1]

    # max rechargeable soc gives the maximum number of charger vertices copies to create (it should be positive!)
    delta_soc = next_stop.departure_time_window[1] - current_stop.departure_time_window[0] - ctv_secs - vtn_secs
    if delta_soc <= 0:
        return 0

    # correct delta_soc to maximum rechargable amount (case depot)
    if current_stop.irID == 'Depot':
        if (decision.max_soc - decision.soc_init + abs(ctv_kwh)) * 3600 / charger.charging_rate < delta_soc:
            delta_soc = math.floor((decision.max_soc - decision.soc_init + abs(ctv_kwh)) * 3600 / charger.charging_rate)

    # correct delta_soc to maximum rechargable amount (case non-depot)
    if (decision.max_soc - decision.min_soc + abs(ctv_kwh)) * 3600 / charger.charging_rate < delta_soc:
        delta_soc = math.floor((decision.max_soc - decision.min_soc + abs(ctv_kwh)) * 3600 / charger.charging_rate)

    # check if it is worth the detour
    if abs(ctv_kwh) + abs(vtn_kwh) - abs(ctn_kwh) > delta_soc * charger.charging_rate / 3600:
        return 0

    # if we don't need that much energy to complete the tour
    if delta_soc > (upper_limit + abs(vtn_kwh) - abs(ctn_kwh)) * 3600 / charger.charging_rate:
        delta_soc = math.ceil((upper_limit + abs(vtn_kwh) - abs(ctn_kwh)) * 3600 / charger.charging_rate)

    # in case the next stop is the depot
    if next_stop.irID == 'Depot':
        # in this case it can never be a valid option to go to the charger
        if abs(ctv_kwh) > abs(ctn_kwh):
            return 0

    return math.ceil(delta_soc/time_step)


def _energy_price(
        arrival_time: float, charging: bool, energy_prices: Union[float, Dict[int, float]], consumption_cost: float
) -> float:
    """
    Return the cost of energy at the arrival time, taking into account whether it's recharging or consuming
    @param arrival_time: float representing time of the day
    @param charging: True, indictates charging takes place
    @param energy_prices: price data
    @param consumption_cost: price for consuming (charging and consuming can happen at different times)
    @return: energy price
    """
    # cost function depending on the arrival time at the charger station
    if charging:
        if isinstance(energy_prices, float):
            return energy_prices
        arrival_time_in_hours = math.ceil(arrival_time / 3600) % 24
        return energy_prices[arrival_time_in_hours]
    return consumption_cost


def _min_recharge_price(
        start_time: float, energy_prices: Union[float, Dict[int, float]]
) -> float:
    """
    Return the minimum future recharging energy price
    @param start_time: current time (i.e., only consider times after this one)
    @param energy_prices: prices
    @return: Minimum future recharging price after start_time as float
    """
    if isinstance(energy_prices, float):
        return energy_prices
    start_time_in_hours = math.ceil(start_time / 3600) % 24
    return min(v for k, v in energy_prices.items() if k <= start_time_in_hours)
