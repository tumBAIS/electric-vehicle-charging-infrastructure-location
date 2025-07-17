import math
import heapq
import bisect
import enum
import logging
import copy
import network_cpp
from iterative_local_search.spprc_network import Network, Vertex, DecisionVariables, decision_spprc_network, Arc
from framework.solution_representation import SolutionRepresentation
from typing import Union, Dict, Tuple, Optional, List
from collections import defaultdict
from itertools import pairwise
from dataclasses import dataclass, replace

import faulthandler
faulthandler.enable()

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class LabelType(enum.IntFlag):
    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    JOINT = enum.auto()


@dataclass(slots=True)
class Label:
    arrival_time: float
    departure_time: float
    energy: float
    cost: float
    consumed_energy: float
    recharged_energy: float
    label_type: LabelType
    is_standalone_static_charger: bool
    crossed_dyn_segments: Optional[str] = None

    #def __eq__(self, other: "Label"):
    #    return self.cost==other.cost and self.energy==other.energy and self.departure_time==other.departure_time

    def __hash__(self):
        return hash((self.departure_time, self.energy, self.cost))

    def __lt__(self, other: "Label"):
        return (self.cost, -self.energy, self.departure_time) < (other.cost, -other.energy, other.departure_time)

    def __copy__(self) -> "Label":
        return replace(self)

    def valid_label(self, vertex: Vertex, soc_bounds: Tuple[float, float], restrict_arrival_time: bool = False) -> bool:
        # Note that we do not check self.time_window[0] <= label.departure_time because such labels are never created
        # The same applies for the lower bound on the arrival time (these labels are never created)
        if self.label_type==1:
            return vertex.departure_time_window[1] >= self.departure_time >= vertex.departure_time_window[0] and \
                   soc_bounds[1] >= self.energy >= soc_bounds[0]
        else:
            return vertex.departure_time_window[1] >= self.departure_time >= vertex.departure_time_window[0] and \
                   soc_bounds[1] >= self.energy and self.arrival_time <= vertex.arrival_time_window[1]

    @property
    def is_forward_label(self) -> bool:
        return bool(LabelType.FORWARD & self.label_type)

    @property
    def is_backward_label(self) -> bool:
        return bool(LabelType.BACKWARD & self.label_type)

    @property
    def is_joint_label(self) -> bool:
        return bool(LabelType.JOINT & self.label_type)


@dataclass(slots=True)
class LabelNode:
    """
    Store the label data needed to retrieve a path after a successful labeling algorithm run.
    Label data contains the current label of a vertex, its precedent vertex, and the label of the precedent vertex that
    leads to the current label.
    """
    current_label: Label
    precedent_vertex: str
    precedent_label: Label

    def __eq__(self, other: "LabelNode"):
        return self.current_label == other.current_label

    def __lt__(self, other: "LabelNode"):
        return self.current_label < other.current_label


@dataclass(slots=True)
class LabelTree:
    type: int
    #labels: defaultdict[str, list[LabelNode]]
    labels_by_cost: defaultdict[str, list[LabelNode]]
    label_bounds: defaultdict[str, list[tuple[float, float]]]

    def __init__(self, type: LabelType):
        self.type = 1 if bool(type & LabelType.FORWARD) else -1
        #self.labels = defaultdict(list)
        self.labels_by_cost = defaultdict(list)
        self.label_bounds = defaultdict(list)

    def add(self, label_node: LabelNode, current_vertex: str, pos: int) -> None:
        """
        Add a new label node to the tree of nodes. The label is inserted at its right position in the list of labels
        sorted by cost. Further the departure time and the energy is inserted into the list label bounds according
        to the position in the list sorted by cost. Afterward the label_bounds is repaired, so that at each position
        the max energy and min time of all labels at a position equal or smaller than the current is stored.
        @param label_node: node to insert
        @param current_vertex:
        @return: None
        """
        labels_by_cost = self.labels_by_cost[current_vertex]
        label_bounds = self.label_bounds[current_vertex]

        # insert label into list, and it's energy and departure time into label_bounds at the same position
        labels_by_cost.insert(pos, label_node)
        label_bounds.insert(pos, (self.type * label_node.current_label.energy, label_node.current_label.departure_time))

        # repair label_bounds if necessary
        if len(label_bounds) > 1:
            current_index = pos
            # if the element is inserted at position zero, we don't have to update the max with elements left of it
            if pos == 0:
                current_index = pos + 1
            values_changed = True
            # Iterate from the inserted label to right until we don't have to update anything
            while values_changed and current_index <= len(label_bounds) - 1:
                # update energy
                max_energy_changed = False
                current_max_energy = label_bounds[current_index][0]
                # compare the current label with the one before
                if label_bounds[current_index - 1][0] > current_max_energy:
                    current_max_energy = label_bounds[current_index - 1][0]
                    max_energy_changed = True
                # update departure time
                min_time_changed = False
                current_min_time = label_bounds[current_index][1]
                if label_bounds[current_index - 1][1] < current_min_time:
                    # compare the current label with the one before
                    current_min_time = label_bounds[current_index - 1][1]
                    min_time_changed = True
                values_changed = max_energy_changed or min_time_changed
                # update the tuple at the current position, if something changed
                if values_changed:
                    label_bounds[current_index] = (current_max_energy, current_min_time)
                current_index += 1
        self.labels_by_cost[current_vertex] = labels_by_cost
        self.label_bounds[current_vertex] = label_bounds

    def is_dominated(self, label_node: LabelNode, current_vertex: str) -> Tuple[bool, int]:
        """
        Checks if the current label of the label node considered is dominated by any of the elements in the labels
        (sorted by cost) at the resident vertex.
        @param label_node: node to check
        @param current_vertex:
        @return: bool
        """
        label_tree = self.labels_by_cost[current_vertex]
        cost_position = -1

        # if the current vertex has already been visited
        if label_tree:
            current_label = label_node.current_label
            # compute the right position of the current label in the sorted list of settled labels at the current vertex
            # Use the cost as key for sorting instead of the implemented label comparator
            cost_position = bisect.bisect(label_tree, current_label.cost, key=lambda x: x.current_label.cost)

            # in case it has the least costly label, it is not dominated
            if cost_position == 0:
                return False, cost_position

            # Bisect returns index of first element with higher cost --> start checking dominance to the left
            cost_position -= 1

            label_bounds = self.label_bounds[current_vertex]

            # check dominance against all the labels that have less cost, in reverse order
            for i in range(cost_position, -1, -1):
                # Check in each iteration if a dominate label can exist
                if (label_bounds[i][0] + 1e-3 < self.type * current_label.energy or
                        label_bounds[i][1] - 1e-1 > current_label.departure_time):
                    return False, cost_position+1
                settled_label = label_tree[i].current_label
                if settled_label.energy * self.type >= current_label.energy * self.type - 1e-3 and \
                        settled_label.departure_time <= current_label.departure_time + 1e-1:
                    return True, cost_position+1

        return False, cost_position+1


def forward_propagate_path(
        lb_tree: LabelTree, cut_vertex: str, best_scenario: LabelNode
) -> [list[tuple[str, float, float, float, float, float, bool, Optional[str]]], float]:
    """
    De-construct 'LabelTree' to find the solution to the subproblem as a sequence of expanded vertex IDs with
    associated resources (arc based attributes such as crossed dynamic segments are associated to the subsequent
    vertex)
    """
    #best_scenario = lb_tree.labels_by_cost[cut_vertex][idx]

    # print("Cost of solution : " + str(best_scenario.current_label.cost))
    path = []
    # Start by last label and go back in label tree
    current = cut_vertex
    current_label = best_scenario.current_label
    precedent = best_scenario.precedent_vertex
    precedent_label = best_scenario.precedent_label

    while current != "Depot":
        path += [
            (current, current_label.arrival_time, current_label.departure_time, current_label.energy,
            abs(current_label.consumed_energy), current_label.recharged_energy,
             current_label.is_standalone_static_charger, current_label.crossed_dyn_segments)
        ]
        found = False
        i = 0
        while not found:
            node = lb_tree.labels_by_cost[precedent][i]
            if node.current_label == precedent_label:
                current = precedent
                current_label = precedent_label
                precedent = node.precedent_vertex
                precedent_label = node.precedent_label
                found = True
            i += 1
    path += [
        ("Depot", precedent_label.arrival_time, precedent_label.departure_time, precedent_label.energy,
         precedent_label.consumed_energy, precedent_label.recharged_energy, False, None)
    ]
    # Return reconstructed path in the right order
    return path[::-1], best_scenario.current_label.cost


def backward_propagate_path(
        lb_tree: LabelTree, cut_vertex: str, best_scenario: LabelNode, soc_offset: float
) -> [list[tuple[str, float, float, float, float, float, bool, Optional[str]]], float]:
    """
    De-construct 'LabelTree' to find the solution to the subproblem as a sequence of expanded vertex IDs with
    associated resources (arc based attributes such as crossed dynamic segments are associated to the subsequent
    vertex)
    """
    #best_scenario = lb_tree.labels_by_cost[cut_point][idx]

    # print("Cost of solution : " + str(best_scenario.current_label.cost))
    path = []
    # Start by last label and go back in label tree
    current = cut_vertex
    current_label = best_scenario.current_label
    precedent = best_scenario.precedent_vertex
    precedent_label = best_scenario.precedent_label

    consumed_energy_inverter = abs(current_label.consumed_energy)
    recharged_energy_inverter = current_label.recharged_energy

    while current != "Depot-end":

        path += [
            (current, current_label.arrival_time, current_label.departure_time, current_label.energy + soc_offset,
             consumed_energy_inverter - abs(current_label.consumed_energy),
             recharged_energy_inverter - current_label.recharged_energy,
             current_label.is_standalone_static_charger, current_label.crossed_dyn_segments)
        ]
        found = False
        i = 0
        while not found:
            node = lb_tree.labels_by_cost[precedent][i]
            if node.current_label == precedent_label:
                current = precedent
                current_label = precedent_label
                precedent = node.precedent_vertex
                precedent_label = node.precedent_label
                found = True
            i += 1
    path += [
        ("Depot-end", precedent_label.arrival_time, precedent_label.departure_time, precedent_label.energy + soc_offset,
         consumed_energy_inverter, recharged_energy_inverter, False, current_label.crossed_dyn_segments)
    ]

    # crossed dynamic segments are always stored in the next label, i.e. we need to reverse them
    path_copy = path
    for p in range(1, len(path)):
        path[p] = path_copy[p][:7] + (path_copy[p-1][7],)

    # Return reconstructed path in the right order
    return path[::1], best_scenario.current_label.cost


def propagate_path(
        lb_tree_fw, lb_tree_bw, cut_point: str, label_node: LabelNode, min_soc: float
) -> [list[tuple[str, float, float, float, float, float, bool, Optional[str]]], float]:
    """
    De-construct 'LabelTree' to find the solution to the subproblem as a sequence of expanded vertex IDs with
    associated resources (arc based attributes such as crossed dynamic segments are associated to the subsequent
    vertex)
    """
    if label_node.current_label.is_forward_label:
        labels_by_cost = lb_tree_bw.labels_by_cost[cut_point]
        for lb_node in labels_by_cost:
            energy_matches = lb_node.current_label.energy <= label_node.current_label.energy
            time_matches = lb_node.current_label.departure_time >= label_node.current_label.departure_time
            if energy_matches and time_matches:
                bw_label_node = lb_node
                break
        fw_label_node = label_node
    else:
        bw_label_node = label_node
        labels_by_cost = lb_tree_fw.labels_by_cost[cut_point]
        for lb_node in labels_by_cost:
            energy_matches = lb_node.current_label.energy >= label_node.current_label.energy
            time_matches = lb_node.current_label.departure_time <= label_node.current_label.departure_time
            if energy_matches and time_matches:
                fw_label_node = lb_node
                break
    soc_offset = fw_label_node.current_label.energy - max(bw_label_node.current_label.energy, min_soc)
    forward_path, forward_cost = forward_propagate_path(lb_tree_fw, cut_point, fw_label_node)
    backward_path, backward_cost = backward_propagate_path(lb_tree_bw, cut_point, bw_label_node, soc_offset)

    # re-construct full path
    path, cost = forward_path+backward_path, forward_cost+backward_cost

    for idx in range(len(forward_path), len(path)):
        path[idx] = path[idx][:4] + (path[idx][4] + forward_path[-1][4], path[idx][5] + forward_path[-1][5],) + path[idx][6:]
    return path, cost


def get_next_label(network: Network, next_vertex: str, current_arc: Arc, current_label: Label,
                   restrict_arrival_time: bool = False) -> Label:
    """
    Compute the label at the next vertex, based on a given label at the current vertex
    @param network: vehicle specific network
    @param next_vertex: vertex to which we extend the current label at the current vertex
    @param current arc: the arc to travers from current to next vertex
    @param current_label: Label currently extended
    @param restrict_arrival_time: bool (defaults to False) indicating if arrival before arrival time window is enforced
    @return: the label at the next vertex
    """
    next_vertex_object = network.get_vertex(next_vertex)
    if current_label.is_forward_label:
        arrival_time = current_label.departure_time + current_arc.time
        if restrict_arrival_time:
            # if we worry about arrival time (i.e., in conflict resolution), arriving early is not allowed
            # this line is basically assuming idling on arcs (e.g., by driving slower) is possible
            arrival_time = max(arrival_time, next_vertex_object.arrival_time_window[0])
        departure_time = next_vertex_object.departure_time_window[0]
        if departure_time < arrival_time:
            # No label can have a departure time that violates the departure time window
            departure_time = arrival_time
        recharged_energy = current_arc.recharged_energy
        consumed_energy = current_arc.consumed_energy
        energy = current_label.energy - consumed_energy + recharged_energy
        cost = current_label.cost + consumed_energy * network.consumption_cost + \
               _energy_price(current_label.departure_time, True, energy_prices=network.energy_prices,
                             consumption_cost=network.consumption_cost) * recharged_energy
        return Label(arrival_time, departure_time, energy, cost, current_label.consumed_energy - consumed_energy,
            current_label.recharged_energy + recharged_energy, LabelType.FORWARD,
            not (next_vertex_object.is_stop or next_vertex_object.is_depot), current_arc.encoded_dyn_charger,
        )

    departure_time = min(current_label.arrival_time - current_arc.time, next_vertex_object.departure_time_window[1])
    arrival_time = next_vertex_object.departure_time_window[1]
    if restrict_arrival_time:
        # if we worry about arrival time (i.e., in conflict resolution), arriving early is not allowed
        # this line is basically assuming idling on arcs (e.g., by driving slower) is possible
        arrival_time = max(arrival_time, next_vertex_object.arrival_time_window[0])
    if departure_time < arrival_time:
        # No label can have a departure time that violates the departure time window
        arrival_time = departure_time
    recharged_energy = current_arc.recharged_energy
    consumed_energy = current_arc.consumed_energy
    energy = max(current_label.energy, network.min_soc_in_kwh) + consumed_energy - recharged_energy
    cost = current_label.cost + consumed_energy * network.consumption_cost + \
           _energy_price(current_label.arrival_time, True, energy_prices=network.energy_prices,
                         consumption_cost=network.consumption_cost) * recharged_energy
    return Label(arrival_time, departure_time, energy, cost, current_label.consumed_energy - consumed_energy,
                 current_label.recharged_energy + recharged_energy, LabelType.BACKWARD,
                 not (next_vertex_object.is_stop or next_vertex_object.is_depot), current_arc.encoded_dyn_charger)


def lower_energy_bounds(spprc_network: Network, route_id: int, decision: DecisionVariables = None) -> dict[str, float]:
    """
    Compute, at each vertex, the minimum cost and energy required to complete the route; the computation relies on
    every vertex of spprc_network either being a stop or the depot or having such direct successor.
    @param spprc_network: Instance of Network
    @param route_id: integer id corresponding to index of route in list of routes
    @param decision: DecisionVariables object
    @return: a map between each vertexID (str) and the lower bound of the cost associated
    """

    first_stat_representations = []
    new_keys = set(spprc_network.vertices.keys())
    remaining_keys = new_keys

    try:
        old_bounds = decision.lower_energy_bounds[route_id]
        old_keys = set(old_bounds.keys())
        similar_keys = new_keys.intersection(old_keys)
        remaining_keys = new_keys-similar_keys

        if len(similar_keys) == len(old_keys) == len(new_keys):
            # Route is not affected by changes. Lower bounds neither
            return old_bounds

        bounds = {k: old_bounds[k] for k in similar_keys}

    except Exception:

        # No lower bounds computed yet
        bounds = {}
        energy = 0

        # first compute the minimum cost and energy required to complete the route at each stop
        for next_stop, current_stop in pairwise(reversed(spprc_network.route.stops)):
            # Consumption is equal on all parallel arcs
            arc = spprc_network.get_arc(current_stop.id, next_stop.id, 0)
            energy += abs(arc.consumed_energy)
            bounds[current_stop.id] = energy
            remaining_keys.remove(current_stop.id)

        bounds['Depot-end'] = 0
        remaining_keys.remove('Depot-end')

    # Compute remaining nodes, these nodes aren't stops
    # The distance between each of them and a stop is not more than 1
    for vertex_id in remaining_keys:
        vertex = spprc_network.get_vertex(vertex_id)
        if not vertex.is_stop and not vertex.is_depot:
            has_stop_successor = False
            successors = [spprc_network.get_vertex(s) for s in spprc_network.successors(vertex_id)]
            cntr = 0
            while cntr < len(successors):
                successor = successors[cntr]
                if successor.is_stop or successor.is_depot:
                    has_stop_successor = True
                    break
                if len(successors)==0:
                    break
                cntr += 1
            if has_stop_successor:
                # the lower bound utilizes consumption (this should be equivalent between all parallel edges)
                # Additionally, these connections (charger - stop/depot) exist only once
                arc = spprc_network.get_arc(vertex.id, successor.id, 0)
                energy = bounds.get(successor.id) + abs(arc.consumed_energy)
                bounds[vertex.id] = energy
            else:
                first_stat_representations.append((vertex, successor))
        else:
            if vertex.id not in bounds.keys():
                raise ValueError

    # only first representations bear no successor that is a stop, but they share the lower bound of the second rep.
    for (first_rep, second_rep) in first_stat_representations:
        bounds[first_rep.id] = bounds[second_rep.id]

    assert len(bounds) == len(spprc_network.vertices)

    # Save new bounds
    decision.lower_energy_bounds[route_id] = bounds
    return bounds


def _heuristic_cost_component_forward(network: Network, lower_energy_bound: float, next_label: Label) -> float:
    """
    Heuristic cost component to guide label exploration in forward setting
    """
    return max(network.min_soc_in_kwh + lower_energy_bound - next_label.energy, 0) * _min_recharge_price(
                next_label.arrival_time, 1e6, network.energy_prices) + lower_energy_bound * network.consumption_cost


def _heuristic_cost_component_backward(network: Network, lower_energy_bound: float, next_label: Label, init_soc: float) -> float:
    """
    Heuristic cost component to guide label exploration in backward setting
    """
    return max(next_label.energy + lower_energy_bound - init_soc, 0) * _min_recharge_price(
                -1e6, next_label.departure_time, network.energy_prices) + lower_energy_bound * network.consumption_cost


def A_star_intermediate(
        network: Network,
        decision: DecisionVariables,
        route_index: int,
        restrict_arrival_time: bool = False
) -> Tuple[LabelTree, LabelTree, str, LabelNode]:
    """
    Build a set / mapping of labels as instance of class LabelTree that contains all created labels when searching for
    the resource constrained shortest path between 'Depot' and 'Depot-end'
    @param network: a vehicle specific instance of class 'Network'
    @param decision: object of DecisionVariables carrying the current configuration and parameters
    @param restrict_arrival_time: bool (defaults to False) indicating if arrival times are restricted (
    (e.g., for conflict resolution)
    @return: Instance of class 'LabelTree'
    """
    # compute the lower bounds and cache them
    lb_energy = lower_energy_bounds(network, route_index, decision)
    heuristic_solution_available = False

    # compute integer IDs
    int_ids = {v.id: i for i, v in enumerate(list(network.vertices.values()))}

    forward_depot_label = Label(
        0, network.get_vertex('Depot').departure_time_window[0], decision.soc_init, 0, 0, 0, LabelType.FORWARD, False)
    backward_depot_label = Label(
        network.get_vertex('Depot-end').departure_time_window[1],
        network.get_vertex('Depot-end').departure_time_window[1], decision.min_soc, 0, 0, 0, LabelType.BACKWARD, False
    )

    U = [
        (0.0, 0, int_ids["Depot"], 'Depot', LabelNode(forward_depot_label, 'Depot', forward_depot_label)),
        (0.0, 0, int_ids["Depot-end"], "Depot-end", LabelNode(backward_depot_label, "Depot-end", backward_depot_label))
    ]
    heapq.heapify(U)
    best_labels_forward = LabelTree(LabelType.FORWARD)
    best_labels_backward = LabelTree(LabelType.BACKWARD)

    def convertToInt(num: float, decimalPlaces: int):
        temp = num * math.pow(10.0, decimalPlaces)
        return math.trunc(temp)

    def can_be_merged(v: str, l_q: LabelNode) -> Tuple[bool, Optional[LabelNode], Optional[int]]:
        label: Optional[LabelNode]
        idx: int
        if l_q.current_label.is_forward_label:
            try:
                for idx, label_node in enumerate(best_labels_backward.labels_by_cost[v]):
                    label: Label = label_node.current_label
                    if l_q.current_label.energy >= label.energy \
                            and l_q.current_label.departure_time <= label.departure_time:
                        return True, label_node, idx
                    continue
                return False, None, None
            except KeyError:
                return False, None, None
        if l_q.current_label.is_backward_label:
            try:
                for idx, label_node in enumerate(best_labels_forward.labels_by_cost[v]):
                    label: Label = label_node.current_label
                    if l_q.current_label.energy <= label.energy \
                            and l_q.current_label.departure_time >= label.departure_time:
                        return True, label_node, idx
                    continue
                return False, None, None
            except KeyError:
                return False, None, None
    iter = 0
    dominated_labels = 0
    invalid_labels = 0
    fw_labels = 0
    bw_labels = 0
    while U:

        # Return heuristic solution in case more than 1e5 iterations required
        iter += 1
        if iter > 1e5 and heuristic_solution_available:
            while True:
                key1, key2, key3, v_q, l_q = heapq.heappop(U)
                if l_q.current_label.is_joint_label:
                    logging.info(f"Return heuristic solution with value {key1/1e3} after {iter-1} iterations")
                    return best_labels_forward, best_labels_backward, v_q, l_q

        # pop the first element of the queue
        key1, key2, key3, v_q, l_q = heapq.heappop(U)

        if l_q.current_label.is_joint_label:
            #logging.info(f"{iter}, {key1/1e3}, {dominated_labels}, {invalid_labels}, {fw_labels}, {bw_labels}")
            return best_labels_forward, best_labels_backward, v_q, l_q

        # as long as the next element of the queue is dominated, keep popping
        # we try now to only check first element in queue for dominance
        if l_q.current_label.is_forward_label:
            is_dominated, position = best_labels_forward.is_dominated(l_q,v_q)
        else:
            is_dominated, position = best_labels_backward.is_dominated(l_q, v_q)

        if is_dominated:
            dominated_labels += 1
            continue

        else:
            # explore the successors / predecessors
            if l_q.current_label.is_forward_label:
                fw_labels += 1
                outgoing_arcs = network.outgoing_arcs[v_q]
                has_valid_successor_label = False
                for outgoing_arc in outgoing_arcs:
                    successor_vertex = network.get_vertex(outgoing_arc[0])
                    next_label = get_next_label(
                        network=network,
                        current_arc=outgoing_arc[2],
                        next_vertex=outgoing_arc[0],
                        current_label=l_q.current_label,
                        restrict_arrival_time=restrict_arrival_time
                    )
                    if next_label.valid_label(
                            successor_vertex, (network.min_soc_in_kwh, network.max_soc_in_kwh), restrict_arrival_time
                    ):
                        has_valid_successor_label = True
                        # compute the key of the successor and check if it's lower than the upper bound
                        next_key = next_label.cost + _heuristic_cost_component_forward(
                            network,
                            lb_energy[outgoing_arc[0]],
                            next_label
                        )
                        is_stop = 0 if (successor_vertex.is_stop or successor_vertex.is_depot) else 1
                        heapq.heappush(U, (convertToInt(next_key, 3), is_stop, -int_ids[outgoing_arc[0]],
                                           outgoing_arc[0], LabelNode(next_label, v_q, l_q.current_label)))
                    else:
                        invalid_labels+=1
                if has_valid_successor_label or v_q=="Depot-end":
                    best_labels_forward.add(l_q, v_q, position)

            elif l_q.current_label.is_backward_label:
                bw_labels += 1
                incoming_arcs = network.incoming_arcs[v_q]
                has_valid_successor_label = False
                for incoming_arc in incoming_arcs:
                    predecessor_vertex = network.get_vertex(incoming_arc[0])
                    next_label = get_next_label(
                        network=network,
                        current_arc=incoming_arc[2],
                        next_vertex=incoming_arc[0],
                        current_label=l_q.current_label,
                        restrict_arrival_time=restrict_arrival_time
                    )
                    if next_label.valid_label(
                            predecessor_vertex,
                            (network.min_soc_in_kwh, network.max_soc_in_kwh),
                            restrict_arrival_time
                    ):
                        # log min. one valid successor label
                        has_valid_successor_label = True
                        # compute the key of the successor and check if it's lower than the upper bound
                        next_key = next_label.cost + _heuristic_cost_component_backward(
                            network,
                            lb_energy["Depot"]-lb_energy[incoming_arc[0]],
                            next_label,
                            decision.soc_init,
                        )
                        is_stop = 0 if (predecessor_vertex.is_stop or predecessor_vertex.is_depot) else 1
                        heapq.heappush(U, (convertToInt(next_key, 3), is_stop, int_ids[incoming_arc[0]] - int_ids["Depot-end"],
                                           incoming_arc[0], LabelNode(next_label, v_q, l_q.current_label)))
                    else:
                        invalid_labels+=1
                if has_valid_successor_label or v_q == "Depot":
                    best_labels_backward.add(l_q, v_q, position)

            joint_label, complementary_label, _ = can_be_merged(v_q, l_q)

            # check if label is a joint label
            if joint_label:
                logging.debug("joint label found")
                new_l_q = copy.copy(l_q.current_label)
                new_l_q.label_type |= LabelType.JOINT
                heapq.heappush(
                    U, (
                        convertToInt(l_q.current_label.cost + complementary_label.current_label.cost, 3),
                        -1, int_ids[v_q], v_q, LabelNode(new_l_q, l_q.precedent_vertex, l_q.precedent_label)
                    )
                )
                heuristic_solution_available = True

    #logging.info(f"{iter}, - , {dominated_labels}, {invalid_labels}, {fw_labels}, {bw_labels}")
    logging.debug("Python subproblem solution finished (infeasible)")
    raise ValueError("No feasible solution found")

def A_star_old(
        network: Network,
        decision: DecisionVariables,
        route_index: int,
        verbose: bool=False,
) -> SolutionRepresentation:
    """
    Apply an A star algorithm to look for the best feasible path for the route in terms of cost.
    If no such a path exists, raise ValueError.
    Otherwise, reconstruct the best path in terms of cost, and returns the corresponding SolutionRepresentation
    object.
    @param network: a vehicle specific instance of class 'Network'
    @param decision: all the decision variables
    @param route_index: the index of the route in the intermediate representation routes (in the decision variable)
    that's being solved
    @param verbose: bool (False by default) - decides whether to print the data table of the solution path
    @param restrict_arrival_time: bool (False by default) indicating if arrival times are restricted
    (e.g., for conflict resolution)
    @return: a SolutionRepresentation object of the best path of the vehicle on the chosen route
    """
    lb_tree_forward, lb_tree_backward, meeting_vertex, label_node = A_star_intermediate(
        network=network, decision=decision, route_index=route_index
    )
    best_path, total_cost = propagate_path(lb_tree_forward, lb_tree_backward, meeting_vertex, label_node, network.min_soc_in_kwh)
    return network.to_solution_representation(
        best_path, decision.soc_init, decision.intermediate_rep.routes[route_index].vehicle_id,
        decision.construct_static_invest(),
        decision.construct_dynamic_invest(), total_cost, verbose
    )

def A_star(
        network: Network,
        decision: DecisionVariables,
        route_index: int,
        verbose: bool=False,
        interpreter: str = "cpp",
) -> SolutionRepresentation:
    """
    Apply an A star algorithm to look for the best feasible path for the route in terms of cost.
    If no such a path exists, raise ValueError.
    Otherwise, reconstruct the best path in terms of cost, and returns the corresponding SolutionRepresentation
    object.
    @param network: a vehicle specific instance of class 'Network'
    @param decision: all the decision variables
    @param route_index: index of route to solve
    @param verbose: message printing (argument not maintained)
    @param interpreter: indicates choice between C++ and Python implementation
    @return: Solution Representation (Attention: may not be final as there is sum defined on this class and creation
    happens step-wise)
    """
    if interpreter=="cpp":
        lower_bound = lower_energy_bounds(network, route_index, decision)

        # Flat representation of network (i.e., preparing sparse representation in C++)
        logging.debug("Preparing plain network representation started")
        route, vertices, arcs, energy_prices, id2index, charger_mapping = network.get_plain_representation()
        logging.debug("Preparing plain network representation finished")
        lower_bound = {id2index[k]: bound for k, bound in lower_bound.items()}
        lower_bound_lst = [lower_bound[i] for i in range(len(lower_bound))]

        # Calling subproblem
        result = network_cpp.spprc(
            lower_bound_lst, route, energy_prices, decision.soc_init, network.consumption_cost,
            vertices, arcs, (network.min_soc_in_kwh, network.max_soc_in_kwh))

        if result is None:
            logging.debug("Cpp subproblem solution finished (infeasible)")
            raise ValueError("No feasible solution found")

        (best_path_raw, total_cost) = result

        # Converting result back to Python Representation
        index2id = {v: k for k, v in id2index.items()}
        best_path = [
            (index2id[x], y, z, k, i, j, not network.get_vertex(index2id[x]).is_stop_or_depot,
             charger_mapping[l])
            for x, y, z, k, i, j, l in best_path_raw
        ]
        sol_rep = network.to_solution_representation(
            path=best_path, soc_init=decision.soc_init,
            vehicle_id=decision.intermediate_rep.routes[route_index].vehicle_id, static_invest=decision.construct_static_invest(),
            dynamic_invest=decision.construct_dynamic_invest(), total_cost=total_cost, verbose=False
        )
        logging.debug(f"Cpp subproblem solution finished {sol_rep.routing_cost}")
    else:
        logging.debug("Python subproblem solution started")
        sol_rep = A_star_old(network, decision, route_index, verbose)
        logging.debug(f"Python subproblem solution finished {sol_rep.routing_cost}")
    return sol_rep


def best_routing(
        decision: DecisionVariables,
        time_step=60,
) -> SolutionRepresentation:
    """
    Compute the best routing of all routes.
    @param decision: instance of class 'DecisionVariables'
    @param time_step: time step of the SPPRC network
    @param verbose: if True, print the solution data table
    @param interpreter: "cpp" for C++ or "python" for Python 3.10.2
    @param multithreading: if True, run multiple threads in parallel
    @return: the solution of the routing
    """
    solution = SolutionRepresentation(
        [],decision.construct_dynamic_invest(), decision.construct_static_invest(), 0.0, 0.0, 0.0
    )

    # sort (such that routes which were previously infeasible are evaluated first)
    # we hope that we can save unnecessary but costly solves of the subproblem
    sorted_routes = sorted(
        [
            (route_index, route, route_index in decision.deprioritize_route_index)
            for route_index, route in enumerate(decision.intermediate_rep.routes)
        ], key=lambda x: x[2]
    )

    for route_index, route, _ in sorted_routes:
        # remove de-prioritization indices if it exists
        decision.deprioritize_route_index.remove(route_index) if route_index in decision.deprioritize_route_index else None
        if not isinstance(decision.route_networks[route_index], int):
            # update static and dynamic invest
            decision.route_networks[route_index].static_invest = decision.construct_static_invest()
            decision.route_networks[route_index].dynamic_invest = decision.construct_dynamic_invest()
            solution += decision.route_networks[route_index]
            logging.debug(f"Route {route_index} does not require recalculation")
        else:
            a = decision_spprc_network(decision, route, time_step)
            sol_rep = A_star(
                network=a,
                decision=decision,
                route_index=route_index,
            )
            logging.debug(f"{route_index}: {sol_rep if sol_rep is None else round(sol_rep.global_cost,2)}")
            if sol_rep is None:
                raise ValueError
            solution += sol_rep
            decision.route_networks[route_index] = sol_rep
        decision.deprioritize_route_index.append(route_index)
    return solution


def objective_cost_value(decision: DecisionVariables, time_step, interpreter, multithreading) -> float:
    """
    Compute the cost of best routing
    @param decision: instance of class 'DecisionVariables'
    @param time_step: time step of the SPPRC network
    @param interpreter: "cpp" vs "python"
    @param multithreading: True if multithreading is allowed
    @return: the cost of the best routing
    """
    return best_routing(
        decision=decision,
        time_step=time_step
    ).global_cost


def _energy_price(arrival_time: float, charging: bool, energy_prices: Union[float, Dict[int, float]], consumption_cost: float) -> float:
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
        arrival_time_in_hours = math.floor(arrival_time / 3600)
        return energy_prices[arrival_time_in_hours]
    return consumption_cost


def _min_recharge_price(start_time: float, end_time: float, energy_prices: Union[float, Dict[int, float]]) -> float:
    """
    Return the minimum future recharging energy price
    @param start_time: current time (i.e., only consider times after this one)
    @param energy_prices: prices
    @return: Minimum future recharging price after start_time as float
    """
    if isinstance(energy_prices, float):
        return energy_prices
    start_time_in_hours = math.floor(start_time / 3600)
    end_time_in_hours = math.floor(end_time / 3600)
    return min(v for k,v in energy_prices.items() if k>=start_time_in_hours and k<=end_time_in_hours)