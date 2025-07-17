import math
import itertools
import enum
from typing import List, Set, Tuple, Optional, Any, Dict
from datetime import timedelta, datetime, timezone
from dataclasses import dataclass, field
from itertools import product, pairwise
from framework.intermediate_representation import VehicleID, VertexID, IntermediateRepresentation
from framework.solution_representation import SolutionRepresentation, Itinerary, Point
from iterative_local_search.conflict import Conflict
from iterative_local_search.conflict import detect_conflict_from_solution
from docplex.mp.model import Model
from docplex.mp.dvar import Var
from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.solution import SolveSolution

EPSILON = 1
# hof & test cases: all big-M constraints 1e5
# solomon:: 1e9
BIG_M_1 = 1e5
BIG_M_2 = 1e5
BIG_M_3 = 1e5
BIG_M_4 = 1e5
HOUR2SECONDS = 3600


class StaggeredArcType(enum.IntFlag):
    CHARGING = enum.auto()
    CONNECTING = enum.auto()


@dataclass(frozen=True)
class StaggeredNode:
    id: int
    name: VertexID
    time_window: Tuple[int, int]

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass(frozen=True)
class StaggeredArc:
    id: Tuple[VertexID, VertexID]
    type: StaggeredArcType
    name: Optional[str] = None

    def __eq__(self, other):
        oid = other.id if isinstance(other, StaggeredArc) else other
        return self.id == oid

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"{self.id[0]}_{self.id[1]}"

    @property
    def is_conflicted(self) -> bool:
        return bool(self.type & StaggeredArcType.CHARGING)


@dataclass
class ConflictResolutionNetwork:
    """
    Maintains the network.
    Class relies on sorted lists, i.e., changing indices in any of the below lists breaks the class.
    """
    vehicles: List[VehicleID]
    trips: List[List[List[StaggeredArc]]]
    travel_times: List[List[List[int]]]  # seconds (algorithm relies on floats - lets correct that here)
    start_time_windows: List[List[List[Tuple[int, int]]]]
    route_time_window: List[Tuple[int, int]]
    helper: Dict[Any, Any]

    def __post_init__(self):
        assert len(self.vehicles) == len(self.trips) == len(self.travel_times)
        for idx in range(len(self.trips)):
            assert len(self.trips[idx]) == len(self.travel_times[idx])

    @property
    def conflicting_arcs(self) -> Set[StaggeredArc]:
        return {a.id for lla in self.trips for la in lla for a in la if a.type == StaggeredArcType.CHARGING}

    @property
    def arcs(self) -> Set[StaggeredArc]:
        return {a.id for lla in self.trips for la in lla for a in la}

    @property
    def trip_indices(self):
        return [(oidx, iidx) for oidx, vl in enumerate(self.trips) for iidx, l in enumerate(vl)]

    @property
    def vehicle_indices(self):
        return range(len(self.vehicles))


def _parse_conflict_resolution_graph(
        sol: SolutionRepresentation, ir: IntermediateRepresentation, vehicle_max_speed: float
) -> ConflictResolutionNetwork:
    """Static function to parse the graph"""
    vehicles = []
    trips = []
    travel_times = []
    windows = []
    route_time_windows =[]

    # object mapping
    mapping = {}
    helper = {}

    for itinerary in sol.itineraries:
        vehicles.append(itinerary.vehicle)
        trips.append([[]])
        travel_times.append([[]])
        windows.append([[]])
        idx = 0

        # filter out dynamic representations & get IR representation of route
        route = [point for point in itinerary.route]# if not point.is_synthetic_dyn_charger_representation]
        ir_route = [point.id for point in route]

        # we somehow need to count occurences
        active_arcs = set()
        route_object = ir.get_route(itinerary.vehicle)
        route_time_windows.append(
            (int(route_object.stop_sequence[0].earliest_time_of_service), int(route_object.stop_sequence[-1].latest_time_of_service))
        )
        for index, (vertex1, vertex2) in enumerate(itertools.pairwise(ir_route)):

            # check if we need to break the sequence
            if (vertex1, vertex2) in active_arcs or (vertex2, vertex2) in active_arcs:
                idx+=1
                trips[-1].append([])
                travel_times[-1].append([])
                windows[-1].append([])
                active_arcs = set()

            # we need to lock information for vertex1
            if (vertex1, vertex2) in mapping:
                staggered_arc = mapping[(vertex1, vertex2)]
            else:
                staggered_arc = StaggeredArc(
                    id = (vertex1, vertex2),
                    type=StaggeredArcType.CONNECTING,
                    name=f"{vertex1}-{vertex2}"
                )
                mapping[(vertex1, vertex2)] = staggered_arc

            # add to active arcs
            active_arcs.add((vertex1, vertex2))

            trips[-1][idx].append(staggered_arc)
            helper[(len(trips)-1,index, vertex1, vertex2)] = ((len(trips)-1,idx),(vertex1, vertex2), len(travel_times[-1][idx]))
            if not route[index].is_synthetic_dyn_charger_representation and not route[index+1].is_synthetic_dyn_charger_representation:
                travel_times[-1][idx].append(int(ir.get_arc(vertex1, vertex2).get_travel_time_seconds(vehicle_max_speed)))
            else:
                travel_times[-1][idx].append(route[index+1].arrival_time_int - route[index].departure_time_int)

            if route[index].is_stop or route[index].is_depot:
                windows[-1][idx].append((route_object.earliest_departure_time, route_object.latest_arrival_time))
            else:
                windows[-1][idx].append((0, math.inf))
            if route[index+1].is_static_charger:
                # We retain objects, i.e., one segment is always represented by the same object
                if (vertex2, vertex2) in mapping:
                    staggered_arc = mapping[(vertex2, vertex2)]
                else:
                    staggered_arc = StaggeredArc(
                        id=(vertex2, vertex2),
                        type=StaggeredArcType.CHARGING,
                        name=f"{vertex2}"
                    )
                active_arcs.add((vertex2, vertex2))
                trips[-1][idx].append(staggered_arc)
                helper[(len(trips)-1, index, vertex2, vertex2)] = ((len(trips)-1, idx), (vertex2, vertex2), len(travel_times[-1][idx]))
                artificial_travel_time = int((
                        route[index+1].accumulated_charged_energy - route[index].accumulated_charged_energy
                ) * HOUR2SECONDS / next(iter(ir.get_vertex(route[index+1].id).constructible_charger)).charging_rate)
                travel_times[-1][idx].append(artificial_travel_time)
                windows[-1][idx].append((route_object.earliest_departure_time, route_object.latest_arrival_time))

    return ConflictResolutionNetwork(
        vehicles=vehicles,
        trips=trips,
        travel_times=travel_times,
        start_time_windows=windows,
        route_time_window=route_time_windows,
        helper=helper,
    )


def _construct_conflict_free_solution(sol: SolutionRepresentation, staggered_solution: Dict[Any, Var], travel_times: List[List[List[int]]], helper: Dict[Any, Any]) -> SolutionRepresentation:
    """private function used to construct the conflict free solution (if available)"""
    # we update inplace and make use of the fact that only time dimension changes

    for v_idx, itin in enumerate(sol.itineraries):
        loop_cntr = 0
        active_arcs = set()
        route = [p for p in itin.route[:-1]]
        for p_idx, (p1, p2) in enumerate(pairwise(route)):

            # in this case we have handled it in the previous iteration
            if not p1.is_static_charger:
                ids = helper[(v_idx, p_idx, p1.id, p2.id)]
                p1.departure_time = int(staggered_solution[(ids[0], ids[1])].solution_value)
                p1.arrival_time = p1.departure_time

            if p2.is_static_charger:
                ids = helper[(v_idx, p_idx, p2.id, p2.id)]
                p2.arrival_time = int(staggered_solution[(ids[0], ids[1])].solution_value)
                tt = travel_times[ids[0][0]][ids[0][1]][ids[2]]
                p2.departure_time = p2.arrival_time_int + tt

        # deal with last point (i.e., depot)
        depot_end_point = itin.route[-1]
        last_point = [p for p in itin.route][-2]
        depot_end_point.arrival_time = last_point.departure_time_int + travel_times[v_idx][-1][-1]
        depot_end_point.departure_time = depot_end_point.arrival_time

    assert len(detect_conflict_from_solution(sol)) == 0, f"{detect_conflict_from_solution(sol)}"
    return sol


def resolve_conflict(
        sol: SolutionRepresentation, ir: IntermediateRepresentation, vehicle_max_speed: float, time_limit: int=600
) -> Optional[SolutionRepresentation]:
    """
    Resolve the conflicts in the given solution by staggered conflict resolution  --- this is the only inter-
    face to ILS. Should throw a ValueError in Case conflict free solution cannot be found by staggering
    """
    network = _parse_conflict_resolution_graph(sol, ir, vehicle_max_speed)
    resolver = StaggeredConflictResolver(network, time_limit)
    start_variables = resolver.solve()
    # infeasible: 3; integer infeasible: 103, time limit infeasible: 108
    if resolver.model.solve_details.status_code in [3, 103, 108]:
        return None
   # resolver.model.print_solution(print_zeros=True)
    return _construct_conflict_free_solution(sol, start_variables, network.travel_times, network.helper)


class StaggeredConflictResolver:
    """
    Constructs and runs the MIP model.
    """
    network: ConflictResolutionNetwork
    model: Model

    def __init__(self, network, timelimit):
        self.network = network
        self.model = Model("StaggeredConflictResolution")

        # set some parameters
        self.model.parameters.mip.tolerances.mipgap = 1  # return first feasible solution?
        self.model.parameters.timelimit = timelimit  # tbd?
        self.model.context.cplex_parameters.preprocessing.presolve = 0

    def solve(self) -> Optional[Dict[Any, Var]]:
        start_variables = self._build_model()
        sol = self.model.solve(log_output=True)
        # if sol is None:
        #     refiner = ConflictRefiner()
        #     conflicts = refiner.refine_conflict(self.model, display=True)
        #
        #     for conf in conflicts:
        #         print(conf.element)
        # status code 3 encodes infeasibility
        if self.model.solve_details.status_code == 3:
            return None
        return start_variables

    def get_relevant_arcs(self, trip: Tuple[int, int]):
        return [a.id for a in self.network.trips[trip[0]][trip[1]]]

    def _build_model(self) -> Dict[Any, Var]:

        # we need a start time for all arcs
        s = self.model.continuous_var_dict(
            keys=((trip, arc) for trip in self.network.trip_indices for arc in self.get_relevant_arcs(trip)),
            lb=0,
            name=lambda key: f"s^{key[0]}_{key[1][0]}_{key[1][1]}"
        )

        # f unrestricted for arcs that are not conflicted
        f = self.model.continuous_var_dict(
            keys=product(self.network.trip_indices, self.network.arcs),
            lb=0,
            name=lambda key: f"f^{key[0]}_{key[1][0]}_{key[1][1]}"
        )

        # helping binary variables
        alpha = self.model.binary_var_dict(
            keys=product(self.network.trip_indices, self.network.trip_indices, self.network.conflicting_arcs),
            name=lambda key: f"alpha^{key[0]}_{key[1]}_{key[2][0]}_{key[2][1]}"
        )
        beta = self.model.binary_var_dict(
            keys=product(self.network.trip_indices, self.network.trip_indices, self.network.conflicting_arcs),
            name=lambda key: f"beta^{key[0]}_{key[1]}_{key[2][0]}_{key[2][1]}"
        )
        gamma = self.model.binary_var_dict(
            keys=product(self.network.trip_indices, self.network.trip_indices, self.network.conflicting_arcs),
            name=lambda key: f"gamma^{key[0]}_{key[1]}_{key[2][0]}_{key[2][1]}"
        )

        # add objective - sum up travel time on potentially conflicting arcs (charger arcs)
        # this set up is very degenerate (there is potentially infinitely many solutions with the same value (all
        # unconflicted ones)
        self.model.minimize(1)

        # add constrains (3.1c)
        self.model.add_constraints(
            (s[trip_index, self.network.trips[trip_index[0]][trip_index[1]][idx+1]] >=
            s[trip_index, arc] + self.network.travel_times[trip_index[0]][trip_index[1]][idx]
            for trip_index in self.network.trip_indices
            for idx, arc in enumerate(self.network.trips[trip_index[0]][trip_index[1]])
            if idx != len(self.network.trips[trip_index[0]][trip_index[1]])-1),  # exclude last arc
            # names=(f"c_{trip_index[0]}_{trip_index[1]}_{idx}"
            # for trip_index in self.network.trip_indices
            # for idx, arc in enumerate(self.network.trips[trip_index[0]][trip_index[1]])
            # if idx != len(self.network.trips[trip_index[0]][trip_index[1]]) - 1)
        )

        # add constraints (3.1e) (adapted to this use case)
        # self.model.add_constraints(
        #     self.network.start_time_windows[trip_index[0]][trip_index[1]][idx][0] <= s[trip_index, arc]
        #     for trip_index in self.network.trip_indices
        #     for idx, arc in enumerate(self.network.trips[trip_index[0]][trip_index[1]])
        # )
        # self.model.add_constraints(
        #     s[trip_index, arc] <= self.network.start_time_windows[trip_index[0]][trip_index[1]][idx][1]
        #     for trip_index in self.network.trip_indices
        #     for idx, arc in enumerate(self.network.trips[trip_index[0]][trip_index[1]])
        # )

        # add linearization constraints (3.1h)
        self.model.add_constraints(
            s[trip_index, arc] - s[trip_index_prime, arc] + EPSILON <= BIG_M_1 * alpha[trip_index, trip_index_prime, arc]
            for trip_index in self.network.trip_indices
            for trip_index_prime in self.network.trip_indices
            for arc in self.network.conflicting_arcs
            if trip_index != trip_index_prime and arc in self.network.trips[trip_index[0]][trip_index[1]] and
            arc in self.network.trips[trip_index_prime[0]][trip_index_prime[1]]
        )

        # add linearization constraints (3.1i)
        self.model.add_constraints(
            s[trip_index_prime, arc] - s[trip_index, arc] <= BIG_M_2 * (1 - alpha[trip_index, trip_index_prime, arc])
            for trip_index in self.network.trip_indices
            for trip_index_prime in self.network.trip_indices
            for arc in self.network.conflicting_arcs
            if trip_index != trip_index_prime and arc in self.network.trips[trip_index[0]][trip_index[1]] and
            arc in self.network.trips[trip_index_prime[0]][trip_index_prime[1]]
        )

        # add linearization constraints (3.1j) x[trip_index_prime, arc] -
        self.model.add_constraints(
            s[trip_index_prime, arc] + self.network.travel_times[trip_index_prime[0]][trip_index_prime[1]][idx] - \
            s[trip_index, arc] <= BIG_M_3 * beta[trip_index, trip_index_prime, arc] # + x[trip_index_prime, arc]
            for trip_index in self.network.trip_indices
            for trip_index_prime in self.network.trip_indices
            for idx, arc in enumerate(self.network.trips[trip_index_prime[0]][trip_index_prime[1]])
            if trip_index != trip_index_prime and arc.is_conflicted and arc in self.network.trips[trip_index[0]][trip_index[1]] and
            arc in self.network.trips[trip_index_prime[0]][trip_index_prime[1]]
        )

        # add linearization constraints (3.1k) x[trip_index, arc] -
        self.model.add_constraints(
            s[trip_index, arc] - self.network.travel_times[trip_index[0]][trip_index[1]][idx] - \
            s[trip_index_prime, arc] + EPSILON <= BIG_M_4 * (1-beta[trip_index, trip_index_prime, arc]) # - x[trip_index_prime, arc]
            for trip_index in self.network.trip_indices
            for trip_index_prime in self.network.trip_indices
            for idx, arc in enumerate(self.network.trips[trip_index[0]][trip_index[1]])
            if trip_index != trip_index_prime and arc.is_conflicted and arc in self.network.trips[trip_index[0]][trip_index[1]] and
            arc in self.network.trips[trip_index_prime[0]][trip_index_prime[1]]
        )

        # add linearization constraint (3.1l)
        self.model.add_constraints(
            gamma[trip_index, trip_index_prime, arc] - alpha[trip_index, trip_index_prime, arc] - beta[trip_index, trip_index_prime, arc] >= -1
            for trip_index in self.network.trip_indices
            for trip_index_prime in self.network.trip_indices
            for arc in self.network.conflicting_arcs
            if trip_index != trip_index_prime and arc in self.network.trips[trip_index[0]][trip_index[1]] and
            arc in self.network.trips[trip_index_prime[0]][trip_index_prime[1]]
        )

        # add linearization constraint (3.1m)
        self.model.add_constraints(
            2*gamma[trip_index, trip_index_prime, arc] - alpha[trip_index, trip_index_prime, arc] - beta[trip_index, trip_index_prime, arc] <= 0
            for trip_index in self.network.trip_indices
            for trip_index_prime in self.network.trip_indices
            for arc in self.network.conflicting_arcs
            if trip_index != trip_index_prime and arc in self.network.trips[trip_index[0]][trip_index[1]] and
            arc in self.network.trips[trip_index_prime[0]][trip_index_prime[1]]
        )

        # add linearization constraint (3.1n)
        self.model.add_constraints(
            f[trip_index, arc] == self.model.sum(
                gamma[trip_index, trip_index_prime, arc]
                for trip_index_prime in self.network.trip_indices
                if trip_index != trip_index_prime
            )
            for trip_index in self.network.trip_indices
            for arc in self.network.conflicting_arcs
            if arc in self.network.trips[trip_index[0]][trip_index[1]]
        )

        # add. constraint: consecutive trips of one vehicle work out in a time dimension (not in paper)
        self.model.add_constraints(
            s[(vehicle_index,idx), self.network.trips[vehicle_index][idx][-1]] + \
            self.network.travel_times[vehicle_index][idx][-1] <= \
            s[(vehicle_index,idx+1), self.network.trips[vehicle_index][idx+1][0]]
            for vehicle_index in self.network.vehicle_indices
            for idx in range(len(self.network.trips[vehicle_index])-1)
        )

        # enforce time window to be met at end depot (note constraints 3.1e) are disabled)
        self.model.add_constraints(
            s[(vehicle_index, idx), self.network.trips[vehicle_index][idx][-1]] + \
            self.network.travel_times[vehicle_index][idx][-1] <= self.network.route_time_window[vehicle_index][1]
            for vehicle_index in self.network.vehicle_indices
            for idx in range(len(self.network.trips[vehicle_index])-1, len(self.network.trips[vehicle_index]))
        )

        # enforce time window to be met at start depot (note constraints 3.1e) are disabled)
        self.model.add_constraints(
            s[(vehicle_index, 0), self.network.trips[vehicle_index][0][0]] >= self.network.route_time_window[vehicle_index][0]
            for vehicle_index in self.network.vehicle_indices
        )

        # these constraints enforce that only unconflicted solutions can be feasible
        # f is the number of other trips on an arc (excluding the trip itself that f belongs to)
        self.model.add_constraints(
            f[trip_index, arc] <= 0
            for trip_index in self.network.trip_indices
            for arc in self.network.conflicting_arcs
            if arc in self.network.trips[trip_index[0]][trip_index[1]]
        )

        return s


