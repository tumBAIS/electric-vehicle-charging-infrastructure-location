import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional
from framework.solution_representation import (
    Point,
    VehicleID,
    VertexID,
    ArcID,
    SolutionRepresentation,
    CustomJSONEncoder
)
from collections import defaultdict
import dataclasses
import json

from iterative_local_search.decision_variables import DecisionVariables
from iterative_local_search.subproblem import best_routing


@dataclass
class Conflict:
    id: str
    vertex: VertexID
    arc: Optional[ArcID]
    start_time: int
    end_time: int
    # `trips` stores route id, conflicting point info and previous stop id.
    trips: List[Tuple[VehicleID, Point, VertexID]]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id and self.trips == other.trips and self.start_time == other.start_time and \
               self.end_time == other.end_time
    
    def __repr__(self):
        vehicles = ','.join(map(str, [t[0] for t in self.trips]))
        return f"{self.id}: {vehicles}"

    def __str__(self):
        return self.id
    
    def get_first_pair(self) -> "Conflict":
        """
        Returns the sub-conflict between the first pair of vehicles (trips) contained 
        in this conflict instance.
        @return: Conflict of degree 2
        """
        return Conflict(self.id + '-{}-{}'.format(self.trips[0][0], self.trips[1][0]),
                        vertex=self.vertex,
                        arc=self.arc,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        trips=[self.trips[0], self.trips[1]])

    def break_down_by_first_pair(self, former_idx: int, cut_time: int) -> Tuple[Optional["Conflict"], Optional["Conflict"]]:
        """
        Breaks this conflict of degree k down into two sub-conflicts of degree k-1 in which the conflicting
        time interval is split at the given cut_time. This function builds up a binary tree when being called recursively and 
        should return None when no sub-conflicts can be created (because the current node is of degree k=2.
        @param former_idx (int): 0 if the first vehicle of the pair arrives at the conflicting charger first, 1 if it is the second one
        @cut_time (float): cut time being found in the binary division conflict resolution algorithm
        @return: two sub-conflicts in a tuple, a tuple of two None's when no leaf can be created
        """
        if len(self.trips) == 2:
            return None, None
        else:
            # "/" in the conflict id means conflict excluding the trip
            c_i = Conflict(self.id + '-/{}'.format(self.trips[1-former_idx][0]),
                           vertex=self.vertex,
                           arc=self.arc,
                           start_time=self.start_time,
                           end_time=cut_time,
                           trips=([self.trips[former_idx]] + self.trips[2:]))
            c_j = Conflict(self.id + '-/{}'.format(self.trips[former_idx][0]),
                           vertex=self.vertex,
                           arc=self.arc,
                           start_time=cut_time,
                           end_time=self.end_time,
                           trips=[self.trips[1-former_idx]] + self.trips[2:])
        return c_i, c_j


def dump_conflicts_to_json(conflicts: List[Conflict], path: str = './conflict.json') -> None:
    dump = [dataclasses.asdict(conflict) for conflict in conflicts]
    with open(path, 'w') as f:
        json.dump(dump, f, cls=CustomJSONEncoder, indent=4)


def dump_conflicts_to_pkl(conflicts: List[Conflict], path: str = './conflict.pkl') -> None:
    """
    Dump list of conflicts to a pkl file
    @param conflicts (List[Conflict]): conflicts to be dumped
    @param path (str, optional): path to the target pkl file
    """
    dump_dir = path.split('/')[-2] if len(path.split('/')) > 1 else '.'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    with open(path, 'wb') as f:
        pickle.dump(conflicts, f)


def load_conflicts_from_pkl(path: str = './conflicts.pkl') -> List[Conflict]:
    """
    Load a list of conflicts from a pkl file
    @param path (str, optional): pkl file to load from
    @return: List[Conflict]
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_fine_grained_conflicts(charger_vid: VertexID, trips: List[Tuple[VehicleID, Point, VertexID]]) -> List[Conflict]:
    """
    Finds all the conflicts for one specific charger
    @param charger_vid (VertexID): used to construct conflict instances
    @param trips (List[Tuple[VehicleID, Point, VertexID]]): trips corresponding to this charger
    @return: List[Conflict]: conflicts detected for this charger
    """
    # Create dictionaries to store the start and end times with their respective intervals
    start_times = defaultdict(list)
    end_times = defaultdict(list)

    # Populate the start_times and end_times dictionaries
    for route_name, stop_time, prev_stop in trips:
        # add tuple of route name and time to prevent disambiguition only using route name
        start_times[stop_time.arrival_time_int].append((route_name, stop_time, prev_stop))
        end_times[stop_time.departure_time_int].append((route_name, stop_time, prev_stop))

    # Sort the unique time points in ascending order
    all_times = sorted(set(start_times.keys()) | set(end_times.keys()))

    # Initialize the overlapping_intervals dictionary
    overlapping_intervals = defaultdict(list)

    # Initialize the active_intervals set to keep track of intervals currently active
    active_intervals = set()

    # Iterate through the sorted time points
    for time_point in all_times:

        # Check for intervals that start at this time_point
        if time_point in start_times:
            for interval_id, stop_time, prev_stop in start_times[time_point]:
                active_intervals.add((interval_id, stop_time, prev_stop))

        # Check for intervals that end at this time_point
        # It is important to do this check in end times second, if a line arrives and departs at the same time
        if time_point in end_times:
            for interval_id, stop_time, prev_stop in end_times[time_point]:
                active_intervals.remove((interval_id, stop_time, prev_stop))

        # If there are overlapping intervals at this time point, add them to the overlapping_intervals dictionary
        if len(active_intervals) > 1:
            overlap_start = time_point
            overlap_end = None
            
            # Find the maximum end time of all active intervals
            max_end_time = max(stop_time.departure_time_int for _, stop_time, _ in active_intervals)
            
            # If there's a next time point and it's within the range of the active intervals, use it as the overlap_end
            next_time_idx = all_times.index(time_point) + 1
            if next_time_idx < len(all_times):
                next_time = all_times[next_time_idx]
                if next_time < max_end_time:
                    overlap_end = next_time

            if overlap_end is not None:
                overlapping_intervals[(overlap_start, overlap_end)] = list(active_intervals)

    conflicts = []
    for k, v in overlapping_intervals.items():
        start_time = k[0]
        end_time = k[1]
        # tolerance of 1 sec
        if end_time - start_time <= 1:
            continue
        conflicts.append(
            Conflict(
                "{}-{}-{}".format(charger_vid, int(start_time), int(end_time)),
                vertex=charger_vid,
                arc=None,
                start_time=start_time,
                end_time=end_time,
                trips=v
            )
        )

    # the above logic will not capture completely overlapping conflicts
    pts = {p for v,p,ve in trips}
    if len(pts) != len(trips):
        trips_duplicated = [p for v,p,ve in trips]
        cnt = {}
        for p in pts:
            cnt[p] = trips_duplicated.count(p)
        for point, count in cnt.items():
            if count <= 1:
                continue
            else:
                t_temp = [(v,p,ve) for (v,p,ve) in trips if p==point]
                start_time = point.arrival_time_int
                end_time = point.departure_time_int
                # tolerance of 1 sec
                if end_time - start_time <= 1:
                    continue
                if start_time!=end_time:
                    conflicts.append(
                        Conflict(
                            "{}-{}-{}".format(charger_vid, int(start_time), int(end_time)),
                            vertex=charger_vid,
                            arc=None,
                            start_time=start_time,
                            end_time=end_time,
                            trips=t_temp,
                        )
                    )

    return conflicts

def get_charger_schedules(solution: SolutionRepresentation) -> dict[VertexID, List[Tuple[VehicleID, Point, VertexID]]]:
    """
    Find all the time slots and corresponding vehicle when a charger is occupied
    @param solution (SolutionRepresentation): solution representation instance
    @return: dict[VertexID, List[Tuple[VehicleID, Point]]]: a dictionary indicating charger vertex and its
             corresponding occupying vehicles (and also the scheduled stop times of the vehicles)
    """
    charger_schedules = {vertex_id: [] for vertex_id in solution.static_invest.keys()}
    for charger_vid in charger_schedules:
        for itinerary in solution.itineraries:
            prev_stop = None
            for point in itinerary.route:
                if point.id == charger_vid and point.is_static_charger:
                    charger_schedules[charger_vid].append((itinerary.vehicle, point, prev_stop))
                prev_stop = point.id
        charger_schedules[charger_vid] = sorted(charger_schedules[charger_vid], key=lambda x: x[1].arrival_time_int)

    return charger_schedules


def detect_conflict_from_solution(solution: SolutionRepresentation) -> List[Conflict]:
    """
    Detect all conflicts from solution
    @param solution (SolutionRepresentation): solution representation instance
    @return List[Conflict]: a list of conflicts
    """
    charger_schedules = get_charger_schedules(solution)
    conflicts = {vertex_id: [] for vertex_id in solution.static_invest.keys()}

    for charger_vid in conflicts:
        conflicts[charger_vid] = find_fine_grained_conflicts(charger_vid, charger_schedules[charger_vid])

    all_conflicts = sum((v for v in conflicts.values()), [])
    return all_conflicts


def is_conflict_free(
        decision: DecisionVariables, time_step: int, solution: Optional[SolutionRepresentation],
) -> Tuple[bool, SolutionRepresentation]:
    """
    Detect and resolve all the conflicts given current decision variable (the class instance).
    Returns a solution that is conflict-free.
    @param decision: instance of class 'DecisionVariables'
    @param time_step: time step of the SPPRC network
    @param solution: SolutionRepresentation including the route information
    @param interpreter: String indicating if 'cpp' or 'python'
    @param multithreading: Boolean indicating if parallel execution wanted
    @return: Boolean indicating if solution is conflict free (True) or not (False)
    """
    # Detect conflict from best routing determined by current decision variable
    if solution is None:
        solution = best_routing(
            decision,
            time_step=time_step,
        )
    conflicts = detect_conflict_from_solution(solution)
    if len(conflicts) == 0:
        return (True, solution)
    return (False, solution)
