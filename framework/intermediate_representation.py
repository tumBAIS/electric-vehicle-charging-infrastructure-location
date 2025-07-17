# coding=utf-8
import copy
import json
import math
import os
import warnings
from collections import defaultdict
import numpy as np
import random
import folium

import osmnx as osm

import framework.utils as util

from functools import cached_property
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union
from pydantic import field_validator, model_validator
from dataclasses import dataclass, asdict
from dataclasses import field
from itertools import pairwise, count
from typing import Iterable, NewType, Optional, Tuple, List, Union, Dict, Set

import networkx as nx

from framework.utils import ValidationError

random.seed(42)
np.random.seed(42)

VehicleID = NewType("VehicleID", str)
VertexID = NewType("VertexID", str)
ChargerID = NewType("ChargerID", str)
TransformerID = NewType("TransformerID", str)
ArcID = Tuple[VertexID, VertexID]
VertexIdentifier = VertexID | "Vertex"

Km = float
KmPerH = float
KwhPerKm = float
Seconds = float
KwhPerH = float
Kwh = float
CostPerMeter = float
Meter = float
SecondsOfDay = float


class UnequalChargerError(BaseException):
    pass


class DynamicSegmentConsolidationError(BaseException):
    pass


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: VehicleID
    max_speed: KmPerH
    consumption: KwhPerKm


@dataclass(frozen=False, eq=False)
class Charger:
    id: ChargerID
    segment_construction_cost: CostPerMeter
    transformer_construction_cost: float
    charging_rate: KwhPerH

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass
class Vertex:
    id: VertexID
    is_depot: bool
    is_stop: bool
    coordinate: util.Coordinate
    constructible_charger: set[Charger] = field(default_factory=set)
    name: Optional[str] = None
    _dummy_of: Optional[VertexID] = None

    def __deepcopy__(self, memodict=None) -> "Vertex":
        if memodict is None:
            memodict = {}

        # Do not create copies of chargers
        memodict.update(
            {id(charger): charger for charger in self.constructible_charger}
        )

        return copy.deepcopy(self, memodict)

    def asdict(self):
        return {
            "id": self.id,
            "is_depot": self.is_depot,
            "is_stop": self.is_stop,
            "coordinate": {"lat": self.coordinate.lat, "lon": self.coordinate.lon},
            "constructible_charger": [asdict(c) for c in self.constructible_charger],
            "name": self.name if self.name is not None else ""
        }

    @field_validator("constructible_charger", mode="after")
    @classmethod
    def only_one_constructible_charger(cls, constructible_charger: set[Charger]):
        if len(constructible_charger) > 1:
            raise ValueError(f"By design only one constructible charger per Arc in IR")
        return constructible_charger

    def __update__(self, state):
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'Arc' object has no attribute '{key}'")

    @property
    def can_construct_charger(self) -> bool:
        return len(self.constructible_charger) > 0

    @property
    def is_dummy(self) -> bool:
        return self._dummy_of is not None and self.id != self._dummy_of

    @property
    def is_stop_or_depot(self) -> bool:
        return self.is_stop or self.is_depot

    @property
    def only_stop_or_depot(self) -> bool:
        return self.is_stop_or_depot and not self.can_construct_charger

    @property
    def is_auxiliary(self) -> bool:
        return not (self.is_stop_or_depot or self.can_construct_charger)

    @property
    def is_standalone_charger(self) -> bool:
        return self.can_construct_charger and not self.is_stop_or_depot

    @property
    def original_vertex_id(self) -> VertexID:
        return self._dummy_of if self._dummy_of is not None else self.id

    def __str__(self):
        prefix = ""
        if self.is_depot:
            prefix += "D"
        elif self.is_stop:
            prefix += "V"
        elif self.can_construct_charger:
            prefix += "S"
        else:
            prefix += "A"
        if self.name is not None:
            return f"{prefix}{self.id} ({self.name})"
        return f"{prefix}{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Arc:
    distance: Km
    # Speed limit in kmh
    speed_limit: KmPerH
    # Chargers constructible on this arc. Chargers may be constructed at arbitrary points and length on the arc.
    constructible_chargers: set[Charger]
    # if consumption directly given, this overwrites the consumption based on vehicle consumption * distance
    absolute_consumption_overwrite: Optional[Union[Kwh, str]]=None
    speed_overwrite_kmh: Optional[Union[float, str]] = None

    def __post_init__(self):
        self.distance = round(self.distance, 6)

    def asdict(self):
        return {
            "distance": self.distance,
            "speed_limit": self.speed_limit,
            "constructible_charger": [asdict(c) for c in self.constructible_chargers],
            "consumption_overwrite": self.absolute_consumption_overwrite if self.absolute_consumption_overwrite is not None else "",
            "travel_time_overwrite": self.speed_overwrite_kmh if self.speed_overwrite_kmh is not None else "",
        }

    @field_validator("speed_limit", mode="after")
    @classmethod
    def positive_speed_limit(cls, speed_limit: float):
        if speed_limit <= 0.0:
            raise ValueError(f"Speed limit is <= 0.0")
        return speed_limit

    @field_validator("distance", mode="after")
    @classmethod
    def positive_distance(cls, distance: float):
        if distance < 0.0:
            raise ValueError(f"Distance is < 0.0")
        return distance

    @field_validator("constructible_chargers", mode="after")
    @classmethod
    def only_one_constructible_charger(cls, constructible_chargers: set[Charger]):
        if len(constructible_chargers) > 1:
            raise ValueError(f"By design only one constructible charger per Arc in IR")
        return constructible_chargers

    @property
    def distance_in_meters(self) -> Meter:
        return self.distance * 1000.0

    @property
    def can_construct_charger(self) -> bool:
        return len(self.constructible_chargers) > 0

    def __deepcopy__(self, memodict=None) -> "Arc":
        if memodict is None:
            memodict = {}
        # Do not create copies of chargers
        memodict.update(
            {id(charger): charger for charger in self.constructible_chargers}
        )
        return copy.deepcopy(self, memodict)

    def __add__(self, other: "Arc") -> "Arc":
        """
        Concatenates two arcs. Adds the distance and computes a weighted average of the speed according to the distance.
        Throws an error if they have different sets of constructible chargers and both arcs are 'real'
        (i.e., distance > 0).
        :param other: 'Arc'
        :return: 'Arc'
        """
        if self.constructible_chargers != other.constructible_chargers:
            raise UnequalChargerError("Cannot add arcs with unequal constructible chargers")
        if len(self.constructible_chargers)>0 or len(other.constructible_chargers)>0:
            raise DynamicSegmentConsolidationError("Cannot contract dynamic charging segments")

        total_distance = self.distance + other.distance
        if total_distance != 0:
            avg_speed = (self.distance / total_distance) * self.speed_limit + (
                    other.distance / total_distance) * other.speed_limit
        else:
            avg_speed = self.speed_limit
        return Arc(distance=self.distance + other.distance, speed_limit=avg_speed,
                   constructible_chargers=self.constructible_chargers.copy())

    def get_travel_time_seconds(self, speed: KmPerH) -> Seconds:
        if self.speed_overwrite_kmh is not None:
            warnings.warn(f"Overwrote speed to {self.speed_overwrite_kmh}")
            speed = min(self.speed_limit, self.speed_overwrite_kmh)
        else:
            speed = min(self.speed_limit, speed)
        travel_time_hours = self.distance / speed
        return travel_time_hours*3600

    def get_travel_time_hours(self, speed: KmPerH) -> float:
        if self.speed_overwrite_kmh is not None:
            warnings.warn(f"Overwrote speed to {self.speed_overwrite_kmh}")
            speed = min(self.speed_limit, self.speed_overwrite_kmh)
        else:
            speed = min(self.speed_limit, speed)
        travel_time_hours = self.distance / speed
        return travel_time_hours

    def get_consumption(self, consumption: KwhPerKm) -> Kwh:
        """
        Leave option to override consumption in dumped instance (JSON) and use the hard coded consumption
        instead of the consumption linearly correlated with distance
        """
        if self.absolute_consumption_overwrite is not None:
            warnings.warn(f"Overwrote consumption from {consumption*self.distance} to {self.absolute_consumption_overwrite} KwhPerKm")
        return self.absolute_consumption_overwrite if self.absolute_consumption_overwrite is not None else consumption * self.distance

    def recharge_amount(self, speed: KmPerH) -> Kwh:
        """Return recharged kwh for a given speed"""
        return self.get_travel_time_hours(speed)*next(iter(self.constructible_chargers)).charging_rate


@dataclass
class Stop:
    vertex_id: VertexID
    stopover_time: float
    earliest_time_of_service: SecondsOfDay
    latest_time_of_service: SecondsOfDay

    @model_validator(mode="after")
    def time_window_valid(cls, values):
        if values.earliest_time_of_service > values.latest_time_of_service:
            raise ValueError("Time window ends before it begins")
        return values

    @field_validator('stopover_time', mode="after")
    def positive_stopover_time(cls, stopover_time):
        if stopover_time < 0:
            raise ValueError("Stopover time cannot be negative!")
        return stopover_time

    @field_validator('latest_time_of_service', mode="after")
    def positive_latest_time_of_service(cls, latest_time_of_service):
        if latest_time_of_service < 0:
            raise ValueError("Latest time of service must be positive")
        return latest_time_of_service

    @field_validator('earliest_time_of_service', mode="after")
    def positive_earliest_time_of_service(cls, earliest_time_of_service):
        if earliest_time_of_service < 0:
            raise ValueError("Earliest time of service must be positive")
        return earliest_time_of_service

    def __hash__(self):
        return hash((self.vertex_id, self.earliest_time_of_service, self.latest_time_of_service))


@dataclass
class Route:
    """
    Sequence of stops. Start and ends at a depot
    """

    stop_sequence: list[Stop]
    vehicle_id: VehicleID
    _cache = Dict[VertexID, List[int]]

    def __init__(self, stop_sequence: list[Stop], vehicle_id: VehicleID):
        self.stop_sequence = stop_sequence
        self.vehicle_id = vehicle_id
        self._cache = {}
        for index, stop in enumerate(self.stop_sequence):
            if stop.vertex_id not in self._cache:
                self._cache[stop.vertex_id] = []
            self._cache[stop.vertex_id].append(index)

    def asdict(self):
        return {
            "stop_sequence": [asdict(s) for s in self.stop_sequence],
            "vehicle_id": self.vehicle_id,
        }

    @field_validator("stop_sequence", mode="after")
    def stops_not_empty(cls, stop_sequence):
        if len(stop_sequence) <= 2:
            raise ValueError("Route has less than two stops")
        return stop_sequence

    @field_validator("stop_sequence", mode="after")
    def start_and_ends_at_same_vertex(cls, stop_sequence):
        if stop_sequence[0].vertex_id != stop_sequence[-1].vertex_id:
            raise ValueError("Route does not start and end at the same vertex")
        return stop_sequence

    @property
    def depot(self) -> Stop:
        return self.stop_sequence[0]

    @property
    def latest_arrival_time(self) -> float:
        return self.stop_sequence[-1].latest_time_of_service

    @property
    def earliest_departure_time(self) -> float:
        return self.depot.earliest_time_of_service

    @property
    def set_stop_ids(self):
        set_of_stop_vertex_ids = set()
        for stop in self.stop_sequence:
            set_of_stop_vertex_ids.add(stop.vertex_id)
        return set_of_stop_vertex_ids

    @property
    def is_cyclic(self):
        return len(self.set_stop_ids) != len([s.vertex_id for s in self.stop_sequence])

    def __len__(self) -> int:
        return len(self.stop_sequence)

    def __getitem__(self, item) -> Stop:
        return self.stop_sequence[item]

    def get_stop(self, vertex: VertexID, occurance: int=1):
        """Return occurance (e.g., first or second) of stop with given Vertex ID"""
        # Get the list of indices for the given vertex
        indices = self._cache.get(vertex, [])

        # Assert the occurrence exists
        if occurance > len(indices):
            raise AssertionError(f"The {occurance}th occurrence of id '{vertex}' does not exist.")

        # Return the stop at the correct occurrence
        return self.stop_sequence[indices[occurance - 1]]


def _create_nx_from_vertices_and_arc_matrix(vertices: Iterable[Vertex], matrix: dict[ArcID, Arc]) -> nx.DiGraph:
    g = nx.DiGraph()
    vertices = sorted(vertices, key=lambda v: v.id)
    matrix = dict(sorted(matrix.items()))
    for v in vertices:
        assert type(v)==Vertex
        g.add_node(v.id, vertex=v)
    for (u, v), arc in matrix.items():
        assert type(arc)==Arc
        assert u in g, f"{u}"
        assert v in g, f"{v}"
        g.add_edge(u, v, arc=copy.copy(arc))

    return g


def _route_respects_time_windows(stop_sequence: List[Stop], minimum_travel_times: dict[ArcID, float]) -> bool:
    departure_at_v = stop_sequence[0].earliest_time_of_service + stop_sequence[0].stopover_time
    for u, v in pairwise(stop_sequence):
        departure_at_v = max(
            v.earliest_time_of_service,
            departure_at_v + minimum_travel_times[(u.vertex_id, v.vertex_id)] + u.stopover_time
        )
        if departure_at_v <= v.latest_time_of_service:
            continue
        else:
            return False
    return True


class IntermediateRepresentation:
    """
    Representation of network of objects if class Vertex and object of class Arc
    Concretely, the intermediate representation of a network corresponds to an "idealized" representation of the source
    network (e.g. pulled from OSM). Here, "idealized" refers to the network not containing any superfluous attributes/nodes.
    This representation aims to provide a basis that specialized solution procedures can build on, i.e., construct their
    specific network representations.
    More concretely, the intermediate representation contains:
    - Stops
    - Charging stations
    - The depot

    Does not contain duplicates even if attributes allow this.

    graph TD
    A[Source 1: OSMNX] --> X[Intermediate Representation];
    B[Source 2: GMaps] --> X[Intermediate Representation];
    C[Source 3: Hand constructed] --> X[Intermediate Representation];
    X --> D[MIP];
    X --> E[Metaheuristic];
    X --> F[...];

    K1[Preprocessor] <--> X;
    K2[Validator] <--> X;

    """

    def __init__(
            self,
            vertices: Optional[Iterable[Vertex]],
            arcs: Optional[dict[ArcID, Arc]],
            routes: Iterable[Route],
            network: Optional[nx.DiGraph]=None,
    ):
        # Create network from vertices and arcs
        assert (vertices and arcs) or network, f"Either vertices and arcs must be given, or complete network"
        self._network = _create_nx_from_vertices_and_arc_matrix(vertices, arcs) if (vertices and arcs) else network
        self._routes = list(routes)
        self._validate()
        self._cached_shortest_paths = {}

    def _validate_only_inductive_charging(self):
        for v in self.charger_nodes:
            if len(v.constructible_charger) > 1:
                raise ValidationError(f"Only one stationary charger per vertex")
        for u,v,arc in self.charger_edges:
            if len(arc.constructible_chargers)>1:
                print(u,v)
                print(arc.constructible_chargers)
                raise ValidationError(f"Only one dynamic charger per arc (by design)")
    def _validate_only_one_depot(self):
        if len({v.coordinate for v in self.vertices if v.is_depot}) > 1:
            raise ValidationError(f"There is depot locations with multiple coordinates")

    def _validate_routes(self):
        for route in self._routes:
            if route.depot.vertex_id != self.depot.id:
                raise ValueError(f'Route {route} does not start/end at depot {self.depot}')
            for stop in route.stop_sequence:
                if not (self.get_vertex(stop.vertex_id).is_stop or self.get_vertex(stop.vertex_id).is_depot):
                    raise ValidationError(f"{stop.vertex_id} is part of route {route} but not labelled as stop/depot")

    def _validate_stops_occur_in_network(self):
        # 1. Check each vertex in routes is in network
        for route in self._routes:
            for s in route.stop_sequence:
                if s.vertex_id not in self._network:
                    raise ValueError(f'Stop {s} does not occur in network')

    def _validate_stops_connected(self):
        for route in self._routes:
            for u, v in pairwise(route.stop_sequence):
                if not nx.has_path(self._network, u.vertex_id, v.vertex_id):
                    raise ValueError(f'Stop {u} is not connected to stop {v}')

    def _validate_segment_start_and_end_no_stops(self):
        for u, v, _ in self.charger_edges:
            if self.get_vertex(u).is_stop:
                raise ValidationError(f"{self.get_vertex(u).coordinate}, {_.constructible_chargers} is stop and start of arc charger")
            if self.get_vertex(v).is_stop:
                raise ValidationError(f"{self.get_vertex(v)} is stop and end of arc charger")
            if self.get_vertex(u).can_construct_charger:
                raise ValidationError(f"Start and end points of segments can not construct stationary charger")
            if self.get_vertex(v).can_construct_charger:
                raise ValidationError(f"Start and end points of segments can not construct stationary charger")

    def _validate_stops_reachable_within_time_windows(self):
        for route in self._routes:
            minimum_travel_times = {
                (u.vertex_id, v.vertex_id): nx.shortest_path_length(
                    self._network, u.vertex_id, v.vertex_id, lambda i, j, edge_data: edge_data['arc'].get_travel_time_seconds(edge_data['arc'].speed_limit)
                )
                for u, v in pairwise(route)}
            if _route_respects_time_windows(route.stop_sequence, minimum_travel_times):
                continue
            raise ValueError(f'Route {route.vehicle_id} violates time windows')
        return True

    def _no_arcs_with_zero_distance(self):
        for u,v,arc in self.arcs:
            assert arc.distance > 0, f"{u}, {v}"

    def _dyn_chargers_connected(self):
        for c, el in self.charger_edges_by_charger.items():
            el_nodes = {element for tup in el for element in tup}
            subgraph = nx.subgraph_view(
                self._network.copy(), filter_node=lambda u: u in el_nodes
            )
            subgraph = subgraph.copy()
            assert nx.is_weakly_connected(subgraph)

    def _validate(self):
        """Assert validity of constructed intermediate representation"""
        self._validate_only_one_depot()
        self._validate_routes()
        self._validate_stops_occur_in_network()
        self._validate_stops_connected()
        self._validate_stops_reachable_within_time_windows()
        self._validate_segment_start_and_end_no_stops()
        #self._no_arcs_with_zero_distance()
        self._dyn_chargers_connected()
        self._validate_only_inductive_charging()

    def _get_vertex_id(self, v: VertexIdentifier) -> VertexID:
        return v.id if type(v)==Vertex else v

    def get_vertex(self, v_id: VertexID) -> Vertex:
        return self._network.nodes(data=True)[v_id]['vertex']

    def __eq__(self, other: "IntermediateRepresentation"):
        """Equality between two objects is defined as equality of vertices and arcs"""

        node_eq = nx.utils.nodes_equal(self._network.nodes, other._network.nodes)
        edges_eq = nx.utils.edges_equal(self._network.edges, other._network.edges)

        return node_eq and edges_eq and self._routes == other._routes

    def __contains__(self, item: VertexIdentifier):
        return self._get_vertex_id(item) in self._network

    def __add__(self, other: "IntermediateRepresentation") -> "IntermediateRepresentation":
        """Adding multiple instances of 'IntermediateRepresentation'"""
        vertices = {*self.vertices, *other.vertices}
        arcs = {(u, v): arc for u,v,arc in self.arcs + other.arcs}
        routes = [*self.routes, *other.routes]
        return IntermediateRepresentation(vertices=vertices, arcs=arcs, routes=routes)

    def invalidate_caches(self):
        """Invalidate all cached_property attributes."""
        # for some reason this needs to be done explicitly
        self._invalidate_vertex_cache()
        self._invalidate_arc_cache()
        try:
            del self.list_of_all_potential_chargers
        except AttributeError:
            pass
        try:
            del self.list_of_all_potential_chargers_dict
        except AttributeError:
            pass
        try:
            del self.min_cost_per_charger
        except AttributeError:
            pass
        try:
            del self.stop_charger_connections
        except AttributeError:
            pass

    def _invalidate_vertex_cache(self):
        try:
            del self.charger_nodes
        except AttributeError:
            pass
        try:
            del self.vertices
        except AttributeError:
            pass
        try:
            del self.charger_station_list
        except AttributeError:
            pass
        try:
            del self.static_chargers
        except AttributeError:
            pass

    def _invalidate_arc_cache(self):
        try:
            del self.charger_edges
        except AttributeError:
            pass
        try:
            del self.arcs
        except AttributeError:
            pass
        try:
            del self.charger_edges_by_charger
        except AttributeError:
            pass
        try:
            del self.dyn_chargers
        except AttributeError:
            pass

    def _invalidate_spp(self):
        self._cached_shortest_paths = {}

    def update_routes(self):
        self._routes = self.routes

    def revalidate(self):
        self._validate()

    def limit_distance_precision(self, precision: int):
        # we limit precision to 6 digits to make sure that we can always convert to integer for accurate comparisons
        for _,_,arc in self.arcs:
            arc.distance = round(arc.distance,precision)
        self._invalidate_arc_cache()
        self._invalidate_spp()

    @cached_property
    def vertices(self) -> List[Vertex]:
        return list(x[1] for x in self._network.nodes.data('vertex'))

    @cached_property
    def arcs(self) -> List[Tuple[VertexID, VertexID, Arc]]:
        return [
            (u, v, arc)
            for u, v, arc in self._network.edges(data="arc")
        ]

    @cached_property
    def depot(self) -> Vertex:
        return next(x for x in self.vertices if x.is_depot)

    @cached_property
    def charger_nodes(self) -> List[Vertex]:
        return [x for x in self.vertices if x.can_construct_charger]

    @cached_property
    def charger_edges(self) -> List[Tuple[VertexID, VertexID, Arc]]:
        return [(u, v, arc) for u, v, arc in self.arcs if arc.can_construct_charger]

    @cached_property
    def charger_edges_by_charger(self) -> Dict[Charger, List[Tuple[VertexID, VertexID]]]:
        "Sorted - e.g., {charger: [(3,4),(4,2),(2,6)]"
        res = {}
        for u,v,a in self.charger_edges:
            c = next(iter(a.constructible_chargers))
            if c in res:
                res[c].append((u,v))
            else:
                res[c] = [(u,v)]
        for c in res:
            res[c] = util.sort_tuples(res[c])
        return res

    @cached_property
    def min_cost_per_charger(self) -> Dict[ChargerID, float]:
        """Return a dictionary mapping between charger IDs and the minimum cost a charger induces in a solution"""
        def _get_min_segment_length(l):
            min = math.inf
            for t in l:
                temp_length = self.get_arc(t[0], t[1]).distance_in_meters
                if temp_length < min:
                    min = temp_length
            return min

        res = {}
        for c, segments in self.charger_edges_by_charger.items():
            res[c.id] = _get_min_segment_length(segments)*c.segment_construction_cost + c.transformer_construction_cost
        return res

    @cached_property
    def max_num_chargers(self) -> int:
        """Yields the number of charger objects, i.e., a count of chargers"""
        return len(self.min_cost_per_charger.keys())

    @cached_property
    def stop_charger_connections(self) -> Dict[VertexID, Dict[ChargerID, bool]]:
        """Calculates a mapping between Vertices representing stops / depots and charger ids while carrying the info
        if they are connected in given representation (i.e., if any segment start node is directly connected)"""
        res = {}
        for vertex in self.vertices:
            if not (vertex.is_stop or vertex.is_depot):
                continue
            else:
                res[vertex.id] = {}
                for c, l in self.charger_edges_by_charger.items():
                    res[vertex.id][c.id]=False
                    for arc in l:
                        if self.has_arc(vertex.id, arc[0]):
                            res[vertex.id][c.id] = True
                            break
        return res

    @cached_property
    def routes(self) -> List[Route]:
        """
        Returns a deep copy of the list of routes. Prevents modification to the routes.
        :return:
        """
        return copy.deepcopy(self._routes)

    @cached_property
    def list_of_all_potential_chargers(self) -> list[tuple[VertexID, VertexID, Charger]]:
        locations = []
        for v in self.vertices:
            if v.can_construct_charger and not v.is_dummy:
                for charger in v.constructible_charger:
                    locations += [(v.id, v.id, charger)]
        for u, v, arc in self.arcs:
            if arc.can_construct_charger and not (self.get_vertex(u).is_dummy and self.get_vertex(v).is_dummy):
                for charger in arc.constructible_chargers:
                    locations += [(u, v, charger)]
        return sorted(locations, key=lambda tuple: (tuple[0], tuple[1]))

    @cached_property
    def list_of_all_potential_chargers_dict(self) -> dict[tuple[VertexID, VertexID], list[Charger]]:
        locations = defaultdict(list)
        for element in self.list_of_all_potential_chargers:
            locations[(element[0], element[1])] += [element[2]]
        return locations

    @cached_property
    def number_of_stops(self) -> int:
        stop_counter = 0
        for route in self.routes:
            stop_counter += len(route) - 2
        return stop_counter

    @property
    def static_chargers(self) -> Set[Charger]:
        return {c for k1, k2, c in self.list_of_all_potential_chargers if k1 == k2}

    @property
    def dyn_chargers(self) -> Set[Charger]:
        return {c for k1, k2, c in self.list_of_all_potential_chargers if k1 != k2}

    def update_static_charger(self, new_cost: float, charging_rate: float)->None:
        """Update cost structure (no invalidation of cache needed as attribute saved directly on object)"""
        self.invalidate_caches()
        for vertex in self.vertices:
            if vertex.can_construct_charger:
                for c in vertex.constructible_charger:
                    c.transformer_construction_cost=new_cost
                    c.charging_rate = charging_rate
        self.invalidate_caches()
        return None

    def update_dynamic_chargers(self, new_fix: float, new_variable_per_meter: float, charging_rate: float) -> None:
        """Update cost structure (no invalidation of cache needed as attribute saved directly on object)"""
        self.invalidate_caches()
        for u,v,arc in self.arcs:
            if arc.can_construct_charger:
                for c in arc.constructible_chargers:
                    c.transformer_construction_cost=new_fix
                    c.segment_construction_cost=new_variable_per_meter
                    c.charging_rate = charging_rate
        self.invalidate_caches()
        return None

    def print_descriptive_stats(self):
        num_static_stops=0
        num_static_not_stops=0
        num_dynamic_segments=0
        total_len_dyn=0
        total_variable_cost=0
        total_transformer_cost_static=0
        total_transformer_cost_dynamic=0
        for k1,k2,c in self.list_of_all_potential_chargers:
            if k1==k2:
                v = self.get_vertex(k1)
                assert not v.is_dummy
                if v.is_stop or v.is_depot:
                    num_static_stops+=1
                else:
                    num_static_not_stops+=1
                total_transformer_cost_static+=c.transformer_construction_cost
            else:
                a = self.get_arc(k1,k2)
                num_dynamic_segments += 1
                total_len_dyn+=a.distance
                total_variable_cost+=a.distance_in_meters*c.segment_construction_cost
                total_transformer_cost_dynamic+=c.transformer_construction_cost
        print(
            f"Number of Static Chargers at Stops: {num_static_stops}\n"
            f"Number of Static Stand-alone Chargers: {num_static_not_stops}\n"
            f"Number of Dynamic Charging Segments: {num_dynamic_segments}\n"
            f"Number of Dynamic Chargers: {len(self.dyn_chargers)}\n"
            f"Total Length of Dynamic Chargers [km]: {total_len_dyn}\n"
            f"Total Variable Cost for all Dynamic Chargers: {total_variable_cost}\n"
            f"Total Fix Cost for all Static Chargers: {total_transformer_cost_static}\n"
            f"Total Fix Cost for all Dynamic Chargers: {total_transformer_cost_dynamic}"
        )

    def get_route(self, vehicle_id: VehicleID) -> Route:
        return next(r for r in self.routes if r.vehicle_id==vehicle_id)

    def get_arc(self, origin: VertexIdentifier, target: VertexIdentifier) -> Optional[Arc]:
        try:
            return self._network.get_edge_data(origin, target)["arc"]
        except TypeError:
            return None

    def has_arc(self, origin: VertexIdentifier, target: VertexIdentifier) -> bool:
        return self._network.has_edge(origin, target)

    def get_neighbors(self, of_vertex: VertexIdentifier) -> List[Vertex]:
        return list(self._network.neighbors(self._get_vertex_id(of_vertex)))

    def get_predecessors(self, of_vertex: VertexIdentifier) -> List[Vertex]:
        return list(self._network.predecessors(self._get_vertex_id(of_vertex)))

    def get_successors(self, of_vertex: VertexIdentifier) -> List[Vertex]:
        return list(self._network.successors(self._get_vertex_id(of_vertex)))

    def get_incoming_arcs(self, of_vertex: VertexIdentifier) -> List[Tuple[VertexID, VertexID, Arc]]:
        if not isinstance(of_vertex, str):
            of_vertex = of_vertex.id
        return [(u, v, data) for u, v, data in self._network.in_edges(of_vertex, data='arc')]

    def get_outgoing_arcs(self, of_vertex: VertexIdentifier) -> List[Tuple[VertexID, VertexID, Arc]]:
        if not isinstance(of_vertex, str):
            of_vertex = of_vertex.id
        return [(u, v, data) for u, v, data in self._network.out_edges(of_vertex, data='arc')]

    def get_adjacent_arcs(
            self, of_vertex: VertexIdentifier
    ) -> List[Tuple[VertexID, VertexID, Arc]]:
        return self.get_incoming_arcs(of_vertex) + self.get_outgoing_arcs(of_vertex)

    def get_weakly_connected_nodes(self, of_vertex: VertexIdentifier) -> List[VertexID]:
        for component in nx.weakly_connected_components(self._network):
            if of_vertex in component:
                return component
        raise ValueError

    def add_vertex(self, v: Vertex) -> None:
        """Add given object of type vertex to intermediate representation
        :param v: Vertex to add to interm. repr.
        """
        if v.is_depot:
            raise ValueError("Cannot add another depot")
        # In future implementation information about the vertices would be required here
        self._network.add_node(v)
        # Reset cache
        self._invalidate_vertex_cache()

    def _remove_vertex(self, v: Vertex) -> None:
        """Remove given object from interm. repr.
        :param vertex: Vertex to remove from interm. repr.
        """
        if v.is_depot:
            raise ValueError("Cannot remove the depot")
        self._network.remove_node(v.id)

    def remove_vertices(self, vertices: Iterable[Vertex]) -> None:
        """Remove all vertices from passed iterable and invalidate cache afterwards"""
        for v in vertices:
            self._remove_vertex(v)
        self._invalidate_vertex_cache()
        self._invalidate_arc_cache()

    def add_arc(
            self, origin: VertexIdentifier, target: VertexIdentifier, a: Arc
    ) -> None:
        """Add object of type arc to interm. repr.
        :param a: Object of type Arc to add to interm. repr.
        """
        self._network.add_edge(self._get_vertex_id(origin), self._get_vertex_id(target), **a.attr_data)
        # Reset cache
        self._invalidate_arc_cache()

    def remove_arc(self, origin: VertexIdentifier, target: VertexIdentifier) -> None:
        """Remove arc from the interm. repr.
        :param a: Object of type Arc to add to interm. repr.
        """
        self._network.remove_edge(self._get_vertex_id(origin), self._get_vertex_id(target))
        # Reset cache
        self._invalidate_arc_cache()

    def spp_via_charger(self, a: VertexIdentifier, b: VertexIdentifier) -> Tuple[bool, List[VertexID]]:
        """Determine if any charger on spp"""
        spp = self.calc_shortest_path(a, b)
        if len(spp)<=2:
            return False, spp

        if len(spp) == 3 and self.get_vertex(spp[1]).can_construct_charger:
            return True, spp

        # stop nodes cannot be segment nodes
        for u,v in pairwise(spp[1:-1]):
            if self.get_arc(u,v).can_construct_charger:
                return True, spp
            elif self.get_vertex(u).can_construct_charger:
                return True, spp
            elif self.get_vertex(v).can_construct_charger:
                return True, spp
        return False, spp

    # Should this always use distance?
    def calc_shortest_path(
            self, origin: VertexIdentifier, target: VertexIdentifier
    ) -> List[VertexID]:
        """
        Calculate shortest path between origin and target vertex weighted by cost.
        :param origin: Origin vertex
        :param target: Target vertex
        :return: List of vertices in shortest path
        """
        origin = self._get_vertex_id(origin)
        target = self._get_vertex_id(target)
        try:
            return self._cached_shortest_paths[(origin, target)][0]
        except KeyError:
            # Path and distance not yet computed. Compute it, return it and add it to the dict
            self.calculate_and_cache_shortest_path(origin, target)
            return self._cached_shortest_paths[(origin, target)][0]

    def calculate_and_cache_shortest_path(self, origin: VertexID, target: VertexID) ->None:
        """This function tries to efficiently calculate both the path and the length in order to store it for future
        calls"""

        # make sure that 6 digits can always be accurately represented in accumulated distance
        # see limited floating point number precision -> representation as int

        def weight_func(u, v, data): return int(data['arc'].distance*1e6)
        path_generator = nx.all_shortest_paths(
            G=self._network,
            source=self._get_vertex_id(origin),
            target=self._get_vertex_id(target),
            weight=weight_func,
            method="dijkstra",
        )

        # we want the shortest path with the most vertices on it --> this contains potentially charging stations
        path: list[int] = next(path_generator)
        num_nodes = len(path)
        while True:
            try:
                p: list[int] = next(path_generator)
                if len(p) > num_nodes:
                    path = p
            except StopIteration:
                break

        # read accumulated weight and finally cache path for later iterations (keep in mind to reconvert to float)
        length = path_weight(self._network, path)
        self._cached_shortest_paths[(origin, target)] = (path, length)
        return None

    def calc_shortest_path_length(self, origin: VertexIdentifier, target: VertexIdentifier) -> float:
        """
        Calculate shortest path between origin and target vertex weighted by cost. If path length was
        already calculated, then reuse this result.
        :param origin: Origin vertex
        :param target: Target vertex
        :return: Lenght of shortest path in cost
        """
        origin = self._get_vertex_id(origin)
        target = self._get_vertex_id(target)
        try:
            # Try to get a cached distance between the two stops
            return self._cached_shortest_paths[(origin, target)][1]
        except KeyError:
            # Path and distance not yet computed. Compute it, return it and add it to the dict
            self.calculate_and_cache_shortest_path(origin, target)
            return self._cached_shortest_paths[(origin, target)][1]

    @cached_property
    def charger_station_list(self) -> List[Tuple[VertexID, VertexID, Charger]]:
        chargers = []
        for key1, key2, c in self.list_of_all_potential_chargers:
            if key1==key2:
                chargers.append((key1, key2, c))
            else:
                chargers.append(
                    (self.charger_edges_by_charger[c][0][0], self.charger_edges_by_charger[c][0][1], c)
                )
        return chargers

    @cached_property
    def distance_dictionary(self) -> dict[tuple[VertexID, VertexID, Charger], dict[VehicleID, int]]:
        """Returns a mapping between charger (i.e., key 1, key 2, charger object) and a dictionary mapping vehicle ids
        to sum of distances between stops on route and charger vertex (in case of dyn segments: segment end)"""
        d = {}

        for charger in self.charger_station_list:
            d_r = {}
            for r in self.routes:
                d_r[r.vehicle_id] = sum(
                    util.distance_euclidean(
                        self.get_vertex(charger[0]).coordinate, self.get_vertex(stop.vertex_id).coordinate
                    )
                    for stop in r)
            d[charger] = d_r
        return d

    # Function to create a sorted version of the existing graph
    def sort_graph_representation(self):
        """Sort nodes/edges in any order to ensure deterministic behaviour of networkx functions (e.g. spp generator)"""
        sorted_G = nx.DiGraph()
        # Add nodes in a sorted order
        for node in sorted(self._network.nodes()):
            sorted_G.add_node(node)

        # Add edges in a sorted order
        for u, v, data in sorted(self._network.edges(data=True), key=lambda x: (x[0], x[1])):
            sorted_G.add_edge(u, v, **data)
        self._graph = sorted_G
        self._invalidate_vertex_cache()
        self._invalidate_arc_cache()

    @cached_property
    def distance_matrix(self):
        distance_dictionary = self.distance_dictionary
        m = []
        for charger in self.charger_station_list:
            m_i = []
            for i in range(len(self.routes)):
                m_i += [distance_dictionary[charger][self.routes[i].vehicle_id]]
            m += [m_i]
        return np.array(m)

    def dump(self, name: str, path: str="dumped_instances"):
        dict_repr = {
            "_network": util.serialise_nx_graph(self._network),
            "routes": [asdict(r) for r in self._routes],
        }
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")
        with open(f"{path}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(dict_repr, f, indent=4, ensure_ascii=False)


    def plot_spp_route(self, vehicle_id):
        seq = [
            (self.get_vertex(s.vertex_id).coordinate.lat,self.get_vertex(s.vertex_id).coordinate.lon)
            for s in self.get_route(vehicle_id).stop_sequence
        ]
        # Get the graph for the area
        # Calculate the convex hull of the given points
        points = [Point(lon, lat) for lat, lon in seq]
        convex_hull = Polygon(unary_union(points).convex_hull)

        # Get bounding box of the convex hull
        minx, miny, maxx, maxy = convex_hull.bounds

        # Download the graph for the bounding box area
        G = osm.graph_from_bbox(north=maxy, south=miny, east=maxx, west=minx, network_type='drive')

        # Project the graph to UTM (for accurate distance calculations)
        G = osm.project_graph(G)

        # Create a folium map centered at the centroid of the convex hull
        centroid = convex_hull.centroid
        folium_map = folium.Map(location=[centroid.y, centroid.x], zoom_start=14)

        for lat, lon in seq:
            folium.Marker([lat, lon], popup='Point').add_to(folium_map)

        # Function to add a polyline to the folium map
        def add_path_to_map(route):
            path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
            folium.PolyLine(path_coords, color='blue', weight=5).add_to(folium_map)

        # Iterate through the list of lat, lon tuples to get the shortest path between each pair
        for i in range(len(seq) - 1):
            orig_node = osm.nearest_nodes(G, X=seq[i][1], Y=seq[i][0])
            dest_node = osm.nearest_nodes(G, X=seq[i + 1][1], Y=seq[i + 1][0])
            shortest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')
            add_path_to_map(shortest_route)

        # Save the map as an HTML file
        folium_map.save(f"Route {vehicle_id} Plot.html")
        print(f"Map saved as 'Route {vehicle_id} Plot.html'")


def project_to_route(network: IntermediateRepresentation, route: Route) -> IntermediateRepresentation:
    """
    Creates a new vehicle network containing only the specified route. This changes the type of all non-required stops
    to auxiliary. Neither the number of vertices nor the number of arcs changes.
    :param network: The existing network
    :param route: The route to project on
    :return: object of intermediate representation
    """
    assert any(
        x.vehicle_id == route.vehicle_id for x in network.routes), "Network does not contain route to be projected on"

    all_stops = {s.vertex_id for route in network.routes for s in route.stop_sequence}
    superfluous_stops = all_stops.difference({s.vertex_id for s in route.stop_sequence})

    def _project_vertex(v: Vertex) -> Vertex:
        if v.id in superfluous_stops:
            _v = copy.copy(v)
            _v.is_stop = False
            return _v
        return v

    projected_vertices = map(_project_vertex, network.vertices)
    return IntermediateRepresentation(projected_vertices, {(u, v): arc for u, v, arc in network.arcs}, routes=[route])


def split_into_vehicle_networks(network: IntermediateRepresentation) -> dict[VehicleID, IntermediateRepresentation]:
    return dict(sorted({route.vehicle_id: project_to_route(network, route) for route in network.routes}.items()))


def _ensure_vertex_to_node_matching(inter_rep: IntermediateRepresentation, network: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Make sure that each vertex has a counter part (by id) in the given street network.
    :param inter_rep: Intermediate representation
    :param network: Street network
    :return: Street network with splitted edges and added nodes where required.
    """
    # assert network.graph["crs"] == "epsg:4326", "Given street network should be projected in WGS84 reference system."
    for v in inter_rep.vertices:
        # for each vertex in intermediate representation check if any corresponding node in network
        if v.id in network.nodes:
            continue
        _, _ = util.project_node_to_edge(network, v.coordinate, v.id)
    return network


def _generate_polygon_from_intermediate_representation(
        network: IntermediateRepresentation,
        buffer: int = 800
) -> Polygon:
    """
    Find Polygon covering all vertex location from given intermediate representation
    :param network: intermediate representation
    :param buffer: buffer around given coordinates (defaults to 800 m)
    :return: Polygon
    """
    coord_list = [c.coordinate for c in network.vertices]
    buffered_coords = util.transform_coordinate_list_to_buffered_list(coord_list, buffer)
    return MultiPolygon(buffered_coords).convex_hull


def load_corresponding_osmnx_street_network_from_intermediate_representation(
        inter_rep: IntermediateRepresentation
) -> nx.MultiDiGraph:
    """
    1. Load osmnx street network corresponding to locations given in intermediate representation
    2. Convert node ids to type sting
    3. And add missing nodes (vertex ids not in street network and distance to nearest node in street network > 1m)
    :param inter_rep: Intermediate representation
    :return: Street network w/ nodes added that the intermediate representation requires
    """
    polygon = _generate_polygon_from_intermediate_representation(inter_rep)
    osmnx_graph = osm.graph_from_polygon(
        polygon,
        network_type="drive",
        truncate_by_edge=True,
        simplify=True,
    )
    # project into utm reference system
    osmnx_graph = osm.project_graph(osmnx_graph, util.get_utm_zone([c.coordinate for c in inter_rep.vertices]))
    util.enforce_string_node_identifier(osmnx_graph)
    network = _ensure_vertex_to_node_matching(inter_rep, osmnx_graph)
    return network


def parse_intermediate_representation(filepath: str)->IntermediateRepresentation:
    """tbd"""
    with open(f'{filepath}', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    network = deserialise_nx_graph(data["_network"])
    routes = [Route(
        vehicle_id=r["vehicle_id"],
        stop_sequence=[Stop(
            vertex_id = s["vertex_id"],
            stopover_time = s["stopover_time"],
            earliest_time_of_service = s["earliest_time_of_service"],
            latest_time_of_service = s["latest_time_of_service"],
        ) for s in r["stop_sequence"]]
    ) for r in data["routes"]
    ]
    return IntermediateRepresentation(network=network, routes=routes, vertices=None, arcs=None)


def path_weight(G, path):
    """Returns total cost associated with specified path and weight

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    path: list
        A list of node labels which defines the path to traverse

    Returns
    -------
    cost: int or float
        An integer or a float representing the total cost with respect to the
        specified weight of the specified path

    Raises
    ------
    NetworkXNoPath
        If the specified edge does not exist.
    """
    cost = 0
    if not nx.is_path(G, path):
        raise nx.NetworkXNoPath("path does not exist")
    for node, nbr in nx.utils.pairwise(path):
        cost += G[node][nbr]["arc"].distance
    return cost


def deserialise_nx_graph(
    data,
    directed=False,
    *,
    source="source",
    target="target",
    name="id",
    key="key",
    link="links",
):
    """Returns graph from node-link data format.
    Useful for de-serialization from JSON.

    Parameters
    ----------
    data : dict
        node-link formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    key : string
        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.
    link : string
        A string that provides the 'link' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    G : NetworkX graph
        A NetworkX graph object


    Notes
    -----
    Attribute 'key' is only used for multigraphs.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.

    See Also
    --------
    node_link_data, adjacency_data, tree_data
    """
    directed = data.get("directed", directed)
    graph = nx.Graph()
    if directed:
        graph = graph.to_directed()

    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None
    graph.graph = data.get("graph", {})
    c = count()
    for d in data["nodes"]:
        node = util._to_tuple(d.get(name, next(c)))
        nodedata = {
            "vertex": Vertex(
                id=VertexID(d["id"]),
                is_depot=d["is_depot"],
                is_stop=d["is_stop"],
                coordinate=util.Coordinate(lat=d["coordinate"]["lat"], lon=d["coordinate"]["lon"]),
                constructible_charger={Charger(
                    id=c["id"],
                    segment_construction_cost=float(c["segment_construction_cost"]),
                    transformer_construction_cost=float(c["transformer_construction_cost"]),
                    charging_rate=float(c["charging_rate"]),
                ) for c in d["constructible_charger"]},
                name=d["name"],
            )
        }
        graph.add_node(node, **nodedata)
    for d in data[link]:
        src = tuple(d[source]) if isinstance(d[source], list) else d[source]
        tgt = tuple(d[target]) if isinstance(d[target], list) else d[target]
        edgedata = {
            "arc": Arc(
                distance=float(d["distance"]),
                speed_limit=float(d["speed_limit"]),
                constructible_chargers={Charger(
                    id=c["id"],
                    segment_construction_cost=float(c["segment_construction_cost"]),
                    transformer_construction_cost=float(c["transformer_construction_cost"]),
                    charging_rate=float(c["charging_rate"]),
                ) for c in d["constructible_charger"]},
                absolute_consumption_overwrite=None if d["consumption_overwrite"]=="" else float(d["consumption_overwrite"]),
                speed_overwrite_kmh=None
            )}
        graph.add_edge(src, tgt, **edgedata)
    return graph
