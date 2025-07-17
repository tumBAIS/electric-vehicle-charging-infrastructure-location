# coding=utf-8
import enum
import graphviz
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, pairwise
from math import isclose
from typing import Iterable, NewType, Optional, Tuple, Dict, Any, Union

import graphviz
import networkx as nx
from docplex.mp.dvar import Var
from networkx.exception import NetworkXNoCycle

Vehicle = NewType("Vehicle", str)
VertexID = NewType("VertexID", str)
ArcID = Tuple[VertexID, VertexID]


class ValidationError(BaseException):
    pass


def certainly_gt(a: float, b: float, toler: float=1e-4) -> bool:
    return a > b and not isclose(a, b, abs_tol=toler)


def certainly_lt(a: float, b: float, toler: float=1e-4) -> bool:
    return a < b and not isclose(a, b, abs_tol=toler)


def certainly_unequal(a: float, b: float) -> bool:
    return certainly_gt(a,b) | certainly_lt(a,b)


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class DynamicCharger:
    id: Optional[int]
    construction_cost_per_km: float
    transformer_construction_cost: float
    charging_rate_kwh_per_h: float
    irID: Optional[str] = None

    def __eq__(self, other):
        return id(self) == id(other)


@dataclass(frozen=True, eq=False)
class StaticCharger:
    id: Optional[int]
    construction_cost: float
    charging_rate_kwh_per_h: float
    irID: Optional[str] = None

    def __eq__(self, other):
        return id(self) == id(other)


class VertexType(enum.IntFlag):
    DEPOT = 1
    STOP = 2
    CHARGER = 4
    AUXILIARY = 8


@dataclass
class Vertex:
    id: VertexID
    type: VertexType
    time_window_begin: Union[float, int]
    time_window_end: Union[float, int]
    charger: Optional[StaticCharger]
    name: Optional[str] = None

    dummy_of: Optional["Vertex"] = None
    replica_level: int = 0

    tau: Optional[Var] = field(default=None, compare=False)
    rho: Optional[Var] = field(default=None, compare=False)
    y: Optional[Var] = field(default=None, compare=False)
    delta_tau: Optional[Var] = field(default=None, compare=False)
    delta_rho: Optional[Var] = field(default=None, compare=False)

    @property
    def is_dummy(self) -> bool:
        return self.dummy_of is not None and self.dummy_of.id != self.id

    @property
    def root_node(self):
        return self.dummy_of if self.is_dummy else self

    @property
    def delta_charge(self) -> float:
        return self.delta_rho.solution_value

    @property
    def delta_time(self) -> float:
        return self.delta_tau.solution_value

    @property
    def can_construct_charger(self) -> bool:
        return bool(self.type & VertexType.CHARGER)

    @property
    def is_depot(self) -> bool:
        return bool(self.type & VertexType.DEPOT)

    @property
    def is_stop(self) -> bool:
        return bool(self.type & VertexType.STOP)

    @property
    def is_auxiliary(self) -> bool:
        # note that auxiliary is only possible standalone while other vertices can be stop and depot
        return bool(self.type & VertexType.AUXILIARY)

    @property
    def charging_rate(self) -> float:
        return self.charger.charging_rate_kwh_per_h

    @property
    def departure_time(self) -> float:
        return self.tau.solution_value

    @property
    def arrival_soc(self) -> float:
        return self.rho.solution_value - self.delta_rho.solution_value

    @property
    def departure_soc(self) -> float:
        return self.rho.solution_value

    def validate_solution(self, time_step=None):
        _time_step = time_step if time_step is not None else 1.
        # No charge if no charger constructed
        if not self.can_construct_charger:
            if self.delta_charge > 0 or self.y.to_bool():
                raise ValidationError(f"Can not charge or construct charger at node {self.id}")
        else:
            if certainly_gt(
                    self.delta_charge,
                    _time_step * self.delta_time * (self.charger.charging_rate_kwh_per_h / 3600),
            ):
                raise ValidationError(
                    f"Vehicles charges {self.delta_charge} at vertex {self} but only "
                    f"{_time_step * self.delta_time * (self.charger.charging_rate_kwh_per_h / 3600)} is expected"
                )
        if certainly_gt(self.departure_time, self.time_window_end) or certainly_lt(
                self.departure_time, self.time_window_begin
        ):
            raise ValidationError(f"Time window violation error at {self.name}")

    def __str__(self):
        return f"{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, type(self.id)):
            return self.id == other
        return self.id == other.id


@dataclass
class Arc:
    """
    There is the  need to clarify units in this representation (as they differ from intermediate representation):
     - max_travel_time: hours
     - cost: cost unit (no specific currency)
     - distance (km!) - in intermediate representation this is km (I dont know why this discrepancy)
     - min_consumption: kwh
     - max_consumption: kwh
    """
    origin: Vertex
    target: Vertex
    max_travel_time: float
    distance: float
    min_consumption: float
    max_consumption: float
    charger: Optional[DynamicCharger]

    dummy_of: Optional["Arc"] = None
    replica_level: int = 0

    z: Optional[Var] = field(default=None, compare=False)
    delta_rho: Optional[Var] = field(default=None, compare=False)
    x: Optional[Var] = field(default=None, compare=False)

    @property
    def segment_construction_cost(self) -> float:
        assert self.charger is not None
        return self.distance * self.charger.construction_cost_per_km

    @property
    def delta_charge(self) -> float:
        return self.delta_rho.solution_value

    @property
    def traversed(self) -> bool:
        return self.x.to_bool()

    def validate_solution(self, time_step=None):
        _time_step = time_step if time_step is not None else 1.
        # Dynamic charger/charging
        if not self.can_construct_charger:
            if self.z.to_bool():
                raise ValidationError("Can't construction charger at arc")
        else:
            if not self.z.to_bool() and self.delta_charge > 1e-6:
                raise ValidationError(f'Recharge at unopened charger {self.z.name}, {self.delta_charge} kWh')
            if self.z.to_bool():
                if certainly_gt(
                        self.delta_charge, self.travel_time * self.charging_rate
                ):
                    raise ValidationError('Recharge amount error')

        if self.traversed:
            # some numerical instability due to second <-> hour conversion (can be accepted as we are still talking
            # about the precision of a second and 1% precision of a kWh)
            # Travel time
            checksum = self.origin.departure_time + self.travel_time_seconds
            if certainly_gt(checksum, self.target.departure_time - _time_step * self.target.delta_time, 1):
                # print(self.origin.arrival_time, self.origin.delta_time, self.travel_time_seconds, self.target.delta_time)
                raise ValidationError(f'Travel time flow error {checksum}, {self.target.departure_time - _time_step * self.target.delta_time}, {self.origin}, {self.target}')

            # Consumption
            checksum = self.origin.arrival_soc - self.consumption + self.delta_charge + self.origin.delta_charge
            if certainly_lt(checksum, self.target.arrival_soc, 1e-3):
                raise ValidationError(f'Consumption flow error {checksum}, {self.target.arrival_soc}')

    @property
    def is_dummy(self):
        return self.dummy_of is not None and (self.dummy_of.origin != self.origin or self.dummy_of.target != self.target)

    @property
    def consumption(self):
        return (
                self.min_consumption + (self.max_consumption - self.min_consumption) / 2.0
        )

    @property
    def can_construct_charger(self) -> bool:
        return self.charger is not None

    @property
    def travel_time(self):
        return self.max_travel_time

    @property
    def travel_time_seconds(self):
        return self.max_travel_time * 3600

    @property
    def charging_rate(self) -> float:
        return self.charger.charging_rate_kwh_per_h if self.charger else 0

    @property
    def attr_data(self) -> Dict[str, Any]:
        return {
            "max_travel_time": self.max_travel_time,
            "distance": self.distance,
            "min_consumption": self.min_consumption,
            "max_consumption": self.max_consumption,
            "charger": self.charger,
            "dummy_of": self.dummy_of,
            "z": self.z,
            "delta_rho": self.delta_rho,
            "x": self.x,
        }

    def __str__(self):
        return f"({self.origin}, {self.target})"

    def __hash__(self):
        return hash((self.origin, self.target))

    def __eq__(self, other):
        return (self.origin == other.origin) & (self.target == other.target)


class VehicleNetwork:
    """
    Should contain replicas for each station. These should not be connected to each other though
    """

    def __init__(self, vehicle: Vehicle, network: nx.DiGraph):
        self._vehicle = vehicle
        self._graph = network

        self._vertex_by_id = {vertex.id: vertex for vertex in self._graph.nodes}
        assert len(self._vertex_by_id) == len(self._graph)

        self._validate()

    def _validate(self):
        # Has end/start depot
        try:
            start_depot = self.start_depot
            end_depot = self.end_depot
            if start_depot is end_depot:
                raise ValueError("Start depot equals end depot!")

        except StopIteration:
            raise ValueError("Start/end depot undefined")

        # Exactly two depots
        if not sum(1 for vertex in self.vertices if vertex.is_depot) == 2:
            raise ValueError("Not exactly 2 depot representations")

        # types are correct
        for v in self.vertices:
            if v.type & VertexType.AUXILIARY:
                assert v.type == VertexType.AUXILIARY
            if v.type & VertexType.STOP:
                assert v.type & VertexType.CHARGER or v.type == VertexType.STOP

    def validate_solution(self, time_step=None):
        # Arc/Vertex constraints
        for v in self.vertices:
            v.validate_solution(time_step=time_step)
        for arc in self.arcs:
            arc.validate_solution(time_step=time_step)
        # All stops served
        for v in self.required_visits:
            if not any(x.traversed for x in self.get_incoming_arcs(v)):
                raise ValidationError
        # Depot left
        if not any(x.traversed for x in self.get_outgoing_arcs(self.start_depot)):
            raise ValidationError
        # No subtours
        subgraph = nx.subgraph_view(
            self._graph.copy(), filter_edge=lambda u, v: self.get_edge(u, v).traversed
        )
        subgraph = subgraph.copy()
        subgraph.remove_nodes_from([x for x in subgraph if subgraph.degree(x) == 0])

        if not nx.is_weakly_connected(subgraph):
            weakly_connected_components = list(nx.weakly_connected_components(subgraph))

            # Print weakly connected components for debugging
            print("Weakly connected components:")
            for component in weakly_connected_components:
                print("==========================")
                print(component)

            print("Traversed arcs:")
            for u,v in subgraph.edges:
                print(u,v)
            raise ValidationError(f"{self._vehicle}")
        if not nx.has_path(subgraph, self.start_depot, self.end_depot):
            raise ValidationError(f"{self._vehicle}")
        # Only one path
        path_iter = nx.all_simple_paths(subgraph, self.start_depot, self.end_depot)
        next(path_iter)
        try:
            next(path_iter)
            raise ValidationError
        except StopIteration:
            pass

    def get_vertex(self, vertex_id: VertexID) -> Vertex:
        return self._vertex_by_id[vertex_id]

    @property
    def arcs(self) -> Iterable[Arc]:
        return (arc for *_, arc in self._graph.edges(data="arc"))

    @property
    def vertices(self) -> Iterable[Vertex]:
        return self._graph.nodes

    @property
    def dynamic_chargers(self) -> Iterable[Arc]:
        return (arc for arc in self.arcs if arc.can_construct_charger)

    @property
    def static_chargers(self) -> Iterable[Vertex]:
        return (vertex for vertex in self.vertices if vertex.can_construct_charger)

    @property
    def _transformers(self) -> Iterable[DynamicCharger]:
        return set(f.charger for f in self.dynamic_chargers)

    @property
    def _chargers_by_transformer(self) -> Iterable[DynamicCharger]:
        charger_sets = defaultdict(set)
        for f in self._transformers:
            charger_sets[f].add(f)
        return charger_sets

    @property
    def stops(self) -> Iterable[Vertex]:
        return (vertex for vertex in self.vertices if vertex.is_stop)

    @property
    def start_depot(self) -> Vertex:
        return next(
            vertex
            for vertex in self.vertices
            if vertex.is_depot and self._graph.in_degree(vertex) == 0
        )

    @property
    def end_depot(self) -> Vertex:
        return next(
            vertex
            for vertex in self.vertices
            if vertex.is_depot and self._graph.out_degree(vertex) == 0
        )

    @property
    def latest_arrival_time(self) -> float:
        return self.end_depot.time_window_end

    @property
    def required_visits(self) -> Iterable[Vertex]:
        return chain([self.end_depot], self.stops)

    def get_edge(self, u: Vertex, v: Vertex) -> Arc:
        return self._graph.edges[u, v]["arc"]

    def get_incoming_arcs(self, vertex: Vertex) -> Iterable[Arc]:
        return (arc for *_, arc in self._graph.in_edges(nbunch=vertex, data="arc"))

    def get_outgoing_arcs(self, vertex: Vertex) -> Iterable[Arc]:
        return (arc for *_, arc in self._graph.out_edges(nbunch=vertex, data="arc"))

    def plot(self, hide_non_traversed_arcs: bool = False):
        viz = graphviz.Digraph(f"Vehicle {self._vehicle}")
        for v in self.vertices:
            if v.is_depot:
                shape = "square"
            elif v.is_stop:
                if v.can_construct_charger:
                    shape = "hexagon"
                else:
                    shape = "circle"
            else:
                shape = "triangle"
            viz.node(str(v), str(v), shape=shape)

        for arc in self.arcs:
            if arc.x is not None and arc.x.to_bool():
                style = "solid"
            elif arc.x is not None:
                if hide_non_traversed_arcs:
                    continue
                style = "dashed"
            else:
                # No solution yet
                style = "solid"
            if arc.z is not None and arc.z.to_bool():
                color = "green"
            else:
                color = "black"
            viz.edge(
                str(arc.origin),
                str(arc.target),
                label=f"t_ij: {arc.travel_time_seconds:.2f} secs, q_ij: {arc.consumption:.2f} kwh, d_ij: {arc.distance:.2f} km",
                color=color,
                style=style,
            )
        viz.render(view=True)
