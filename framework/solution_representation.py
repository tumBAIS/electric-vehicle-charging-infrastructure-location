# coding=utf-8
import datetime as dt
import json
import pytz
import copy
import re
import dataclasses
import math
import os
import enum
import networkx as nx
from docplex.mp.linear import Var
from itertools import pairwise
from collections import defaultdict
from dataclasses import dataclass
from pydantic.tools import parse_obj_as
from pydantic import field_validator, model_validator, GetCoreSchemaHandler
from typing import List, Dict, Tuple, Optional, Any
from framework.intermediate_representation import (
    VehicleID,
    VertexID,
    ArcID,
    ChargerID,
    IntermediateRepresentation
)
from framework.utils import ValidationError

ArrivalDepartureTimes = Tuple[dt.datetime, dt.datetime]
LineID = str


class PointType(enum.IntFlag):
    DEPOT = 1
    STOP = 2
    STATIC_CHARGER = 4
    DYN_REPRESENTATION = 8


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, float):
            return round(o, 4)  # Round floats to 4 decimal places
        elif isinstance(o, int):
            return o
        elif isinstance(o, dt.datetime):
            return {'_type': 'datetime.timestamp', '_value': o.strftime("%m/%d/%Y, %H:%M:%S")}
        elif isinstance(o, enum.IntFlag):
            # Handle Point serialization
            return {'_type': 'PointType', '_value': int(o)}
        elif isinstance(o, Point):
            # Serialize the Point object as a dictionary
            return {
                '_type': 'Point',
                'id': o.id,
                'arrival_time': self.default(o.arrival_time),  # Serialize datetime
                'departure_time': self.default(o.departure_time),  # Serialize datetime
                'soc': o.soc,
                'type': self.default(o.type),  # Serialize PointType
                'accumulated_charged_energy': self.default(o.accumulated_charged_energy),
                'accumulated_consumed_energy': self.default(o.accumulated_consumed_energy)
            }

        return super().default(o)

def parse_point(d):
    def parse_datetime(t):
        if isinstance(t, int):
            return t
        return dt.datetime.strptime(t['_value'], "%m/%d/%Y, %H:%M:%S")

    def parse_point_type(pt):
        if isinstance(pt, dict):
            return PointType(pt["_value"])
        #print(type(pt))
        return PointType(pt)

    is_static_charger = bool(parse_point_type(d["type"]) & PointType.STATIC_CHARGER)
    if d["id"]=="Depot":
        is_static_charger=False
    return Point(
        id=d['id'],
        arrival_time=parse_datetime(d['arrival_time']),  # Deserialize datetime
        departure_time=parse_datetime(d['departure_time']),  # Deserialize datetime
        soc=d['soc'],
        is_depot=bool(parse_point_type(d["type"]) & PointType.DEPOT),
        is_stop=bool(parse_point_type(d["type"]) & PointType.STOP),
        accumulated_charged_energy=d['accumulated_charged_energy'],
        accumulated_consumed_energy=d['accumulated_consumed_energy'],
        is_static_charger=is_static_charger,
        is_synthetic_dyn_charger_representation=bool(parse_point_type(d["type"]) & PointType.DYN_REPRESENTATION),
    )


class Point:
    id: VertexID
    arrival_time: int
    departure_time: int
    soc: float
    type: PointType
    accumulated_charged_energy: float
    accumulated_consumed_energy: float

    @model_validator(mode="after")
    def _valid_schedule(cls, values):
        assert (values.departure_time - values.arrival_time).total_seconds() >= 0.0, f"Schedule at vertex {values.id} not valid"
        assert not (values.is_depot and values.is_stop), f"Point {values.id} cannot be depot and stop."
        return values

    def __init__(self, id: VertexID, arrival_time: int, departure_time: int, soc: float, is_depot: bool,
                 is_stop: bool, accumulated_consumed_energy: float, accumulated_charged_energy: float, is_static_charger=False,
                 is_synthetic_dyn_charger_representation=False):
        """
        The constructor via the single attributes (is_depot, is_stop, ...) is a legacy product and should be simplified
        """
        self.id = id
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.soc = soc
        self.type = PointType(0)
        if is_depot:
            self.type |= PointType.DEPOT
        if is_stop:
            self.type |= PointType.STOP
        if is_static_charger:
            self.type |= PointType.STATIC_CHARGER
        if is_synthetic_dyn_charger_representation:
            self.type |= PointType.DYN_REPRESENTATION
        self.accumulated_charged_energy = accumulated_charged_energy
        self.accumulated_consumed_energy = accumulated_consumed_energy

    def __add__(self, other: "Point"):
        assert self.id == other.id

        # logical
        arrival_time = min(self.arrival_time, other.arrival_time)
        departure_time = max(self.departure_time, other.departure_time)
        soc = max(self.soc, other.soc)

        # when merging, we set the attribute 'is_first_charger_node' to False because this is a characteristic which
        # cannot exist / make sense in merged cases (only original spprc network)
        return Point(
            id=self.id,
            arrival_time = arrival_time,
            departure_time = departure_time,
            soc = soc,
            is_depot = False,
            is_stop = self.is_stop or other.is_stop,
            accumulated_consumed_energy = min(self.accumulated_consumed_energy,other.accumulated_consumed_energy),
            accumulated_charged_energy = max(self.accumulated_charged_energy, other.accumulated_charged_energy),
            is_static_charger=self.is_static_charger or other.is_static_charger,
            is_synthetic_dyn_charger_representation=False
        )

    def __eq__(self, other):
        return self.id == other.id and self.arrival_time == other.arrival_time and \
           self.departure_time == other.departure_time

    def __hash__(self):
        return hash((self.id, self.soc, self.arrival_time, self.departure_time, self.is_depot, self.is_stop,
                     self.is_static_charger))

    def __str__(self):
        return str(self.id)

    @property
    def is_depot(self):
        return bool(self.type & PointType.DEPOT)

    @property
    def is_stop(self):
        return bool(self.type & PointType.STOP)

    @property
    def is_static_charger(self):
        return bool(self.type & PointType.STATIC_CHARGER)

    @property
    def is_synthetic_dyn_charger_representation(self):
        return bool(self.type & PointType.DYN_REPRESENTATION)

    # only for debugging
    @property
    def arrival_time_int(self):
        if isinstance(self.arrival_time, dt.datetime):
            return (self.arrival_time - dt.datetime(1970, 1, 1)).total_seconds()
        return self.arrival_time

    @property
    def departure_time_int(self):
        if isinstance(self.departure_time, dt.datetime):
            return (self.departure_time - dt.datetime(1970, 1, 1)).total_seconds()
        return self.departure_time


def _refine(route: List[Point]):
    for u, v in pairwise(route):
        u.id = u.id.split('_')[0]
        v.id = v.id.split('_')[0]


@dataclass
class Itinerary:
    vehicle: VehicleID
    route: List[Point]

    def __init__(self, vehicle: VehicleID, route: List[Point]):
        self.vehicle = vehicle
        self.route = route

    @model_validator(mode="after")
    def _valid_schedule(cls, values):
        for p1, p2 in pairwise(values.route):
            assert p1.departure_time <= p2.arrival_time
        return values


@dataclass
class SolCharger:
    """On segment level (in dynamic case)"""
    id: ChargerID
    charger_cost: float
    transformer_cost_share: Optional[float]  # only for dynamic chargers

    def __init__(self, id: ChargerID, charger_cost: float, transformer_cost_share: Optional[float]):
        self.id = id
        self.charger_cost = charger_cost
        self.transformer_cost_share = transformer_cost_share

    def __post_init__(self):
        # dynamic stations must be inductive & only dynamic stations segments share transformers
        assert self.transformer_cost_share is None

    @property
    def cost(self):
        return self.charger_cost if self.transformer_cost_share is None else self.charger_cost + self.transformer_cost_share


@dataclass
class SolutionRepresentation:
    itineraries: List[Itinerary]
    dynamic_invest: Optional[Dict[ArcID, SolCharger]]  # design choice: each segment has its own arc
    static_invest: Optional[Dict[VertexID, SolCharger]]
    routing_cost: float
    consumed_energy: float
    recharged_energy: float

    def __init__(self, itineraries: list[Itinerary], dynamic_invest: Dict[ArcID, SolCharger],
                 static_invest: Dict[VertexID, SolCharger], routing_cost: float,consumed_energy: float,
                 recharged_energy: float):
        self.itineraries = itineraries
        self.dynamic_invest = dynamic_invest
        self.static_invest = static_invest
        self.routing_cost = routing_cost
        self.consumed_energy = consumed_energy
        self.recharged_energy = recharged_energy

    def __deepcopy__(self, memo=None):
        # Create a deepcopy of the object, avoiding infinite recursion by using memo
        if memo is None:
            memo = {}

        # Create the deepcopied object
        copied_object = SolutionRepresentation(
            itineraries=copy.deepcopy(self.itineraries, memo),
            dynamic_invest=copy.deepcopy(self.dynamic_invest, memo),
            static_invest=copy.deepcopy(self.static_invest, memo),
            routing_cost=self.routing_cost,
            consumed_energy=self.consumed_energy,
            recharged_energy=self.recharged_energy
        )

        # Optionally, add this object to the memo dictionary
        memo[id(self)] = copied_object
        return copied_object

    def __add__(self, other: "SolutionRepresentation"):
        """This is required because the solution representation is used in the algorithmic implementation :("""
        assert other.dynamic_invest == self.dynamic_invest
        assert other.static_invest == self.static_invest
        return SolutionRepresentation(
            self.itineraries+other.itineraries,
            self.dynamic_invest,
            self.static_invest,
            self.routing_cost+other.routing_cost,
            self.consumed_energy+other.consumed_energy,
            self.recharged_energy+other.recharged_energy,
        )

    @property
    def investment_cost(self):
        return sum(charger.cost for charger in self.dynamic_invest.values()) + \
            sum(charger.cost for charger in self.static_invest.values())

    @property
    def global_cost(self):
        return self.routing_cost + self.investment_cost

    def dump_as_json(self, path='./solution_representation.json'):
        """dump solution representation"""
        def convert_tuples_in_dict(d):
            """ Recursively converts tuple keys in a dictionary to lists. """
            # Convert tuples in data_dict to a serializable format
            def handle_tuple_key(k, v):
                # Convert tuple keys into a string joined by '-'
                new_key = '-'.join(map(str, k))
                # Initialize a new dictionary or use the existing one
                nested_dict = v if isinstance(v, dict) else {'value': v}
                # Add tuple elements to the dictionary
                for idx, elem in enumerate(k):
                    nested_dict[f'key_{idx}'] = elem
                return new_key, nested_dict

            flattened = {}
            for k, v in d.items():
                if isinstance(k, tuple):
                    new_key, new_value = handle_tuple_key(k, v)
                    flattened[new_key] = new_value
                elif isinstance(v, dict):
                    # Recursively flatten nested dictionaries
                    flattened[k] = convert_tuples_in_dict(v)
                elif isinstance(v, float):
                    flattened[k] = round(v, 4)
                else:
                    # Copy other key-value pairs as-is
                    flattened[k] = v

            return flattened

        # convert to dictionary with built-in function of python dataclasses, then convert
        dump_orig = dataclasses.asdict(self)
        dump_orig["investment_cost"] = round(self.investment_cost,4)
        dump_orig["total_cost"] = round(self.global_cost,4)
        dump = convert_tuples_in_dict(dump_orig)

        # create path and write file
        dump_dir = path.split('/')[:-1] if len(path.split('/')) > 1 else '.'
        dump_dir = "/".join(dump_dir)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(dump, f, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)

    @staticmethod
    def load_from_json(path='./solution_representation.json'):
        with open(path, 'r') as f:
            dump = json.load(f)
            for itinerary in dump['itineraries']:
                itinerary['route'] = [parse_point(point) for point in itinerary['route']]
            for key, value in dump["static_invest"].items():
                dump["static_invest"][key] = SolCharger(value["id"], value["charger_cost"], value["transformer_cost_share"])
            for key, value in dump["dynamic_invest"].items():
                dump["dynamic_invest"][key] = SolCharger(value["id"], value["charger_cost"], value["transformer_cost_share"])
            return SolutionRepresentation(
                itineraries=[Itinerary(d["vehicle"], route=d["route"]) for d in dump["itineraries"]],
                dynamic_invest=dump["dynamic_invest"],
                static_invest=dump["static_invest"],
                routing_cost=dump["routing_cost"],
                consumed_energy=dump["consumed_energy"],
                recharged_energy=dump["recharged_energy"],
            )

    def validate(
            self, ir:IntermediateRepresentation, soc_bounds: Tuple[float, float], precision: Optional[float]=1e-4,
            check_time_windows: Optional[bool]=True):
        # check routes
        dyn_chargers = [c.id for k,c in self.dynamic_invest.items()]
        for itin in self.itineraries:
            # assert soc profile
            for point in itin.route:
                assert soc_bounds[0] - precision <= point.soc <= soc_bounds[1] + precision, f"{point.id}, {point.soc}"

            # assert re-charge only at stations and feasible amount
            checksum = 0.0
            for point1, point2 in pairwise(itin.route):
                checksum += (point2.accumulated_charged_energy - point1.accumulated_charged_energy)
                checksum -= (point2.accumulated_consumed_energy - point1.accumulated_consumed_energy)
                if abs(point2.accumulated_charged_energy - point1.accumulated_charged_energy) > precision:
                    assert point2.is_static_charger or point2.is_synthetic_dyn_charger_representation or \
                           point1.is_synthetic_dyn_charger_representation, f"Charging profile between {point1.id} and {point2.id}"
                    if not point2.id in self.static_invest.keys() or point1.id in dyn_chargers \
                           or point2.id in dyn_chargers or (point1.id, point2.id) in self.dynamic_invest.keys() or f"{point1.id}-{point2.id}" in self.dynamic_invest.keys():
                        f"Charging profile between {point1.id} and {point2.id}, {point1.accumulated_charged_energy}, {point2.accumulated_charged_energy}"
            assert abs(checksum + itin.route[0].soc - itin.route[-1].soc) <= precision, f"{checksum}; {itin.route[0].soc}, {itin.route[-1].soc}"
            assert abs(itin.route[-1].accumulated_charged_energy - itin.route[-1].accumulated_consumed_energy - checksum) <= precision

            # assert every stop serviced in time window
            route = ir.get_route(itin.vehicle)
            itin_idx = 0
            for stop in route.stop_sequence[1:-1]:
                found = False
                for idx, point in enumerate(itin.route[itin_idx:]):
                    # time in seconds has precision of >1sec
                    if (point.id == stop.vertex_id) and \
                            ((stop.earliest_time_of_service - 1 <= point.departure_time_int) and \
                            (point.departure_time_int <= stop.latest_time_of_service + 1) or not check_time_windows):
                        itin_idx += idx
                        found = True
                        break
                if found:
                    continue
                raise ValueError(f"stop {stop} has no representation in Solution")

        # assert top level cost structure
        assert abs(self.global_cost - self.routing_cost - self.investment_cost) < precision

        # assert low level cost structure
        checksum = 0.0
        for key, sol_charger in self.static_invest.items():
            checksum += sol_charger.charger_cost + sol_charger.transformer_cost_share

        for key, sol_charger in self.dynamic_invest.items():
            checksum += sol_charger.charger_cost + sol_charger.transformer_cost_share
        assert abs(self.investment_cost - checksum) < precision, f"{checksum}"


def parse_solution_representation_to_warmstart(warmstart: str, var_list: List[Var], time_step: int) -> Dict[str, int]:
    """This parser only works for the solomon instances - has been developed for debugging"""
    warmstart_values = {}
    warmstart_solution = SolutionRepresentation.load_from_json(warmstart)
    warmstart_itineraries = {itin.vehicle: itin.route for itin in warmstart_solution.itineraries}

    warmstart_transformers = [c.id for c in warmstart_solution.dynamic_invest.values()]
    occurences = {v_id: defaultdict(int) for v_id, arcs in warmstart_itineraries.items()}
    for v_id, points in warmstart_itineraries.items():
        for i in range(len(points) - 1):
            u = points[i]
            v = points[i + 1]

            if u.id == "Depot":
                u.id = "Depot_START"
            if v.id == "Depot":
                v.id = "Depot_END"
            # has been dealt with in previous iteration
            if bool(u.type & PointType.DYN_REPRESENTATION) or bool(u.type & PointType.STATIC_CHARGER):
                continue

            # dyn charger rep
            if bool(v.type & PointType.DYN_REPRESENTATION):
                node_str = v.id.split("_")[0] + "_" + v.id.split("_")[2]
                if occurences[v_id][v.id] == 0:
                    node_id = f"DC_{node_str}_0"
                else:
                    node_id = f"DC_{node_str}_0_R{occurences[v_id][v.id]}"
                end_id = points[i + 2].id if points[i + 2].id != "Depot" else "Depot_END"
                spp = nx.shortest_path(
                    self._vehicle_networks[v_id]._graph,
                    source=self._vehicle_networks[v_id].get_vertex(node_id),
                    target=self._vehicle_networks[v_id].get_vertex(end_id)
                )
                for a, b in pairwise([u.id] + spp):
                    warmstart_values[f"x_{v_id}_({a},{b})"] = 1
                end_id = points[i + 2].id if points[i + 2].id != "Depot" else "Depot_END"
                warmstart_values[f"delta_tau_{v_id}_{end_id}"] = int(math.floor(
                    (points[i + 2].departure_time - points[
                        i + 2].arrival_time + 1) / time_step))
            elif bool(v.type & PointType.STATIC_CHARGER):
                if occurences[v_id][v.id] > 0:
                    node_id = v.id+f"_R{occurences[v_id][v.id]}"
                else:
                    node_id = v.id
                warmstart_values[f"x_{v_id}_({u.id},{node_id})"] = 1
                warmstart_values[f"delta_tau_{v_id}_{node_id}"] = int(
                    math.floor((v.departure_time - v.arrival_time + 1) / time_step))
                end_id = points[i + 2].id if points[i + 2].id != "Depot" else "Depot_END"
                warmstart_values[f"x_{v_id}_({node_id},{end_id})"] = 1
                warmstart_values[f"delta_tau_{v_id}_{end_id}"] = int(
                    math.floor((points[i+2].departure_time - points[i+2].arrival_time + 1) / time_step))
            else:
                # neither stop nor charger (attention: this does not cover, loops on the same route)
                warmstart_values[f"x_{v_id}_({u.id},{v.id})"] = 1
                warmstart_values[f"delta_tau_{v_id}_{v.id}"] = int(
                    math.floor((v.departure_time - v.arrival_time + 1) / time_step))
            occurences[v_id][v.id] += 1

    # should include binary variables
    for var in list(var_list):
        name = var.name
        ws_value = 0.0
        # there should be some case
        # 1. x
        if "x_" in name:
            # determined above
            if name not in warmstart_values:
                ws_value = 0.0
            else:
                continue
        elif "y_" in name:
            vtx = "_".join(name.split("_")[1:5])
            ws_value = 1 if vtx in warmstart_solution.static_invest.keys() else 0
        elif "z_" in name:
            arc = re.sub(r'_R\d+', '', "-".join(name[3:-1].split(",")))
            ws_value = 1 if arc in warmstart_solution.dynamic_invest.keys() else 0
        elif "w_" in name:
            transformer = "_".join(name.split("_")[1:])
            ws_value = 1 if transformer in warmstart_transformers else 0
        else:
            raise ValidationError(f"integer variable with unknown naming convention: {name}?")
        warmstart_values[name] = ws_value

    with open('variables.json', 'w') as f:
        json.dump(warmstart_values, f)

    return warmstart_values
