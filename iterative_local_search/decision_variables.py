import copy
import json
import os
import pickle
import logging
from functools import cached_property
from dataclasses import dataclass, asdict, field
import framework.intermediate_representation as ir
from typing import Tuple, Dict, Union, Optional, List, Set
from framework.utils import distance_euclidean
from collections import Counter
from framework.solution_representation import SolCharger, SolutionRepresentation
from framework.intermediate_representation import Charger, IntermediateRepresentation, VertexID, ChargerID
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


@dataclass
class DecisionVariables:
    intermediate_rep: ir.IntermediateRepresentation
    vertex_chargers: dict[ir.VertexID, ir.Charger]
    arc_chargers: dict[tuple[ir.VertexID, ir.VertexID], ir.Charger]
    vehicle_max_speed: ir.KmPerH
    vehicle_consumption: ir.KwhPerKm
    soc_init: ir.Kwh
    max_soc: ir.Kwh
    min_soc: ir.Kwh
    energy_prices: Union[float, Dict[int, float]]
    consumption_cost: float
    lower_energy_bounds: Dict[int, Dict] = field(default_factory=dict) # Mapping RouteId, Vertexid
    deprioritize_route_index: List = field(default_factory=list)
    route_networks: Dict[int, Union[int, SolutionRepresentation]] = field(default_factory=dict)

    def __post_init__(self):
        self.route_networks = {idx: -1 for idx, _ in enumerate(self.intermediate_rep.routes)}

    def __eq__(self, other: "DecisionVariables"):
        return self.vertex_chargers == other.vertex_chargers and self.arc_chargers == other.arc_chargers

    def copy(self, retain_route_networks: bool = True):
        if retain_route_networks:
            return copy.deepcopy(self.vertex_chargers), copy.deepcopy(self.arc_chargers), copy.deepcopy(self.route_networks)
        return copy.deepcopy(self.vertex_chargers), copy.deepcopy(self.arc_chargers)

    @cached_property
    def lower_energy_bounds_per_route(self) -> List[float]:
        """The node based lower bound at the depot is always a valid lower bound"""
        return [d["Depot"] for d in self.lower_energy_bounds.values()]

    @cached_property
    def lower_cost_bound_per_route(self) -> List[float]:
        """Based on the lower bound of energy consumption per vehicle, compute the min. cost per route"""
        cost_bound = []
        recharge_price = self.energy_prices if isinstance(self.energy_prices, float) else min(self.energy_prices.values())
        energy = self.lower_energy_bounds_per_route
        for i in range(len(energy)):
            cost_bound.append(
                energy[i] * self.consumption_cost + max(energy[i] - self.soc_init + self.min_soc, 0.0) * recharge_price
            )
        return cost_bound

    def current_number_of_stationary_chargers(self) -> int:
        return len(self.vertex_chargers)

    def current_number_of_dynamic_chargers(self) -> int:
        return len({c for (u,v),c in self.arc_chargers.items()})

    def current_number_of_chargers(self)->int:
        return self.current_number_of_stationary_chargers() + self.current_number_of_dynamic_chargers()

    def get_arc_properties(self, current_vertex: ir.VertexID, next_vertex: ir.VertexID) -> \
            tuple[Optional[float], Optional[float]]:
        """
        @param current_vertex: origin vertex id
        @param next_vertex: target vertex id
        @return: tuple of (time, energy) that are needed to go from the origin to the target
        """
        if current_vertex == next_vertex:
            return 0.0, 0.0
        arc = self.intermediate_rep.get_arc(current_vertex, next_vertex)
        if arc is None:
            return None, None
        return arc.get_travel_time_seconds(self.vehicle_max_speed), arc.get_consumption(self.vehicle_consumption)

    def get_distance(self, origin_id: ir.VertexID, destination_id: ir.VertexID) -> float:
        """
        @param origin_id: origin vertex id
        @param destination_id: destination vertex id
        @return: the euclidean distance between origin and destination
        """
        return distance_euclidean(
            self.intermediate_rep.get_vertex(origin_id).coordinate,
            self.intermediate_rep.get_vertex(destination_id).coordinate
        )

    def get_travel_time(self, origin_id: ir.VertexID, destination_id: ir.VertexID) -> float:
        """
        @param origin_id: origin vertex id
        @param destination_id: destination vertex id
        @return: the time needed to go from the origin to the destination
        """
        shortest_path = self.intermediate_rep.calc_shortest_path(origin_id, destination_id)
        t = 0
        for u, v in ir.pairwise(shortest_path):
            t += self.intermediate_rep.get_arc(u, v).get_travel_time_seconds(self.vehicle_max_speed)
        return t

    def energy_to_complete_route(self, current_stop: int, route: ir.Route) -> float:
        """
        Compute the total energy needed to complete the route starting from the current stop
        @param current_stop: index of the current stop in the route sequence
        @param route: Route object (vehicle specific)
        @return: energy [kWh] to complete route from given stop 'current_stop' to 'Depot'
        """
        return sum(abs(self.get_arc_properties(route[i].vertex_id, route[i+1].vertex_id)[1])
                   for i in range(current_stop, len(route.stop_sequence) - 1))

    def set_configuration(
            self,
            v_chargers: dict[ir.VertexID, ir.Charger],
            a_chargers: dict[tuple[ir.VertexID, ir.VertexID], ir.Charger],
            route_networks: Optional[Dict[int, Union[int, SolutionRepresentation]]]=None
    ):
        """Setter method to set configuration details"""
        if route_networks is not None:
            self.route_networks = copy.deepcopy(route_networks)
        if v_chargers != self.vertex_chargers or a_chargers != self.arc_chargers:
            # after resetting the configuration, all info about previous configs is useless (if we did not retain it)
            if route_networks is None:
                for idx in self.route_networks.keys():
                    self.route_networks[idx] = -1
            self.vertex_chargers = copy.deepcopy(v_chargers)
            self.arc_chargers = copy.deepcopy(a_chargers)
        else:
            logging.debug("You try to set the new configuration to the current configuration")

    def set_of_charging_stations_for_sampling(
            self,
            filter_for_constructed: bool,
            filter_for_static: bool = False,
            filter_for_dynamic: bool = False
    ) -> list[tuple[str, str, ir.Charger]]:
        """We need this set in which we include all stationary stations + all dynamic stations --> here the
        dynamic stations carry any keys. We need this representation in order to perform a true
        random sampling in which all stations are sampled with equal, uniform probability"""
        assert not (filter_for_static and filter_for_dynamic)
        stations = []
        potential_chargers = {c:(key1, key2) for key1, key2, c in self.intermediate_rep.list_of_all_potential_chargers}
        # if 'filter_for_constructed = True' we filter for the ones in the current configuration, if False we
        # filter exactly for the ones not in the current configuration
        for charger, key in potential_chargers.items():
            if filter_for_static and key[0]!=key[1]:
                continue
            if filter_for_dynamic and key[0]==key[1]:
                continue
            if key[0]==key[1] and (charger in self.vertex_chargers.values())==filter_for_constructed:
                stations.append((key[0], key[1], charger))
            elif key[0]!=key[1] and (charger in self.arc_chargers.values())==filter_for_constructed:
                stations.append((key[0], key[1], charger))
        return stations

    def configuration_set(self) -> Tuple[int]:
        """
        Return the list of indices representing the current config (the mapping to string is retained in this class for now)
        @return: a conf. repr. by [idx, idx2, ..] where every idx represent the charger with index in list of all chargers
        """
        configuration = []
        for k,c in self.vertex_chargers.items():
            configuration.append(self._configuration_mapping[(k, k, c.id)])
        for (k1, k2), c in self.arc_chargers.items():
            configuration.append(self._configuration_mapping[(k1, k2, c.id)])
        return tuple(configuration)

    @cached_property
    def _configuration_mapping(self) -> Dict[Tuple[str, str, ir.ChargerID], int]:
        """
        Return the set of the opened chargers id with corresponding info on their location (i.e., vertex, arc).
        @return: a conf. repr. by (key1, key2, charger id) where charger is the charger object and key1,key2 the location
        """
        mapping = {}
        for (k1, k2, c) in self.intermediate_rep.list_of_all_potential_chargers:
            mapping[(k1, k2, c.id)] = self.intermediate_rep.list_of_all_potential_chargers.index((k1,k2,c))
        return mapping

    def construct_static_invest(self) -> dict[ir.VertexID, SolCharger]:
        """
        Construct the static investment needed for a Solution Representation
        @return: static_invest dictionary
        """
        ids = [charger.id for charger in self.vertex_chargers.values()]
        num_charger_per_transformer = Counter(ids)

        static_invest: dict[ir.VertexID, SolCharger] = {}
        for vertex, charger in self.vertex_chargers.items():
            if charger:
                c: SolCharger
                num_chargers_per_transformer = num_charger_per_transformer[charger.id]
                assert num_chargers_per_transformer == 1
                c = SolCharger(
                    charger.id,
                    0.0,
                    charger.transformer_construction_cost / num_chargers_per_transformer
                )
                static_invest[vertex] = c
        return static_invest

    def construct_dynamic_invest(self) -> dict[tuple[ir.VertexID, ir.VertexID], SolCharger]:
        """
        Construct the dynamic investment needed for a Solution Representation
        @return: dynamic_invest dictionary
        """
        ids = [charger.id for charger in self.arc_chargers.values()]
        num_charger_per_transformer = Counter(ids)

        dynamic_invest = {}
        for arc_id, charger in self.arc_chargers.items():
            if not charger:
                continue
            arc = self.intermediate_rep.get_arc(origin=arc_id[0], target=arc_id[1])
            transformer_cost_share = charger.transformer_construction_cost / num_charger_per_transformer[charger.id]
            dynamic_invest[arc_id] = SolCharger(
                charger.id,
                charger.segment_construction_cost*arc.distance_in_meters,
                transformer_cost_share,
            )
        return dynamic_invest

    def configuration_cost(self) -> float:
        """Return cost of current configuration"""
        # fix cost of stat chargers
        cost = sum(c.transformer_construction_cost for c in self.vertex_chargers.values())
        # variable of dyn chargers
        cost += sum(
            c.segment_construction_cost * self.intermediate_rep.get_arc(k[0], k[1]).distance_in_meters
            for k,c in self.arc_chargers.items()
        )
        # fix cost fo dyn chargers
        cost += sum(charger.transformer_construction_cost for charger in {c for c in self.arc_chargers.values()})
        return cost

    def dump_as_json(self, path='./decision_variable.json'):
        dump = asdict(self)

        dump_dir = path.split('/')[-2] if len(path.split('/')) > 1 else '.'
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        with open(path, 'w') as f:
            json.dump(dump, f, indent=4)

    @staticmethod
    def load_from_json(path='./decision_variable.json'):
        with open(path, 'r') as f:
            dump = json.load(f)
            for itinerary in dump['itineraries']:
                # itinerary['route'] = [parse_obj_as(Point, point) for point in itinerary['route']]
                pass
            # return parse_obj_as(SolutionRepresentation, dump)

    def dump_as_pkl(self, path='./decision_variable.pkl'):
        dump_dir = path.split('/')[-2] if len(path.split('/')) > 1 else '.'
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pkl(path="./decision_variable.pkl"):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def check_routes_affected_by_new_charger(self, charger: tuple[str, str, ir.Charger]):
        """
        This is not straight forward -> set back all route cost such that they have to recomputed
        """
        self.route_networks = {key: -1 for key in self.route_networks}

    def check_routes_affected_by_closed_charger(self, charger: tuple[str, str, ir.Charger]):
        """
        A removed charger does not affect a vehicle in case the removed charger was not part of the vehicles charge
        schedule
        """
        for route_index, sol_rep in self.route_networks.items():

            # if solution is not saved, we need to recompute even if it was not necessary :)
            if isinstance(sol_rep, int):
                continue

            # determine if the removed charger was part of a vehicle schedule
            affected = False

            # for static segments (i.e., given charger is static charger)
            if charger[0]==charger[1] :
                # if any charger on the route corresponds to the removed one we can immediately reject unaffectedness
                # chargers are identified via the IR VertexID
                chargers_on_route = [p.id for p in sol_rep.itineraries[0].route if p.is_static_charger]
                if charger[0] in chargers_on_route:
                    affected = True

            # dynamic case (i.e., given charger is a dynamic segment)
            if charger[0]!=charger[1]:
                # Dynamic chargers are identified via a synthetic Point in the Solution Representation that carries
                # as ID the IR Charger ID (its ressources are only interpolated)
                chargers_on_route = [p.id for p in sol_rep.itineraries[0].route if p.is_synthetic_dyn_charger_representation]
                if charger[2].id in chargers_on_route:
                    affected = True

            if not affected:
                # logging.info(f"Route {route_index} is not affected by removing charger {charger[2].id}")
                continue

            # set back cost for route as in this case we do not know
            self.route_networks[route_index] = -1

    def remove_charging_station(self, charger: tuple[str, str, Charger]) -> None:
        """
        Close the given facility (vertex or arc)
        @param self: instance of class 'DecisionVariables'
        @param charger: key 1, key 2, charger object tuple
        @return: None
        """
        # removing means all info about feasibility is void
        # Vertex Charger
        if charger[0] == charger[1]:
            if VertexID(charger[0]) in self.vertex_chargers:
                del self.vertex_chargers[VertexID(charger[0])]
        # Arc Charger
        else:
            for segment in self.intermediate_rep.charger_edges_by_charger[charger[2]]:
                if (VertexID(segment[0]), VertexID(segment[1])) in self.arc_chargers.keys():
                    del self.arc_chargers[(VertexID(segment[0]), VertexID(segment[1]))]
        self.check_routes_affected_by_closed_charger(charger)

    def remove_single_segment(self, charger: tuple[str, str, Charger]) -> None:
        """
        Remove the given dynamic segment from the configuration
        @param self: instance of class 'DecisionVariables'
        @param charger: key 1, key 2, charger object tuple
        @return: None
        """
        # removing means all info about feasibility is void
        # Arc Charger
        if (VertexID(charger[0]), VertexID(charger[1])) in self.arc_chargers.keys():
            del self.arc_chargers[(VertexID(charger[0]), VertexID(charger[1]))]
        self.check_routes_affected_by_closed_charger(charger)

    def add_charging_station(
            self,
            charger: tuple[str, str, Charger],
            route_networks: Optional[Dict[int, Union[int, SolutionRepresentation]]] = None
    ) -> None:
        """
        Add the given stationary station or dynamic station associated with the given segment
        @param self: instance of class 'DecisionVariables'
        @param charger: key 1, key 2, charger object tuple
        @param route_networks: we may give
        @return: None
        """
        # Vertex Charger
        if charger[0] == charger[1]:
            self.vertex_chargers[VertexID(charger[0])] = charger[2]
        # Arc Charger (open means all segments are opened)
        else:
            for segment in self.intermediate_rep.charger_edges_by_charger[charger[2]]:
                self.arc_chargers[(VertexID(segment[0]), VertexID(segment[1]))] = charger[2]
        self.check_routes_affected_by_new_charger(charger)
        return None

    def add_single_segment(self, charger: tuple[str, str, Charger]) -> None:
        """
        Add the given dynamic segment
        @param self: instance of class 'DecisionVariables'
        @param charger: key 1, key 2, charger object tuple
        @return: None
        """
        # Arc Charger (open means all segments are opened)
        self.arc_chargers[(VertexID(charger[0]), VertexID(charger[1]))] = charger[2]
        self.check_routes_affected_by_new_charger(charger)

    def furthest_charger(self, target_charger: tuple[str, str, Charger],
                         list_of_chargers: list) -> tuple[str, str, Charger]:
        """
        Compute the furthest charger to the target among a list of chargers
        @param self: DecisionVariables
        @param target_charger: Distance to which charger object is decisive
        @param list_of_chargers: List of charger objects to consider
        @return: the furthest charger
        """
        distances = []
        for charger in list_of_chargers:
            distance = min(self.get_distance(VertexID(target_charger[1]), charger[0]),
                           self.get_distance(charger[1], VertexID(target_charger[0])))
            distances += [(charger, distance)]

        return max(distances, key=lambda item: item[1])[0]

    def _sort_charger_combinations(
            self, combinations: List[Tuple[Tuple[VertexID, VertexID, Charger]]], num_items: int
    ) -> List[Tuple[Tuple[VertexID, VertexID, Charger]]]:
        """Return sorted list by last element in tuples """
        if len(combinations) > num_items:
            num_items = len(combinations)
        combinations = combinations[0:num_items]
        sorted_list = sorted(combinations, key=lambda x: sum(c[2].charging_rate for c in x), reverse=True)
        return sorted_list


