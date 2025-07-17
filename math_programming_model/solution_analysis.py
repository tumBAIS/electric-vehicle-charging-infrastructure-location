# coding: utf-8
import folium
import time
import os
import matplotlib
import osmnx as osm
import networkx as nx
import framework.utils as util
from typing import List, Tuple, Dict
from itertools import pairwise

from framework.intermediate_representation import (
    IntermediateRepresentation,
    Vertex,
    VehicleID,
)
from framework.solution_representation import (
    ArrivalDepartureTimes,
    SolutionRepresentation,
)

NXNode = str


class SolutionMap:
    """Very simple class wrapping folium web map class and defining special plot types for routes, chargers, ..."""

    def __init__(self, g: nx.MultiDiGraph, fol_map: folium.folium.Map):
        self._g = g
        self._map = fol_map
        self._free_colors = ['darkred', 'blue', 'darkgreen', 'lightgreen', 'orange']
        self._color_dict = {}
        self._feature_group = {}
        self._validate()

    def _validate(self):
        assert self.street_network.graph["crs"] == "epsg:4326"
        return None

    @property
    def street_network(self):
        return self._g

    @property
    def feature_groups(self):
        return self._feature_group.values()

    def _chose_color(self):
        color = self._free_colors.pop(0)
        return color

    def _add_feature_group(self, ft_group: folium.FeatureGroup):
        self._map.add_child(ft_group)
        return None

    def _get_or_create_color(self, route: VehicleID):
        if route in self._color_dict.keys():
            return self._color_dict[route]
        else:
            color = self._chose_color()
            self._color_dict[route] = color
            return color

    def get_or_create_feature_group(self, route: VehicleID):
        if route in self._feature_group.keys():
            return self._feature_group[route]
        else:
            ft_group = folium.FeatureGroup(name=f"Vehicle or Route {route}")
            self._feature_group[route] = ft_group
            return ft_group

    def build(self, path_name: str = "."):
        for ft_group in self.feature_groups:
            self._add_feature_group(ft_group)
        folium.LayerControl().add_to(self._map)
        timestring = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        self._map.save(f'{path_name}/{timestring} - solution map.html')
        return None

    def add_stop(self, coord: util.Coordinate, vehicle: VehicleID, name: str, feature_group: folium.FeatureGroup):
        color = self._get_or_create_color(vehicle)
        folium.Marker(
            location=[coord.lat, coord.lon],
            popup=name,
            icon=folium.map.Icon(icon='pause', color=color, prefix="glyphicon"),
        ).add_to(feature_group)
        return None

    def add_depot(self, coord: util.Coordinate):
        folium.Marker(
            location=[coord.lat, coord.lon],
            popup="Depot",
            icon=folium.map.Icon(icon='home', color='gray', prefix="glyphicon"),
        ).add_to(self._map)
        return None

    def add_static_charger(self, coord: util.Coordinate, investment: float):
        if is_inductive:
            popupstr = "Inductive charger"
        else:
            popupstr = "Conductive charger"
        # circle marker to indicate charger in presence of a stop at same location
        folium.CircleMarker(
            location=(coord.lat, coord.lon),
            radius=15,
            color="green",
            fill=True,
            fill_opacity=0.2,
            popup=f"{popupstr} with total construction cost: {investment}",
        ).add_to(self._map)
        # usual marker to make it more specific when looking at the static chargers specifically
        folium.Marker(
            location=[coord.lat, coord.lon],
            popup=f"{popupstr} with total construction cost: {investment}",
            icon=folium.map.Icon(icon='flash', color='green', prefix="glyphicon")
        ).add_to(self._map)
        return None

    def add_dynamic_charger(self, orig_id: str, dest_id: str, **data):
        osm.folium.plot_route_folium(
            self.street_network,
            route=[orig_id, dest_id],
            route_map=self._map,
            color=matplotlib.colors.cnames["green"],
            weight=10,
            **data
        )
        return None

    def add_route(self, route: List[NXNode], vehicle: VehicleID):
        color = self._get_or_create_color(vehicle)  # assign color as already used
        osm.plot_route_folium(
            self.street_network,
            route=route,
            route_map=self.get_or_create_feature_group(vehicle),
            color=matplotlib.colors.cnames[color],
        )
        return None

    def add_schedule(self, schedule: Dict[NXNode, Tuple[ArrivalDepartureTimes]]):
        raise NotImplementedError


def add_route_to_map(
        solution_map: SolutionMap,
        inter_rep: IntermediateRepresentation,
        solution: SolutionRepresentation,
        depot: bool = True,
):
    for itinerary in solution.itineraries:
        vertices: List[Vertex] = []
        for point in itinerary.route:
            if point.is_depot and depot:
                vertices += [inter_rep.depot]
            elif not point.is_depot:
                vertices.append(inter_rep.get_vertex(point.id))
            else:
                continue  # this should only be the point is depot case in which we do not want depots
        shortest_path: List[NXNode] = []
        for v1, v2 in pairwise(vertices):
            assert solution_map.street_network.graph["crs"] == "epsg:4326"
            # Note: when loading the osmnx street network from the intermediate representation, we make sure that each
            # vertex in the intermediate representation has a counterpart in the street network (by id)
            # This is enforced by adding nodes and splitting edges if necessary
            sp = nx.shortest_path(solution_map.street_network, source=v1.id, target=v2.id, weight='length')
            solution_map.add_route(sp, itinerary.vehicle)
    return None


def add_static_chargers_to_map(
        solution_map: SolutionMap,
        inter_rep: IntermediateRepresentation,
        solution: SolutionRepresentation,
):
    for vertex_id in solution.static_invest.keys():
        vertex = inter_rep.get_vertex(vertex_id)
        for charger in vertex.constructible_charger:
            solution_map.add_static_charger(
                coord=vertex.coordinate,
                investment=charger.segment_construction_cost + charger.transformer_construction_cost
            )
    return None


def add_stops_to_map(
        solution_map: SolutionMap,
        inter_rep: IntermediateRepresentation,
        solution: SolutionRepresentation,
):
    for itinerary in solution.itineraries:
        vertices: List[Vertex] = []
        ft_group = solution_map.get_or_create_feature_group(itinerary.vehicle)
        for point in itinerary.route:
            if point.is_stop:
                vertices.append(inter_rep.get_vertex(point.id))
            else:
                continue
        for v in vertices:
            if not v.is_stop:
                continue
            solution_map.add_stop(
                coord=v.coordinate,
                vehicle=itinerary.vehicle,
                name=f"Stop {vertices.index(v)}: {v.name}",
                feature_group=ft_group,
            )
    return None


def add_depots_to_map(
        solution_map: SolutionMap,
        inter_rep: IntermediateRepresentation,
        solution: SolutionRepresentation,
):
    for itinerary in solution.itineraries:
        vertices: List[Vertex] = []
        for point in itinerary.route:
            if not point.is_depot:
                vertices.append(inter_rep.get_vertex(point.id))
            else:
                vertices.append(inter_rep.depot)
        for v in vertices:
            if not v.is_depot:
                continue
            solution_map.add_depot(
                coord=v.coordinate,
            )
    return None


def add_dynamic_charger_to_map(
        solution_map: SolutionMap,
        inter_rep: IntermediateRepresentation,
        solution: SolutionRepresentation,
):
    for arc_id, chargers in solution.dynamic_invest.items():
        if arc_id[0] in solution_map.street_network.nodes:
            orig = arc_id[0]
        else:
            v = inter_rep.get_vertex(arc_id[0])
            orig = osm.nearest_nodes(
                solution_map.street_network, Y=v.coordinate.lat, X=v.coordinate.lon
            )
        if arc_id[1] in solution_map.street_network.nodes:
            dest = arc_id[1]
        else:
            v = inter_rep.get_vertex(arc_id[1])
            dest = osm.nearest_nodes(
                solution_map.street_network, Y=v.coordinate.lat, X=v.coordinate.lon
            )
        solution_map.add_dynamic_charger(orig, dest)
    return None
