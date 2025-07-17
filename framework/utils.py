# coding: utf-8
from __future__ import annotations

import math
# import pytz
import uuid
import warnings
import itertools

# import osmnx as osm
# import pandas as pd
import networkx as nx
import gtfs_kit as gk
import pyproj as pj
import random as rd

from datetime import datetime as dt
from datetime import timedelta as td

from functools import partial, cache
from dataclasses import dataclass
from typing import Tuple, List, Any, Union, Optional, overload, Iterable

# import shapely.geometry
from pyproj import CRS, Proj, Transformer
from shapely.geometry import MultiPolygon, Point, Polygon, LineString
from shapely.ops import transform

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# some type aliases
OSMNetwork = nx.MultiDiGraph
OSMArcID = tuple[str, str, int]

# we make this globally accessible
wgs84 = Proj(proj="latlong", datum="WGS84")  # WGS84
utm = Proj(proj="utm", zone=33, datum="WGS84")  # Example UTM Zone 33, adjust as needed
transformer = Transformer.from_proj(wgs84, utm)

class ValidationError(BaseException):
    pass

class ProblemInfeasibleError(BaseException):
    pass


@dataclass(frozen=True)
class Coordinate:
    """
    Coordinate in WGS84
    """
    lat: float
    lon: float

    def __eq__(self, other):
        return self.lat == other.lat and self.lon == other.lon

    def __add__(self, second):
        if isinstance(second, Coordinate):
            return Coordinate(self.lat + second.lat, self.lon + second.lon)
        elif isinstance(second, int):
            return Coordinate(self.lat + second, self.lon + second)
        elif isinstance(second, float):
            return Coordinate(self.lat + second, self.lon + second)

    def __sub__(self, second):
        if isinstance(second, Coordinate):
            return Coordinate(self.lat - second.lat, self.lon - second.lon)
        elif isinstance(second, int):
            return Coordinate(self.lat - second, self.lon - second)
        elif isinstance(second, float):
            return Coordinate(self.lat - second, self.lon - second)


    def __mul__(self, second):
        if isinstance(second, Coordinate):
            return Coordinate(self.lat * second.lat, self.lon * second.lon)
        elif isinstance(second, int):
            return Coordinate(self.lat * second, self.lon * second)
        elif isinstance(second, float):
            return Coordinate(self.lat * second, self.lon * second)

    def convert_coordinate_system(self) -> Coordinate:
        """
        Converts WGS84 (lat, lon) to UTM coordinates (x, y).
        :param coord: Coordinate in WGS84
        :return: Coordinate in UTM (x, y)
        """
        # Set the UTM projection (proj4 string or EPSG code)
        if -90 <= self.lat <= 90 and -180 <= self.lon <= 180:
            x, y = transformer.transform(self.lon, self.lat)
            return Coordinate(x, y)
        return Coordinate(self.lat, self.lon)


def concatenate_unordered_line_strings(line_strings: Iterable[LineString], first_string: Optional[LineString] = None) -> LineString:
    remaining_segments = list(line_strings)
    if first_string is None:
        concatenated_string = list(remaining_segments.pop().coords)
    else:
        assert first_string in remaining_segments
        concatenated_string = list(first_string.coords)
        remaining_segments.remove(first_string)

    prev_len = len(remaining_segments)
    while len(remaining_segments) > 0:
        # Look for next matching line string
        for next_string in remaining_segments:
            if next_string.coords[0] == concatenated_string[0]:
                # Reverse next string and prepend
                concatenated_string = list(itertools.chain(reversed(next_string.coords), concatenated_string))
            elif next_string.coords[0] == concatenated_string[-1]:
                # Extend
                concatenated_string.extend(next_string.coords)
            elif next_string.coords[-1] == concatenated_string[0]:
                # Prepend
                concatenated_string = list(itertools.chain(next_string.coords, concatenated_string))
            elif next_string.coords[-1] == concatenated_string[-1]:
                # Reverse next string and extend
                concatenated_string.extend(reversed(next_string.coords))
            else:
                continue
            remaining_segments.remove(next_string)
            break
        if prev_len == len(remaining_segments):
            raise ValueError
        prev_len = len(remaining_segments)
    return LineString(concatenated_string)


def convert_gtfs_locations_to_list(stops: pd.DataFrame) -> List[Coordinate]:
    """
    Convert dataframe extracted from gtfs feed into simple locations represented by (lon, lat) pairs - conserving order
    :param stops: Dataframe containing required columns '' and ''
    :return: Stop locations as pairs of (lon, lat)
    """
    assert {
        "stop_id",
        "arrival_time",
        "departure_time",
        "stop_lat",
        "stop_lon",
    }.issubset(stops.columns), "Missing column in given stop list."

    stop_location_list = []
    for stop in stops.iloc:
        stop_location_list.append(Coordinate(lat=stop.stop_lat, lon=stop.stop_lon))

    return stop_location_list


def _buffer_points_to_polygon(lat: float, lng: float, radius: int, utm_zone: str) -> Polygon:
    """
    Convert point into polygon representing a buffer around the point (from https://gist.github.com/joaofig/4a68db62ba1b9a7049d2eb50571ec9bd)
    :param lng: longitude
    :param lat: latitude
    :param radius: radius of buffer in meters
    :param utm_zone: string describing the crs of relevant utm time zone
    :return: Polygon representing the buffered location
    """
    proj_meters = pj.Proj(utm_zone)
    proj_latlng = pj.Proj("epsg:4326")

    project_to_meters = partial(pj.transform, proj_latlng, proj_meters)
    project_to_latlng = partial(pj.transform, proj_meters, proj_latlng)

    pt_latlng = Point(
        lng, lat
    )  # reverse order of lat and lon (https://pyproj4.github.io/pyproj/stable/api/proj.html)

    pt_meters = transform(project_to_meters, pt_latlng)

    buffer_meters = pt_meters.buffer(radius)
    buffer_latlng = transform(project_to_latlng, buffer_meters)
    return buffer_latlng


def transform_coordinate_list_to_buffered_list(stops: List[Coordinate], radius: int = 300) -> List[Polygon]:
    """
    Add some buffer around the stops
    :param stops: List of (lon, lat) pairs representing the stops
    :param radius: radius of the buffer in meters, defaults to 300
    :return: List of polygons representing the buffered stop locations
    """
    utm_zone = get_utm_zone(stops)
    buffered_stops = []
    for s in stops:
        buffered_stops.append(_buffer_points_to_polygon(s.lat, s.lon, radius, utm_zone))
    return buffered_stops

@overload
def project_locations_into_avg_utm_zone(location: Coordinate) -> Point:
    pass

@overload
def project_locations_into_avg_utm_zone(location: list[Coordinate]) -> list[Point]:
    pass

def project_locations_into_avg_utm_zone(locations: List[Coordinate] | Coordinate) -> List[Point] | Point:
    """
    Projects station (lat, lon) to utm crs
    :param locations: List of locations
    :return: CRS locations of type shapely.geometry.Point
    """

    utm_zone = get_utm_zone(locations)
    pj_utm = pj.Proj(utm_zone)

    if isinstance(locations, Coordinate):
        return Point(pj_utm(locations.lon, locations.lat))
    points = []
    for s in locations:
        points.append(
            Point(pj_utm(s.lon, s.lat))
        )  # reverse order of lat and lon for pyproj

    return points


def get_utm_zone(locations: Union[List[Coordinate], Coordinate]) -> str:
    """Generate crs representing string for average longitude in input series
    :param locations: List of (lon, lat) pairs representing locations; or single coordinate
    :return: proj.4 string representing hte projection of lon, lat into utm system
    """
    if isinstance(locations, Coordinate):
        locations = [locations]
    mean_lon = sum(loc.lon for loc in locations) / (len(locations))
    utm_int = int(math.floor((mean_lon + 180) / 6) + 1)
    return f"+proj=utm +zone={utm_int} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"


def extract_osmnx_network(stops: List[Polygon]) -> OSMNetwork:
    """
    Load exactly the subgraph from osmnx that covers the stops
    :param stops: List of polygons representing the buffered stop locations
    :return: networkx DiGraph representing the convex hull of all given locations
    """
    osmnx_graph = osm.graph_from_polygon(
        MultiPolygon(stops).convex_hull,
        network_type="drive",
        truncate_by_edge=True,
        simplify=True,
    )

    # assert graph is weakly connected
    assert nx.is_weakly_connected(osmnx_graph), "Graph graph is not weakly connected!"

    return osmnx_graph


def distance_euclidean(_from: Coordinate, _to: Coordinate) -> float:
    """
    Calculates the distance between two positions (in utm crs) pairs in given unit
    :param _from: Origin location
    :param _to: Target location
    :return: Distance between the two locations in m
    """
    # the conversion checks the values (if they are within the WGS84 range) and in case they are spheric, we convert to
    # utm zone 33 (berlin); if not the coordinate remains the same (i.e., in UTM)
    _from = _from.convert_coordinate_system()
    _to = _to.convert_coordinate_system()
    return math.sqrt((_from.lat - _to.lat) ** 2 + (_from.lon - _to.lon) ** 2)


def read_geometry(osm_network: OSMNetwork, edge: Tuple[Any]) -> LineString:
    """
    If edge contains geometry object, return it (will be in same crs). If not, construct default geometry in network crs
    :param osm_network: Graph from osmnx
    :param edge: Tuple (u, v, k) identifying the relevant edge in network
    :return: Geometry of edge
    """
    try:
        edge_geometry = osm_network.edges[edge]["geometry"]
    except KeyError:
        edge_geometry = LineString(
            [
                Point(osm_network.nodes[edge[0]]["x"], osm_network.nodes[edge[0]]["y"]),
                Point(osm_network.nodes[edge[1]]["x"], osm_network.nodes[edge[1]]["y"]),
            ]
        )
    return edge_geometry


def reverse_edge_data(edge_data: dict) -> dict:
    reversed_data = edge_data.copy()
    if (geometry := reversed_data.get('geometry')) is not None:
        if isinstance(geometry, LineString):
            reversed_data['geometry'] = LineString(reversed(geometry.coords))
        elif isinstance(geometry, Point):
            reversed_data['geometry'] = Point(reversed(geometry.coords))  # this case is not intended but since function only reverses the geometry we allow it here
        else:
            raise ValueError("Geometry object is not of type Linestring or Point.")
    return reversed_data


def split_edge(edge_data: dict, split_at_location: Point) -> tuple[dict, dict]:
    if "geometry" not in edge_data.keys():
        print("Fds")
    first_segment_data = extract_osm_edge_data_from_segment(
        edge_data, start_point=Point(edge_data['geometry'].coords[0]), end_point=split_at_location)
    second_segment_data = extract_osm_edge_data_from_segment(
        edge_data, start_point=split_at_location, end_point=Point(edge_data['geometry'].coords[-1]))

    return first_segment_data, second_segment_data


def add_split_edge(osm_network: OSMNetwork, i, j, edge_data: dict, split_at_location: Point, split_point_name: str, replace=False, bidir=False) \
        -> list[OSMArcID, OSMArcID]:
    first_segment_data, second_segment_data = split_edge(edge_data, split_at_location)

    if replace:
        osm_network.remove_edge(i, j)
        if bidir:
            osm_network.remove_edge(j, i)

    osm_network.add_node(split_point_name, x=split_at_location.x, y=split_at_location.y)

    osm_network.add_edge(i, split_point_name, key=0, **first_segment_data)
    osm_network.add_edge(split_point_name, j, key=0, **second_segment_data)
    if bidir:
        osm_network.add_edge(split_point_name, i, key=0, **reverse_edge_data(first_segment_data))
        osm_network.add_edge(j, split_point_name, key=0, **reverse_edge_data(second_segment_data))

    return [(i, split_point_name, 0), (split_point_name, j, 0)] + ([] if not bidir else [(split_point_name, i, 0), (j, split_point_name, 0)])


def _add_edge(
        osm_network: OSMNetwork,
        origin: str,
        target: str,
        dist: float,
        bidirectional: bool = True,
        **data,
) -> None:
    """
    Adds an edge (including it's reversed edge) to the network
    :param osm_network: osmnx street network
    :param origin: The id of the origin node
    :param target: The id of the target node
    :param dist: The length of the newly added edge, in meters
    :param bidirectional: if to add edges for both directions
    :param data: Any additional edge attributes
    """
    # Add origin -> dest edge
    osm_network.add_edge(
        origin,
        target,
        length=dist,
        **data,
    )
    if bidirectional:
        reverse_data = data.copy()
        if 'geometry' in reverse_data:
            reverse_data['geometry'] = shapely.geometry.LineString(reversed(reverse_data['geometry'].coords))
        osm_network.add_edge(
            target,
            origin,
            length=dist,
            **reverse_data,
        )
    return None


def _remove_edge(
        osm_network: OSMNetwork,
        origin: str,
        target: str,
        bidirectional: bool = True,
) -> None:
    """
    Removes an edge (including it's reversed edge) from a network
    :param osm_network: osmnx street network
    :param origin: The id of the origin node
    :param target: The id of the target node
    :param bidirectional: if to add edges for both directions
    """

    # Add origin -> dest edge
    osm_network.remove_edge(
        origin,
        target,
    )
    if bidirectional:
        osm_network.remove_edge(
            target,
            origin,
        )
    return None


def isfloat(num):
    try:
        float(num)
        return True
    except (ValueError, TypeError):
        return False


def extract_osm_edge_data_from_segment(
        osm_edge_data: dict,
        start_point: shapely.geometry.Point,
        end_point: shapely.geometry.Point,
        default_maxspeed: float = 50.0
):
    # start/end should be projected according to the underlying network
    edge_geometry = osm_edge_data['geometry']
    edge_max_speed = osm_edge_data.get('maxspeed', default_maxspeed)
    if isinstance(edge_max_speed, list) or isinstance(edge_max_speed, tuple):
        edge_max_speed = default_maxspeed
    assert isfloat(edge_max_speed)

    start_offset = edge_geometry.project(start_point, normalized=False)
    end_offset = edge_geometry.project(end_point, normalized=False)
    if start_offset > end_offset:
        start_point, end_point = (end_point, start_point)
        start_offset, end_offset = (end_offset, start_offset)
    assert start_offset <= end_offset

    segment_geometry = shapely.ops.substring(
        edge_geometry, start_dist=start_offset, end_dist=end_offset, normalized=False
    )
    segment_length = end_offset - start_offset
    segment_max_speed = edge_max_speed

    base_data = osm_edge_data.copy()
    base_data.update({
            'geometry': segment_geometry,
            'length': segment_length,
            'maxspeed': segment_max_speed,
    })

    return base_data


def add_node_via_split(
        osm_network: OSMNetwork,
        location: Coordinate,
        name: Optional[str] = None,
        default_maxspeed: float = 50.0,
        **data) -> Tuple[list[str], list[OSMArcID]]:
    """
    Call add_nodes for all routes given
    :param osm_network: Street network in its osmnx representation in utm crs
    :param location: Coordinate of node to add to network
    :param name: Name of node to add to network
    :param default_maxspeed: Default value if maxspeed attribute of splitted edge is None
    :param data: Dictionary containing additional attributes for assignment to node to add to network
    """

    # set default name for new node
    if name is None:
        name = str(uuid.uuid4())

    # assert graph crs and project given location into utm crs
    target_crs = get_utm_zone([location])
    projected_location = project_locations_into_avg_utm_zone([location])[0]
    assert (
        osm_network.graph["crs"].to_epsg() == CRS.from_proj4(target_crs).to_epsg()
    ), "Unexpected network crs."

    # assert no node exist at same location already
    nearest_node, dist_to_node = osm.nearest_nodes(
        osm_network,
        projected_location.x,
        projected_location.y,
        return_dist=True
    )

    if dist_to_node <= 1:
        warnings.warn(f"Node {nearest_node} already exists; and {name} will not be created individually.")

    if dist_to_node <= 1:
        return [], []

    # find nearest edge and assert no ties (i.e., type is tuple and not list of tuples)
    nearest_edge, dist_to_edge = osm.nearest_edges(
        osm_network,
        projected_location.x,
        projected_location.y,
        return_dist=True
    )
    assert (
        isinstance(nearest_edge, Tuple)
    ), f"Multiple edges are equally close to location f{name} and split cannot be performed."
    nearest_edge_data = osm_network.get_edge_data(*nearest_edge)

    # check if edge that we split is a bidirectional edge and extract relevant information
    bidirectional = osm_network.has_edge(nearest_edge[1], nearest_edge[0])
    max_speed = osm_network.edges[nearest_edge].get("maxspeed", default_maxspeed)
    edge_geometry = read_geometry(osm_network, nearest_edge)

    # get point to cut edge and add a node to the network
    halfway_point = edge_geometry.interpolate(edge_geometry.project(projected_location))
    halfway_point_name = (
        f"{nearest_edge[0]}-{name}-{nearest_edge[1]}"
    )
    # check if the halfway point coincides with an existing point on the edge
    # don't need to split if there exists
    need_split = True
    for node_id in nearest_edge[:2]:
        node_on_edge_point = Point(osm_network.nodes[node_id]['x'], osm_network.nodes[node_id]['y'])
        dist_to_endpoint = node_on_edge_point.distance(halfway_point)
        if dist_to_endpoint > 1:
            continue
        else:
            # print(f"{node_id} is only {dist_to_endpoint} m away")
            need_split = False
            halfway_point = node_on_edge_point
            halfway_point_name = node_id
            break

    # Add a node for the stop
    osm_network.add_node(
        name,
        x=projected_location.x,
        y=projected_location.y,
        **data,
    )

    # Replace the original edge
    added_edges = []
    if need_split:
        if "geometry" not in nearest_edge_data.keys():
            # fake shapely geometry added to arc (simple straight line)
            nearest_edge_data['geometry'] = shapely.geometry.LineString(
                ([(osm_network.nodes[nearest_edge[0]]["x"], osm_network.nodes[nearest_edge[0]]["y"]),
                  (osm_network.nodes[nearest_edge[1]]["x"], osm_network.nodes[nearest_edge[1]]["y"])])
            )
        added_edges = add_split_edge(osm_network, nearest_edge[0], nearest_edge[1], nearest_edge_data,
                       split_at_location=halfway_point, split_point_name=halfway_point_name, replace=True, bidir=bidirectional)

    # Add Edge from halfway point to projected location
    # If node exists already, the halfway point will be the existing node --> works
    _add_edge(
        osm_network,
        origin=halfway_point_name,
        target=name,
        dist=distance_euclidean(
            Coordinate(lat=halfway_point.x, lon=halfway_point.y),
            Coordinate(lat=projected_location.x, lon=projected_location.y)
        ),
        bidirectional=True,  # Must be true, otherwise we cannot return from this node
        maxspeed=max_speed,
        geometry=LineString([halfway_point, projected_location])
    )
    added_edges.extend([(halfway_point_name, name, 0), (name, halfway_point_name, 0)])

    # construct return lists
    added_node_ids = [halfway_point_name, name]

    return added_node_ids, added_edges


def project_on_arcs(
        osm_network: OSMNetwork,
        start_location: Coordinate,
        end_location: Coordinate,
        include_reverse_arcs: bool=False
) -> set[tuple[str, str]]:
    """
    Find arcs in network (utm reference) that potentially cover the dynamic charger endpoints (in lat, lon)
    :return list of two or more vertices that describes one (or more consecutive) arcs
    """

    if include_reverse_arcs:
        raise ValidationError("Do not set this setting to true, it might induce inconsistent behaviour as of now")

    def _get_closest_endpoint(*node_ids: str | int, to:tuple[float, float], projected_network: OSMNetwork):
        return min(node_ids,
                   key=lambda node_id: distance_euclidean(
                       Coordinate(projected_network.nodes[node_id]["x"], projected_network.nodes[node_id]["y"]),
                       Coordinate(*to)))

    target_crs = get_utm_zone([start_location])

    # Look for nearest arc at start location
    assert (
            osm_network.graph["crs"].to_epsg() == CRS.from_proj4(target_crs).to_epsg()
    ), "Unexpected network crs."

    # Project the network
    # project start, end location to nearest edge
    projected_locations = project_locations_into_avg_utm_zone([start_location, end_location])

    arcs, distances = osm.nearest_edges(osm_network, X=[stop.x for stop in projected_locations],
                                          Y=[stop.y for stop in projected_locations], return_dist=True)

    # if no attribute 'geometry', add simple direct line
    for i,j,_ in arcs:
        if "geometry" in osm_network.get_edge_data(i,j,_).keys():
            continue
        else:
            osm_network.get_edge_data(i, j, _)['geometry'] = shapely.geometry.LineString(
                ([(osm_network.nodes[i]["x"], osm_network.nodes[i]["y"]),
                  (osm_network.nodes[j]["x"], osm_network.nodes[j]["y"])])
            )

    # 1. Case: Dynamic charger spans only two nodes (with 1 or 2 arcs - depending on bidirectional case handling)
    if arcs[0] != arcs[1] and arcs[0][:2] == tuple(reversed(arcs[1][:2])) and include_reverse_arcs:
        return {arcs[0][:2], arcs[1][:2]}
    elif arcs[0] != arcs[1] and arcs[0][:2] == tuple(reversed(arcs[1][:2])) and not include_reverse_arcs:
        # pick the direction that is given by relation of start and end location
        # (note: linestrings have an implicit direction along which is measured)
        arc_geometries = list(map(lambda x: osm_network.get_edge_data(x[0], x[1], 0)['geometry'], arcs))
        if arc_geometries[0].project(projected_locations[0]) > arc_geometries[0].project(projected_locations[1]):
            return {arcs[1][:2]}
        else:
            return {arcs[0][:2]}

    for x, dist in zip((start_location, end_location), distances):
        if dist > 5:
            warnings.warn(f"Nearest edge of location {x} is more than 5 meters away ({dist} meters)")

    # 2 Case: Dynamic charger edges span more than 2 nodes
    def _get_path_between_arcs(start_arc_id: int, end_arc_id: int) -> tuple[list, float]:
        # Get the shortest path from the closest endpoint of each edge
        start_node = _get_closest_endpoint(
            arcs[start_arc_id][0],
            arcs[start_arc_id][1],
            to=(projected_locations[start_arc_id].x, projected_locations[start_arc_id].y),
            projected_network=osm_network
        )
        end_node = _get_closest_endpoint(
            arcs[end_arc_id][0],
            arcs[end_arc_id][1],
            to=(projected_locations[end_arc_id].x, projected_locations[end_arc_id].y),
            projected_network=osm_network
        )

        shortest_path = nx.shortest_path(osm_network, start_node, end_node)
        assert shortest_path is not None, f'Could not find a shortest path between {start_node} and {end_node}'

        for v in shortest_path:
            if not v in osm_network.nodes:
                print("debug")

        distance = sum(osm_network.get_edge_data(u, v, 0)['length'] for u, v in itertools.pairwise(shortest_path))

        # May happen if start_node == end_node
        if len(shortest_path) == 1:
            assert start_node == end_node
            # Shortest path should be shorter of both connections
            shortest_path = arcs[start_arc_id][:2] if osm_network.get_edge_data(*arcs[start_arc_id])['length'] \
                else arcs[end_arc_id][:2]

        return shortest_path, distance

    # Path from start->end may be longer than end->start if one-way streets cause detours. Try both and pick shorter one
    shortest_path = min(filter(lambda x: x[0] is not None, (_get_path_between_arcs(start_arc_id, end_arc_id)
                         for start_arc_id, end_arc_id in ((0, 1), (1, 0)))), key=lambda x: x[1])[0]

    charger_arcs = set(itertools.pairwise(shortest_path))

    # Deal with bidirectional arcs - add other direction for any arc where other direction exists in network
    if include_reverse_arcs:
        for x in charger_arcs:
            if (x[1], x[0]) in osm_network.edges:
                charger_arcs.add(x)

    assert all(x in osm_network.edges for x in charger_arcs)

    return charger_arcs


def convert_utm_to_coordinate(loc: Tuple[float, float], utm_zone: str) -> Coordinate:
    """
    Conmvert given locations in utm system into Coordinate object
    :param loc: Location as tuple (lat, lon)
    :param utm_zone: Utm zone of given location
    :return: Coordinate object representing the given location
    """
    p = pj.Proj(utm_zone)
    lon, lat = p(loc[0], loc[1], inverse=True)  # reverse order of lat, lon in pyproj
    return Coordinate(lat=lat, lon=lon)


def clean_osmnx_graph(osmnx_network: OSMNetwork, default_maxspeed: float = 50.0) -> None:
    """Call cleaning functions and apply to osmnx_network"""
    start = len(osmnx_network)
    remove_parallel_edges(osmnx_network)
    enforce_string_node_identifier(osmnx_network)
    ensure_maxspeed(osmnx_network, default_maxspeed=default_maxspeed)
    assert len(osmnx_network) == start


def ensure_maxspeed(osmnx_network, default_maxspeed: float) -> None:
    for *_, data in osmnx_network.edges.data():
        network_speed = data.get('maxspeed')
        if network_speed is None:
            maxspeed = default_maxspeed
        elif isinstance(network_speed, str):
            maxspeed = default_maxspeed
        elif isinstance(network_speed, list):
            # Edge case `network_speed = ['signals', 'none']` must be handled
            try:
                maxspeed = sum(map(float, network_speed)) / len(network_speed)
            except ValueError:
                maxspeed = default_maxspeed
        else:
            raise ValueError(f"Unknown maxspeed type: {type(network_speed)}")
        data['maxspeed'] = maxspeed


def enforce_string_node_identifier(osmnx_network: OSMNetwork) -> None:
    """Enforce only string names in entire further logic and program"""
    mapping = {name: str(name) for name in osmnx_network.nodes}
    nx.relabel_nodes(osmnx_network, mapping=mapping, copy=False)


def remove_parallel_edges(osmnx_network: OSMNetwork) -> None:
    """Remove all circular (u==v) edges and parallel edges based on tie breaking criteria:
    1. keep road with shorter length
    2. tbd --> nothing implemented yet
    :param osmnx_network: osmnx graph (as MultiDiGraph) with edges containing attributes 'length'
    """
    edge_view = list(osmnx_network.edges(data=True))
    for u, v, d in edge_view:
        if u == v:
            osmnx_network.remove_edge(u, v)
            continue
        edge_list = [
            (origin, target, key)
            for (origin, target, key) in osmnx_network.edges
            if (origin == u) & (target == v)
        ]
        if len(edge_list) > 1:
            # create dictionary which assigns each duplicated u,v pair their data attribute dictionary
            data_dicts = {
                key: osmnx_network.get_edge_data(origin, target, key)
                for (origin, target, key) in edge_list
            }
            # find minimum length
            min_length = min([value["length"] for value in data_dicts.values()])
            # get negative, hence dictionary with all duplicated edge keys that should be removed
            data_dicts_delete = {
                key: value
                for key, value in data_dicts.items()
                if value["length"] > min_length
            }
            for key in data_dicts_delete.keys():
                osmnx_network.remove_edge(u, v, key)
        else:
            continue

    # assert no duplicates left
    resulting_edge_view = [(u, v) for (u, v) in osmnx_network.edges(data=False)]
    assert (
            len([e for e in resulting_edge_view if resulting_edge_view.count(e) > 1]) == 0
    )

    return None


def project_node_to_edge(
        osm_network: OSMNetwork,
        location: Coordinate,
        name: str = None,
        **data,
    ) -> Tuple[list[str], list[OSMArcID]]:
    """
    Call add_nodes for all routes given
    :param osm_network: Street network in its osmnx representation in utm crs
    :param location: Coordinate of node to project on arc
    :param name: Name of node to add to network
    """
    # assert graph crs and project given location into utm crs
    target_crs = get_utm_zone([location])
    projected_location = project_locations_into_avg_utm_zone(location)
    assert (
        osm_network.graph["crs"].to_epsg() == CRS.from_proj4(target_crs).to_epsg()
    ), "Unexpected network crs."

    # find nearest edge and assert no ties (i.e., type is tuple and not list of tuples)
    nearest_edge, dist_to_edge = osm.nearest_edges(
        osm_network,
        projected_location.x,
        projected_location.y,
        return_dist=True
    )
    assert (
        isinstance(nearest_edge, Tuple)
    ), f"Multiple edges are equally close to location f{name} and split cannot be performed."
    nearest_edge_data = osm_network.get_edge_data(*nearest_edge)

    if "geometry" not in nearest_edge_data.keys():
        nearest_edge_data['geometry'] = shapely.geometry.LineString(
            ([(osm_network.nodes[nearest_edge[0]]["x"], osm_network.nodes[nearest_edge[0]]["y"]),
              (osm_network.nodes[nearest_edge[1]]["x"], osm_network.nodes[nearest_edge[1]]["y"])])
        )

    # check if edge that we split is a bidirectional edge and extract relevant information
    bidirectional = osm_network.has_edge(nearest_edge[1], nearest_edge[0])
    edge_geometry = read_geometry(osm_network, nearest_edge)

    # get point to cut edge and add a node to the network
    intersection = edge_geometry.interpolate(edge_geometry.project(projected_location))

    # Replace the original edge
    added_edges = add_split_edge(
        osm_network,
        nearest_edge[0],
        nearest_edge[1],
        nearest_edge_data,
        split_at_location=intersection,
        split_point_name=name,
        replace=True,
        bidir=bidirectional
    )

    # construct return lists
    added_node_ids = [name]
    return added_node_ids, added_edges


def serialise_nx_graph(
    G,
    *,
    source="source",
    target="target",
    name="id",
    link="links",
):
    """Returns data in node-link format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph
    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    link : string
        A string that provides the 'link' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If the values of 'source', 'target' and 'key' are not unique.

    Notes
    -----
    Graph, node, and link attributes are stored in this format.  Note that
    attribute keys will be converted to strings in order to comply with JSON.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.


    See Also
    --------
    node_link_graph, adjacency_data, tree_data
    """
    assert not G.is_multigraph()

    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None
    if len({source, target, key}) < 3:
        raise nx.NetworkXError("Attribute names are not unique.")

    data = {"directed": G.is_directed(), "multigraph": False, "graph": G.graph,
            "nodes": [{**d["vertex"].asdict(), name: n} for n, d in G.nodes(data=True)],
            link: [{**d["arc"].asdict(), source: u, target: v} for u, v, d in G.edges(data=True)]}

    return data


def _to_tuple(x):
    """Converts lists to tuples, including nested lists.
    All other non-list inputs are passed through unmodified. This function is
    intended to be used to convert potentially nested lists from json files
    into valid nodes.
    """
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(map(_to_tuple, x))

def _get_first_index(l: list[int], i: int) -> int:
    """This is a helper function to return the first index from an index list that is bigger than the given input"""
    for e in l:
        if e > i:
            return e
    raise ValueError("no such element in index list")


def _get_last_index(l: list[int], i: int) -> int:
    """This is a helper function to return the first index from an index list that is bigger than the given input"""
    for e in reversed(l):
        if e < i:
            return e
    raise ValueError("no such element in index list")


def convert_to_timedelta(t: dt, iterator: int) -> td:
    return td(seconds=(t - dt(1970, 1, 1)).total_seconds()*iterator)


def _generate_random_subsets(original_list: list, num_subsets: int, size: int) -> list[list]:
    subsets = []
    cntr = 0
    while cntr < num_subsets:
        subset = rd.sample(original_list, size)
        if subset not in subsets:
            subsets.append(subset)
            cntr += 1
    return subsets


def sort_tuples(tuples):
    from collections import defaultdict

    # Create a dictionary to map elements to their connections
    connection_map = defaultdict(list)
    for first, second in tuples:
        connection_map[first].append(second)

    # Find the starting point
    all_firsts = [first for first, _ in tuples]
    all_seconds = [second for _, second in tuples]

    start = None
    for first in all_firsts:
        if all_firsts.count(first) > all_seconds.count(first):
            start = first
            break

    # If the start is not found using above logic, find the first unique start
    if start is None:
        unique_firsts = set(all_firsts) - set(all_seconds)
        if unique_firsts:
            start = unique_firsts.pop()
        else:
            raise ValueError("No unique starting point found.")

    # Build the chain
    sorted_tuples = []
    while start in connection_map and connection_map[start]:
        next_start = connection_map[start].pop()
        sorted_tuples.append((start, next_start))
        start = next_start

    return sorted_tuples


def filter_tuple_list(list, start_cut, end_cut) -> List[Tuple[Any]]:
    return list[list.index(next(e for e in list if e[0] == start_cut)): list.index(
                    next(e for e in list if e[1] == end_cut)) + 1]










