# coding=utf-8
import warnings
import csv
import os
import random

import xml.etree.ElementTree as ET

from itertools import product
from typing import List

from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.model import Model
from docplex.mp.linear import Var
from docplex.mp.constants import EffortLevel
from docplex.mp.solution import SolveSolution
from docplex.mp.progress import *


from .network import *
from .solution import Route, Solution

from framework.solution_representation import parse_solution_representation_to_warmstart


random.seed(42)


def _progress_data_to_dict(progress_data):
    return {
        "current_objective": progress_data.current_objective,
        "best_bound": progress_data.best_bound,
        "mip_gap": progress_data.mip_gap,
        "current_nb_nodes": progress_data.current_nb_nodes,
        "current_nb_iterations": progress_data.current_nb_iterations,
        "remaining_nb_nodes": progress_data.remaining_nb_nodes,
        "time": progress_data.time,
        "det_time": progress_data.det_time
    }


@dataclass
class DynamicChargerLRPParameters:
    battery_capacity_in_kwh: float
    # Multipliers for min/max battery capacity. 0 <= ... <= 1
    min_soc_as_share: float
    max_soc_as_share: float
    initial_soc_as_share: float
    # Energy price
    energy_price_per_kwh: float
    # consumption cost
    consumption_cost_per_kwh: float
    # Time step for the time discrete math_programming_model
    time_step_in_seconds: float

    def __init__(self, battery_capacity_in_kwh: float, min_soc_as_share: float, max_soc_as_share: float,
                 initial_soc_as_share: float, energy_price_per_kwh: float, consumption_cost_per_kwh: float,
                 time_step_in_seconds=None):
        self.battery_capacity_in_kwh = battery_capacity_in_kwh
        self.min_soc_as_share = min_soc_as_share
        self.max_soc_as_share = max_soc_as_share
        self.initial_soc_as_share = initial_soc_as_share
        if isinstance(energy_price_per_kwh, dict):
            self.energy_price_per_kwh = sum(energy_price_per_kwh.values()) / len(energy_price_per_kwh)
        else:
            self.energy_price_per_kwh = energy_price_per_kwh
        self.consumption_cost_per_kwh = consumption_cost_per_kwh
        self.time_step_in_seconds = time_step_in_seconds

    @property
    def kwh_span(self) -> float:
        return (self.max_soc_as_share - self.min_soc_as_share) * self.battery_capacity_in_kwh

    @property
    def min_kwh(self) -> float:
        return self.min_soc_as_share * self.battery_capacity_in_kwh

    @property
    def max_kwh(self) -> float:
        return self.max_soc_as_share * self.battery_capacity_in_kwh

    @property
    def initial_kwh(self):
        return self.initial_soc_as_share * self.battery_capacity_in_kwh


def _convert_seconds_to_hour(s: int) -> float:
    return s/3600


def _construct_route(network: VehicleNetwork):
    successors_arcs = {arc.origin: arc for arc in network.arcs if arc.x.to_bool()}
    route = [network.start_depot]
    accumulated_consumed_energy = [0.0]
    accumulated_recharger_energy = [0.0]
    while route[-1] is not network.end_depot:
        arc = successors_arcs[route[-1]]
        accumulated_consumed_energy.append(accumulated_consumed_energy[-1] + abs(arc.consumption))
        accumulated_recharger_energy.append(
            accumulated_recharger_energy[-1] + arc.delta_charge + arc.target.delta_charge)
        route.append(arc.target)
    return route, accumulated_consumed_energy, accumulated_recharger_energy


class DynamicChargerLRP:
    """
    network is already preprocessed.
        ->
    Route is a sequence of nodes in network.
    """

    def __init__(
        self,
        vehicle_networks: Dict[Vehicle, VehicleNetwork],
        routes: Dict[Vehicle, Route],
        parameters: DynamicChargerLRPParameters,
    ):
        self._param = parameters

        self._vehicle_networks = vehicle_networks
        self._routes = routes

        self._model = Model()

        self.build_model()

    @property
    def _vehicles(self) -> List[Vehicle]:
        return list(self._vehicle_networks.keys())

    @property
    def _arcs(self) -> Iterable[Arc]:
        return {
            arc for network in self._vehicle_networks.values() for arc in network.arcs
        }

    @property
    def _vertices(self) -> Iterable[Vertex]:
        return {
            vertex
            for network in self._vehicle_networks.values()
            for vertex in network.vertices
        }

    @property
    def _static_chargers(self) -> Iterable[Vertex]:
        return set(
            vertex
            for network in self._vehicle_networks.values()
            for vertex in network.static_chargers
        )

    @property
    def _dynamic_chargers(self) -> Iterable[Arc]:
        return set(
            arc
            for network in self._vehicle_networks.values()
            for arc in network.dynamic_chargers
        )

    @property
    def _dynamic_charger_arcs_by_transformer(self) -> Dict[DynamicCharger, set[Arc]]:
        charger_sets = defaultdict(set)
        for a in self._dynamic_chargers:
            charger_sets[a.charger].add(a)
        return charger_sets

    @property
    def _transformers(self) -> Iterable[DynamicCharger]:
        return set(f.charger for f in self._dynamic_chargers if not f.is_dummy)

    def _define_variables(self):
        model = self._model

        #   x^k_ij -> Vehicle k takes (i, j)
        x = model.binary_var_dict(
            product(self._vehicles, self._arcs),
            name=lambda key: f"x_{key[0]}_({key[1].origin},{key[1].target})",
        )

        #   y_i -> Static charger on i?
        y = model.binary_var_dict(self._vertices, name=lambda v: f"y_{v}")

        #   rho^k_i -> Arrival SoC at i
        rho = model.continuous_var_dict(
            product(self._vehicles, self._vertices),
            name=lambda key: f"rho_{key[0]}_{key[1]}",
        )
        #   tau^k_i -> Arrival time at i
        tau = model.continuous_var_dict(
            product(self._vehicles, self._vertices),
            name=lambda key: f"tau_{key[0]}_{key[1]}",
        )
        if self._param.time_step_in_seconds is not None:
            delta_tau = model.integer_var_dict(
                product(self._vehicles, self._vertices),
                name=lambda key: f"delta_tau_{key[0]}_{key[1]}",
            )
        else:
            delta_tau = model.continuous_var_dict(
                product(self._vehicles, self._vertices),
                name=lambda key: f"delta_tau_{key[0]}_{key[1]}",
            )
        #   delta_rho^k_i -> Charge replenished at i
        delta_rho_v = model.continuous_var_dict(
            product(self._vehicles, self._vertices),
            name=lambda key: f"delta_rho_{key[0]}_{key[1]}",
        )

        #   delta_rho^k_i,j -> Charge replenished on i,j
        delta_rho_arc = model.continuous_var_dict(
            product(self._vehicles, self._arcs),
            name=lambda key: f"delta_rho_{key[0]}_({key[1].origin},{key[1].target})",
        )
        #   z_ij -> Dynamic charger on (i, j)?
        z = model.binary_var_dict(
            self._arcs,
            name=lambda arc: f"z_({arc.origin},{arc.target})"
        )

        #   w_f -> Transformer of set of dyn chargers f build?
        self._w = model.binary_var_dict(
            self._transformers,
            name=lambda transformer: f"w_{transformer.irID}"
        )

        # Save in vehicle networks
        for k, network in self._vehicle_networks.items():
            for v in network.vertices:
                if v.is_dummy:
                    v.y = y[v.dummy_of]
                else:
                    v.y = y[v]
                v.rho = rho[k, v]
                v.tau = tau[k, v]
                v.delta_rho = delta_rho_v[k, v]
                v.delta_tau = delta_tau[k, v]

            for arc in network.arcs:
                if arc.is_dummy:
                    arc.z = z[arc.dummy_of]
                else:
                    arc.z = z[arc]
                arc.delta_rho = delta_rho_arc[k, arc]
                arc.x = x[k, arc]


    def _build_objective(self):
        model = self._model
        objective = self._model.linear_expr()

        # Cost of static chargers
        objective += model.sum(
            v.y * v.charger.construction_cost
            for v in self._static_chargers
            if not v.is_dummy
        )
        # Cost of dyn. chargers
        objective += model.sum(
            arc.z * arc.charger.construction_cost_per_km * arc.distance
            for arc in self._dynamic_chargers
            if not arc.is_dummy
        )
        # Cost of transformers
        objective += model.sum(
            self._w[b] * b.transformer_construction_cost for b in self._transformers
        )

        for k, network in self._vehicle_networks.items():
            # Charging cost at edges
            objective += model.sum(arc.delta_rho * self._param.energy_price_per_kwh
                                   for arc in network.arcs)
            # Charging cost at vertices
            objective += model.sum(v.delta_rho * self._param.energy_price_per_kwh
                                   for v in network.vertices)

            # Consumption cost
            objective += model.sum(a.x * a.consumption * self._param.consumption_cost_per_kwh for a in network.arcs)
        return objective

    def _build_vehicle_constraints(
        self, vehicle: Vehicle, network: VehicleNetwork, route: Route
    ):
        model = self._model
        param = self._param
        time_step = param.time_step_in_seconds if param.time_step_in_seconds is not None else 1

        # Flow conservation (non-depots)
        model.add_constraints(
            (
                model.sum(
                    incoming_arc.x for incoming_arc in network.get_incoming_arcs(v)
                )
                - model.sum(
                    outgoing_arc.x for outgoing_arc in network.get_outgoing_arcs(v)
                )
                == 0
                for v in network.vertices
                if not v.is_depot
            ),
            names=f"flow_conservation_{vehicle}",
        )

        # Start depot has outgoing arc
        model.add_constraint(
            model.sum(
                outgoing_arc.x
                for outgoing_arc in network.get_outgoing_arcs(network.start_depot)
            )
            == 1,
            ctname=f"outgoing_arc_start_depot_{vehicle}",
        )

        # Required visits
        model.add_constraints(
            (
                model.sum(
                    incoming_arc.x for incoming_arc in network.get_incoming_arcs(v)
                )
                >= 1
                for v in network.required_visits
            ),
            names=f"required_visit_{vehicle}",
        )

        # Visit sequence
        for u, v in pairwise(route):
            model.add_constraint(u.tau <= v.tau, ctname=f"route_position_{v}")

        # Consumption propagation
        model.add_constraints(
            (
                arc.origin.rho - arc.consumption + arc.delta_rho
                >= arc.target.rho - arc.target.delta_rho - (1 - arc.x) * (param.kwh_span + arc.consumption)
                for arc in network.arcs
            ),
            names=f"propagate_cons_{vehicle}",
        )
        model.add_constraints(
            (
                arc.origin.rho - arc.consumption + arc.delta_rho
                <= arc.target.rho - arc.target.delta_rho + (1 - arc.x) * (param.kwh_span + arc.consumption)
                for arc in network.arcs
            ),
            names=f"propagate_cons_eq_{vehicle}",
        )

        # Time propagation
        model.add_constraints(
            (
                arc.origin.tau + arc.travel_time_seconds
                <= arc.target.tau - time_step * arc.target.delta_tau + (1 - arc.x) * network.latest_arrival_time
                for arc in network.arcs
            ),
            names=f"propagate_time_{vehicle}",
        )

        # Charge limit
        model.add_constraints(
            (v.rho - v.delta_rho >= param.min_kwh for v in network.vertices),
            names=f"min_soc_{vehicle}",
        )
        model.add_constraints(
            (v.rho <= param.max_kwh for v in network.vertices),
            names=f"max_soc_{vehicle}",
        )

        # Time window
        model.add_constraints(
            (v.tau >= v.time_window_begin for v in network.vertices),
            names=f"time_window_begin_{vehicle}",
        )
        model.add_constraints(
            (v.tau <= v.time_window_end for v in network.vertices),
            names=f"time_window_end_{vehicle}",
        )
        model.add_constraints(
            (v.delta_tau == 0 for v in network.vertices if v.is_auxiliary),
            names=f"no_stopover_at_segment_nodes_{vehicle}",
        )

        # Initial time
        model.add_constraint(
            network.start_depot.tau == network.start_depot.time_window_begin, ctname=f"init_time_{vehicle}"
        )

        # Initial SoC
        model.add_constraint(
            network.start_depot.rho == param.initial_kwh, ctname=f"init_soc_{vehicle}"
        )

        # Charging at static charger
        model.add_constraints(
            (v.delta_rho == _convert_seconds_to_hour(time_step) * v.delta_tau * v.charging_rate
             for v in network.vertices if v.can_construct_charger),
            names=f"charging_static_rate_{vehicle}",
        )

        model.add_constraints(
            (v.delta_rho <= v.y * param.kwh_span for v in network.vertices),
            names=f"charging_static_open_{vehicle}",
        )

        model.add_constraints(
            (v.y == 0 for v in network.vertices if not v.can_construct_charger),
            names="no_stat_chargers_at_non_candidate_stations"
        )

        model.add_constraints(
            (a.z == 0 for a in network.arcs if not a.can_construct_charger),
            names="no_dyn_chargers_at_non_candidate_stations"
        )

        model.add_constraints(
            (
                arc.delta_rho <= arc.travel_time * arc.charging_rate * arc.x
                for arc in network.arcs
            ),
            names=f"charging_dynamic_decision_{vehicle}",
        )

        model.add_constraints(
            (arc.delta_rho <= arc.z * param.kwh_span for arc in network.arcs),
            names=f"charging_dynamic_open_{vehicle}",
        )

    def _build_station_investment_constraints(self):
        model = self._model

        charger_sets = self._dynamic_charger_arcs_by_transformer

        # make sure that fix cost share from transformers occurs if a dyn segment is constructed
        model.add_constraints(
            (model.sum(f.z for f in charger_sets[b]) <= len(charger_sets[b]) * self._w[b]
            for b in self._transformers),
            names="transformer_construction"
        )

    def build_model(self) -> Model:
        model = self._model

        self._set_parameters()

        self._define_variables()

        objective = self._build_objective()
        model.minimize(objective)

        self._build_station_investment_constraints()

        for k, network in self._vehicle_networks.items():
            self._build_vehicle_constraints(k, network, self._routes[k])

        return model

    def _set_parameters(self):
        self._model.context.cplex_parameters.threads = 1
        self._model.parameters.preprocessing.presolve = 0
        self._model.parameters.lpmethod = 0
        self._model.parameters.mip.tolerances.mipgap = 0.01

    def _construct_solution(self) -> Solution:

        routes = {}
        consumed_energy_per_vehicle = {}
        recharger_energy_per_vehicle = {}
        for k, network in self._vehicle_networks.items():
            route, accumulated_consumed_energy, accumulated_recharger_energy = _construct_route(network)
            routes[k] = route
            consumed_energy_per_vehicle[k] = accumulated_consumed_energy
            recharger_energy_per_vehicle[k] = accumulated_recharger_energy

        investments = {x for x in self._static_chargers if x.y.to_bool() and not x.is_dummy} | {
            x for x in self._dynamic_chargers if x.z.to_bool() and not x.is_dummy
        }
        consumed_energy = sum(
            arc.consumption for k,network in self._vehicle_networks.items() for arc in network.arcs if arc.x.to_bool()
        )
        recharged_energy = sum(
            arc.delta_charge for k,network in self._vehicle_networks.items() for arc in network.arcs if arc.x.to_bool()
        )
        recharged_energy += sum(
            v.delta_charge for k, network in self._vehicle_networks.items() for v in network.vertices
        )
        return Solution(
            investments=investments,
            routes=routes,
            global_cost=self._model.solution.get_objective_value(),
            consumed_energy=consumed_energy,
            recharged_energy=recharged_energy,
            consumed_energy_per_vehicle=consumed_energy_per_vehicle,
            recharger_energy_per_vehicle=recharger_energy_per_vehicle
        )

    def optimize(self, time_limit, results_file, warmstart: Optional[str]=None) -> Optional[Solution]:
        if time_limit is not None:
            self._model.set_time_limit(time_limit)

        progress = ProgressDataRecorder(clock=ProgressClock.All)
        self._model.add_progress_listener(progress)

        if warmstart is not None:

            # Some parameters
            self._model.parameters.mip.display.set(4)
            self._model.parameters.mip.limits.repairtries.set(3)  # Enable repair mode

            # Parse the passed warmstart file (mst (cplex export) and json (solution representation) format accepted)
            warmstart_values = parse_to_warmstart(
                warmstart=warmstart, var_list=self._model.iter_binary_vars(), timestep=self._param.time_step_in_seconds
            )

            # Pass the warm start solution
            # The DoCplex API requires a SolveSolution object which cannot be saved directly from the reduced model
            # Hence we manually make sure that the variables in .mst file are a valid subset and parse manually to
            # a SolveSolution
            mip_start_solution = SolveSolution(self._model, warmstart_values)
            self._model.add_mip_start(mip_start_solution, effort_level=EffortLevel.Auto)


        sol = self._model.solve(log_output=True)
        progress_data = progress.iter_recorded

        if results_file is not None and progress_data:
            # Create the directory if it doesn't exist
            os.makedirs(results_file, exist_ok=True)

            # Define the fieldnames (header) based on the keys of the dictionary
            fieldnames = ["current_objective", "best_bound", "mip_gap", "current_nb_nodes", "current_nb_iterations",
                          "remaining_nb_nodes", "time", "det_time"]

            # Convert progress_data objects to dictionaries
            progress_data_dicts = [_progress_data_to_dict(pd) for pd in progress_data]

            with open(f"{results_file}/progress_data.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(progress_data_dicts)
        if sol is None and self._model.solve_details.status_code == 103:
            refiner = ConflictRefiner()
            conflicts = refiner.refine_conflict(self._model, display=True)

            for conf in conflicts:
                print(conf.element)

            return None
        elif sol is None:
            return None
        return self._construct_solution()


def parse_mst_to_warmstart(warmstart: str, var_list: List[Var]) -> Dict[str,int]:
    # Parse the .mst file as XML
    tree = ET.parse(warmstart)
    root = tree.getroot()

    # Extract variable names from the warmstart
    warmstart_vars = set()
    warmstart_values = dict()
    for var in root.findall(".//variable"):
        warmstart_vars.add(var.get("name"))
        warmstart_values[var.get("name")] = int(float(var.get("value")))

    return warmstart_values


def parse_to_warmstart(warmstart: str, var_list: List[Var], timestep: int) -> Dict[str, int]:
    if warmstart.split(".")[-1] == "mst":
        warmstart_values = parse_mst_to_warmstart(warmstart=warmstart, var_list=var_list)
    elif warmstart.split(".")[-1] == "json":
        warmstart_values = parse_solution_representation_to_warmstart(warmstart=warmstart, var_list=var_list, time_step=timestep)
    else:
        raise ValueError
    return warmstart_values
