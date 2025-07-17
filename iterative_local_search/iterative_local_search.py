import os
import math
import logging
import copy
import gc
import warnings
import random as rd
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict

import pandas as pd

from framework.instance_parameter import InstanceParameters
from framework.staggered_conflict_resolution import resolve_conflict
from framework.solution_representation import SolutionRepresentation
from framework.solver import SolverParameters, Solver
from iterative_local_search.spprc_network import DecisionVariables
from iterative_local_search.conflict import dump_conflicts_to_json, detect_conflict_from_solution
from framework.intermediate_representation import Charger, IntermediateRepresentation, VertexID, ChargerID
from framework.solver_clock import SolverClock
from iterative_local_search.subproblem import (
    best_routing,
    objective_cost_value,
)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
plt.set_loglevel("critical")

rd.seed(42)
np.random.seed(42)

toler = 1e-3


class BinaryTightener:
    decisions: DecisionVariables
    sorted_segment_list: List[Tuple[VertexID, VertexID]]
    best_solution_value: float
    charger: Charger
    best_decisions: Tuple[Dict[VertexID, Charger], Dict[Tuple[VertexID, VertexID], Charger]]
    current_index: int
    upper_bound_index: int
    lower_bound_index: int

    def __init__(
            self, decision: DecisionVariables, segments: List[Tuple[VertexID, VertexID]],
            current_solution: float, charger: Charger):
        self.decisions = decision
        self.sorted_segment_list = segments
        self.best_solution_value = current_solution
        self.charger = charger
        self.best_decisions = decision.copy(retain_route_networks=True)
        self.current_index = len(segments)
        self.upper_bound_index = len(segments)  # denotes the idx such that segment_list[0:idx-1] is feasible
        self.lower_bound_index = -1  # denotes the idx such that segment_list[0:idx] is infeasible

    def has_converged(self) -> bool:
        # the second condition makes sure that lb=0 and ub=1 is not accepted; in this case ub=0 has to be tested
        return self.upper_bound_index - self.lower_bound_index <= 1

    def cut_segments(self, feasible: bool) -> DecisionVariables:
        if feasible:
            self.upper_bound_index = self.current_index
            self.current_index = self.upper_bound_index - math.ceil(self.upper_bound_index/2)
        else:
            self.lower_bound_index = self.current_index
            self.current_index = self.lower_bound_index + math.floor((self.upper_bound_index-self.current_index)/2)
        for i, keys in enumerate(self.sorted_segment_list):
            if i < self.current_index:
                self.decisions.add_single_segment((keys[0], keys[1], self.charger))
            else:
                self.decisions.remove_single_segment((keys[0], keys[1], self.charger))
        return self.decisions

    def update_best_solution(self, new_decision, new_value):
        self.best_decisions = new_decision.copy(retain_route_networks=True)
        self.best_solution_value = new_value


@dataclass(frozen=True)
class ILSParameters(SolverParameters):
    interpreter: str
    multithreading: bool
    ls_num_configs_to_explore: int
    check_conflicts: bool
    restrict_arrival_times: bool = False
    share_repair_iterations: float = 0.1
    max_conflict_resolution_iterations: int=3
    init_opening_stepsize: int=2
    i_opt: int = 10
    strength: int = 2


class ILSSolver(Solver):
    solver_parameters: ILSParameters
    infeasible_configurations: set[tuple[int]]
    visited_configurations: dict[tuple[int], float]
    warmstart_solution: Optional[SolutionRepresentation]
    solver_clock: SolverClock
    number_iterations: int = 0
    run_statistics: List = []

    def __init__(
            self,
            instance_parameters: InstanceParameters,
            solver_parameters: ILSParameters,
            warmstart_solution: Optional[SolutionRepresentation] = None
    ):
        super().__init__(instance_parameters, solver_parameters)
        self.infeasible_configurations = set()
        self.visited_configurations = {}
        self.warmstart_solution = warmstart_solution
        self.solver_clock = SolverClock(self.solver_parameters.run_time_in_seconds, start_time=None)
        self.incumbent = math.inf
        self.incumbent_configuration = ({}, {})

    def _hash_config(self):
        return hash(tuple(frozenset(d.items()) for d in list(self.incumbent_configuration)[0:-1]))

    def _report_run_statistics(self, path):
        """Write out csv file with progress data of ILS"""
        df = pd.DataFrame(
            self.run_statistics,
            columns= ["Iteration", "LS Steps", "Time(s)", "Best Solution", "Current Solution", "Perturbed Solution", "Perturbation Strength", "Infeasible Conf.", "Cached Conf."]
        )
        os.makedirs(path, exist_ok=True)
        df.to_csv(path + "/progress_data.csv", encoding='utf-8')

    def _report_conflicts(self, path, sol_rep: SolutionRepresentation):
        os.makedirs(path, exist_ok=True)
        conflicts = detect_conflict_from_solution(sol_rep)
        dump_conflicts_to_json(conflicts, path + "/conflicts.json")

    def _is_conflict_repair_iteration(self) -> bool:
        """
        This method returns boolean indicating if we allow infeasibility w. r. to conflicts and try to repair them after
        the local search. The logic can be replaced. Currently 10% of iterations are targeted, i.e., every 10th iteration
        we allow the conflicts and invoke the resolution algorithm.
        """
        return self.number_iterations % math.floor(1 / self.solver_parameters.share_repair_iterations) == 0

    def solve(self, intermediate_rep: IntermediateRepresentation) -> Optional[SolutionRepresentation]:

        logging.info(f"######################## Start ILS ########################")
        self.solver_clock.start()

        warmstart_failed = False
        if self.warmstart_solution is not None:
            decision = DecisionVariables(
                intermediate_rep, {}, {}, self.instance_parameters.max_speed(intermediate_rep.routes[0].vehicle_id),
                self.instance_parameters.max_consumption(intermediate_rep.routes[0].vehicle_id),
                self.instance_parameters.soc_init, self.instance_parameters.soc_max, self.instance_parameters.soc_min,
                self.instance_parameters.energy_prices, self.instance_parameters.consumption_cost
            )
            decision.set_configuration(
                v_chargers={
                    si: next(iter(intermediate_rep.get_vertex(si).constructible_charger))
                    for si, sol_charger in self.warmstart_solution.static_invest.items()
                },
                a_chargers={
                    (si.split("-")[0], si.split("-")[1]): next(
                        iter(intermediate_rep.get_arc(si.split("-")[0], si.split("-")[1]).constructible_chargers))
                    for si, data in self.warmstart_solution.dynamic_invest.items()
                }
            )
            self.warmstart_solution.validate(intermediate_rep,
                                                 (self.instance_parameters.soc_min, self.instance_parameters.soc_max))

                #print("start evaluating")
            try:
                sol_init = objective_cost_value(decision, self.solver_parameters.time_step_in_seconds, "cpp", False)
                logging.info(f"Warmstart Solution evaluated with value {sol_init}")
            except:
                warmstart_failed=True

        if self.warmstart_solution is None or warmstart_failed:
            logging.debug("Initializing ILS from Scratch (no warmstarting solution given / or not functioning b/c of edge cases (i.e., cplex can partially recharge on dyn segments)")

            # Homogenous vehicle speeds and consumption --> we assign any speed / consumption
            decision = DecisionVariables(
                intermediate_rep, {}, {}, self.instance_parameters.max_speed(intermediate_rep.routes[0].vehicle_id),
                self.instance_parameters.max_consumption(intermediate_rep.routes[0].vehicle_id),
                self.instance_parameters.soc_init, self.instance_parameters.soc_max, self.instance_parameters.soc_min,
                self.instance_parameters.energy_prices, self.instance_parameters.consumption_cost
            )
            try:
                sol_init = self.initial_feasible_solution(decision)
            except ValueError:
                return None

        logging.info(f"Cost of initial feasible solution: {sol_init}")

        # first LS step & save as incumbent (must be an improvement by design)
        max_config_size = decision.current_number_of_chargers()
        ls_solution_value, num_local_search_iterations = self.local_search(decision, sol_init)
        self.incumbent = ls_solution_value
        self.incumbent_configuration = decision.copy(retain_route_networks=True)

        # the maximum strength is restricted to the number of charging stations after the first local search iteration
        max_strength = decision.current_number_of_chargers()
        strength = self.solver_parameters.strength
        unimproved_iterations = 0
        while self.solver_clock.get_remaining_time() > 0:

            # Update iteration counter
            self.number_iterations += 1

            if unimproved_iterations >= self.solver_parameters.i_opt:
                strength = min(strength + self.solver_parameters.strength, max_strength)

            # reset configuration to incumbent
            decision.set_configuration(*self.incumbent_configuration)

            # structured perturbation
            perturbed_solution_value = self.structured_perturbation(
                decision, current_strength=strength, max_config_size=max_config_size
            )

            # Call Local Search
            ls_solution_value, num_local_search_iterations = self.local_search(decision, perturbed_solution_value)
            ls_solution_value_print = ls_solution_value

            # Acceptance criterion
            if ls_solution_value <= self.incumbent + toler:
                # store the best solution so far
                logging.info(f"Update best found solution from {self.incumbent} to {ls_solution_value}")
                self.incumbent = ls_solution_value
                self.incumbent_configuration = decision.copy(retain_route_networks=True)
                logging.info(
                    f"New solution ({len(self.incumbent_configuration[0]), len(self.incumbent_configuration[1])}) has hash {self._hash_config()}")
                unimproved_iterations = 0
                strength = self.solver_parameters.strength
            else:
                unimproved_iterations += 1

            # log for later analysis
            self.run_statistics.append((
                self.number_iterations, num_local_search_iterations, self.solver_clock.get_elapsed_time(),self.incumbent,
                ls_solution_value_print, perturbed_solution_value, strength, len(self.infeasible_configurations),
                len(self.visited_configurations.keys())
            ))

        # set back decision attributes to optimal setting found
        logging.info(f"Reset to incumbent solution with hash {self._hash_config()}")
        decision.set_configuration(*self.incumbent_configuration)
        final_solution = best_routing(
            decision, self.solver_parameters.time_step_in_seconds
        )

        # report run statistics
        logging.info("Postprocess (i.e. reconstruct solution) and write logs and files")
        self._report_run_statistics(self.solver_parameters.get_full_solver_path)
        self._report_conflicts(self.solver_parameters.get_full_solver_path, final_solution)

        logging.info(f"Solution Value of Best Found Solution: {final_solution.global_cost}")
        logging.info(f"######################## End ILS ########################")

        return final_solution

    def local_search(self, decision: DecisionVariables, solution: float) -> Tuple[float, int]:
        """
        Perform a local search around the given solution. Close, swap and open as long as the solution is improved.
        @param decision: DecisionVariables
        @param solution: current solution to improve
        @return: the best solution found, number of iterations performed
        """
        logging.info(f"---Local Search---")
        old_solution = solution
        improved = True
        iter_i = 1
        while improved and self.solver_clock.get_remaining_time()>0:
            logging.info(f"Local search iteration: {iter_i}")

            new_solution = self.remove_dynamic_operator(decision, old_solution)

            new_solution = self.remove_static_operator(decision, new_solution)

            new_solution = self.swap_static_to_dynamic_operator(decision, new_solution)

            new_solution = self.swap_any_to_static_operator(decision, new_solution)

            new_solution = self.add_dynamic_operator(decision, new_solution)

            new_solution = self.add_static_operator(decision, new_solution)

            iter_i += 1
            improved = new_solution + toler < old_solution
            if improved:
                old_solution = new_solution

        logging.info(f"---Local Search final solution value: {old_solution}---")
        return old_solution, iter_i

    def initial_feasible_solution(self, decision: DecisionVariables, verbose: bool=True) -> float:
        """
        Compute an initial  feasible solution for the problem. Update the decision chargers with the chargers to open
        @param decision: DecisionVariables
        @param verbose: Boolean indicating if log should be printed
        @return: Solution Value of initial solution
        """
        if verbose:
            logging.info("---Compute Initial Solution---")

        # open random chargers until solution becomes feasible
        while self.solver_clock.get_remaining_time()>0:
            closed_stations = decision.set_of_charging_stations_for_sampling(filter_for_constructed=False)
            k = self.solver_parameters.init_opening_stepsize
            if len(closed_stations)<self.solver_parameters.init_opening_stepsize:
                k = len(closed_stations)
                if k == 0:
                    logging.info("Problem infeasible - All Chargers in Configuration but no Solution found")
                    raise ValueError

            # this set only contains each dyn charging station once (with a random pair of keys :/ )
            chargers = rd.sample(
                closed_stations,
                k=k
            )

            # add to configuration (full station, i.e., all segments belonging to a certain station)
            for charger in chargers:
                logging.info(f"Add {charger}")
                decision.add_charging_station(charger)

            # update the sets of chargers to open for all routes
            try:
                solution = best_routing(
                    decision=decision,
                    time_step=self.solver_parameters.time_step_in_seconds,
                )
                self.visited_configurations[decision.configuration_set()] = solution.global_cost
                if verbose:
                    logging.info(f"---Initial Solution Greedily Computed---")
                return solution.global_cost
            except ValueError:
                self.infeasible_configurations.add(decision.configuration_set())

    def swap_any_to_static_operator(self, decision: DecisionVariables, solution: float) -> float:
        """
        Look for the nearest static charger, and attempt to swap.
        The final decision is the result of the swap that reduces total cost the most
        @param decision: DecisionVariables
        @param solution: Float current solution_value to compare with
        @return: the best solution found
        """
        logging.info(f"Swapping to static charger started")
        list_of_opened_chargers = decision.set_of_charging_stations_for_sampling(filter_for_constructed=True)
        list_of_closed_chargers = decision.set_of_charging_stations_for_sampling(filter_for_constructed=False, filter_for_static=True)

        success_flag = False
        current_solution = solution
        best_solution = solution
        best_decision = decision.copy(retain_route_networks=True)

        # list of closed chargers empty, return given solution
        if len(list_of_closed_chargers) == 0:
            return current_solution

        # select chargers to check
        random_chargers_to_check = rd.sample(
            list_of_opened_chargers, k=min(self.solver_parameters.ls_num_configs_to_explore, len(list_of_opened_chargers))
        )

        for charger in random_chargers_to_check:
            charger_to_open = nearest_charger(decision, charger, list_of_closed_chargers)

            # swap
            decision.remove_charging_station(charger)

            if decision.configuration_cost() + \
                    decision.intermediate_rep.min_cost_per_charger.get(
                        charger_to_open[2].id, charger_to_open[2].transformer_construction_cost
                    ) + sum(decision.lower_cost_bound_per_route) >= best_solution:
                logging.info(f"Prune based on lower bounds")
                decision.add_charging_station(charger)
                continue

            decision.add_charging_station(charger_to_open)
            conf = decision.configuration_set()
            if conf not in self.infeasible_configurations:
                try:
                    if conf in self.visited_configurations:
                        current_solution = self.visited_configurations[conf]
                    else:
                        current_solution = objective_cost_value(
                            decision,
                            time_step=self.solver_parameters.time_step_in_seconds,
                            interpreter=self.solver_parameters.interpreter,
                            multithreading=self.solver_parameters.multithreading,
                        )
                        self.visited_configurations[conf] = current_solution

                    if current_solution + toler < best_solution:
                        # store
                        success_flag = True
                        best_solution = current_solution
                        best_decision = decision.copy()

                except ValueError:
                    self.infeasible_configurations.add(decision.configuration_set())

            # reset
            decision.add_charging_station(charger)
            decision.remove_charging_station(charger_to_open)

        decision.set_configuration(*best_decision)
        logging.info(f"Swapping to static charger finished ({success_flag})")
        return best_solution

    def swap_static_to_dynamic_operator(self, decision: DecisionVariables, solution: float) -> float:
        """
        For each static charger in sample, look for the n nearest dynamic chargers, and attempt to swap (1-to-n swap where n is the
        number of dynamic charging stations to retain feasbility). The final decision is the result of the swap that
        reduces total cost the most.
        @param decision: DecisionVariables
        @param solution: Float current solution_value to compare with
        @return: the best solution found
        """
        logging.info(f"Swapping to dynamic charger started")
        list_of_opened_chargers = decision.set_of_charging_stations_for_sampling(
            filter_for_constructed=True, filter_for_static=True
        )
        list_of_closed_chargers = decision.set_of_charging_stations_for_sampling(
            filter_for_constructed=False, filter_for_dynamic=True
        )

        success_flag = False
        current_solution = solution
        best_solution = solution
        best_decision = decision.copy(retain_route_networks=True)
        charger_to_open: tuple[str, str, Charger]

        # list of closed chargers empty, return given solution
        if len(list_of_closed_chargers) == 0:
            logging.info(f"Swapping to dynamic charger finished (no chargers to open)")
            return current_solution

        # select chargers to check
        random_chargers_to_check = rd.sample(
            list_of_opened_chargers,
            k=min(self.solver_parameters.ls_num_configs_to_explore, len(list_of_opened_chargers))
        )

        for charger in random_chargers_to_check:
            # remove once
            decision.remove_charging_station(charger)
            charger_to_open = nearest_charger(decision, charger, list_of_closed_chargers)

            # Prune based on lower bound;
            # lower bound is current config + min add. cost due to new charger + lower bound on routes
            if decision.configuration_cost() + \
                    decision.intermediate_rep.min_cost_per_charger.get(
                        charger_to_open[2].id, charger_to_open[2].transformer_construction_cost
                    ) + sum(decision.lower_cost_bound_per_route) >= best_solution:
                logging.info(f"Prune based on lower bounds")
                decision.add_charging_station(charger)
                continue

            decision.add_charging_station(charger_to_open)
            conf = decision.configuration_set()
            if conf not in self.infeasible_configurations:
                try:
                    if conf in self.visited_configurations:
                        current_solution = self.visited_configurations[conf]
                    else:
                        current_solution = objective_cost_value(
                            decision,
                            time_step=self.solver_parameters.time_step_in_seconds,
                            interpreter=self.solver_parameters.interpreter,
                            multithreading=self.solver_parameters.multithreading,
                        )
                        self.visited_configurations[conf] = current_solution

                    # tighten (configurations are saved in the tightening function)
                    current_solution, decision = self.tighten_dynamic_station_configuration(
                        decision,charger_to_open,current_solution
                    )

                    if current_solution + toler < best_solution:
                        # store
                        logging.debug(f"Feasible swap with new sol value {current_solution}; past {best_solution}")
                        success_flag = True
                        best_solution = current_solution
                        best_decision = decision.copy()

                    # if we made it until here, the config is feasible and we leave the while loop
                    logging.debug(f"Feasible swap")
                except ValueError:
                    self.infeasible_configurations.add(decision.configuration_set())
            else:
                logging.debug(f"Infeasible swap")

            # reset
            logging.debug("Reset to initial configuration")
            decision.add_charging_station(charger)
            decision.remove_charging_station(charger_to_open)

        decision.set_configuration(*best_decision)
        logging.info(f"Swapping to dynamic charger finished ({success_flag})")
        return best_solution

    def add_static_operator(self, decision: DecisionVariables, solution: float) -> float:
        """
        Add stationary stations to the configuration.
        @param decision: DecisionVariables
        @param solution: Float current solution_value to compare with
        @return: the best solution found
        """
        logging.info(f"Adding static stations started")
        list_of_opened_chargers = decision.set_of_charging_stations_for_sampling(filter_for_constructed=True)

        success_flag = False
        current_solution = solution
        best_solution = solution
        best_decision = decision.copy(retain_route_networks=True)

        # list of closed chargers empty, return given solution
        if len(decision.set_of_charging_stations_for_sampling(filter_for_constructed=False, filter_for_static=True)) == 0:
            return current_solution

        random_chargers_to_check = rd.sample(
            list_of_opened_chargers, k=min(self.solver_parameters.ls_num_configs_to_explore, len(list_of_opened_chargers))
        )
        for charger in random_chargers_to_check:
            charger_to_open = decision.furthest_charger(
                charger,
                decision.set_of_charging_stations_for_sampling(filter_for_constructed=False, filter_for_static=True)
            )

            # Prune based on lower bound
            if decision.configuration_cost() + \
                    charger_to_open[2].transformer_construction_cost + \
                    sum(decision.lower_cost_bound_per_route) >= best_solution:
                continue

            decision.add_charging_station(charger_to_open)
            conf = decision.configuration_set()
            try:
                if conf in self.visited_configurations:
                    logging.debug("..this configuration has already been explored")
                    current_solution = self.visited_configurations[conf]
                else:
                    current_solution = objective_cost_value(
                        decision, time_step=self.solver_parameters.time_step_in_seconds,
                        interpreter=self.solver_parameters.interpreter,
                        multithreading=self.solver_parameters.multithreading,
                    )
                    self.visited_configurations[conf] = current_solution
                if current_solution + toler < best_solution:
                    # store
                    success_flag = True
                    best_solution = current_solution
                    best_decision = decision.copy()

            except ValueError:
                self.infeasible_configurations.add(decision.configuration_set())

            # reset
            decision.remove_charging_station(charger_to_open)

        logging.info(f"Adding static stations finished ({success_flag})")
        decision.set_configuration(*best_decision)
        return best_solution

    def add_dynamic_operator(self, decision: DecisionVariables, solution: float) -> float:
        """
        Add dynamic stations (incl. deconstruction) from a sample based neighborhood.
        @param decision: DecisionVariables
        @param solution: Float current solution_value to compare with
        @return: the best solution found
        """
        logging.info(f"Adding dynamic stations started")
        list_of_opened_chargers = decision.set_of_charging_stations_for_sampling(filter_for_constructed=True)

        success_flag = False
        current_solution = solution
        best_solution = solution
        best_decision = decision.copy(retain_route_networks=True)

        # list of closed chargers empty, return given solution
        if len(decision.set_of_charging_stations_for_sampling(filter_for_constructed=False, filter_for_dynamic=True)) == 0:
            return current_solution

        random_chargers_to_check = rd.sample(
            list_of_opened_chargers, k=min(self.solver_parameters.ls_num_configs_to_explore, len(list_of_opened_chargers))
        )
        for charger in random_chargers_to_check:
            charger_to_open = decision.furthest_charger(
                charger, decision.set_of_charging_stations_for_sampling(filter_for_constructed=False, filter_for_dynamic=True)
            )

            # Prune based on lower bound; lower bound is current config + min add. cost due to new charger + lower bound
            # on routes
            if decision.configuration_cost() + decision.intermediate_rep.min_cost_per_charger[charger_to_open[2].id] + \
                    sum(decision.lower_cost_bound_per_route) >= best_solution:
                logging.debug(f"Pruned based on lower bounds")
                continue

            decision.add_charging_station(charger_to_open)
            conf = decision.configuration_set()
            try:
                if conf in self.visited_configurations:
                    logging.debug("..this configuration has already been explored")
                    current_solution = self.visited_configurations[conf]
                else:
                    current_solution = objective_cost_value(
                        decision, time_step=self.solver_parameters.time_step_in_seconds,
                        interpreter=self.solver_parameters.interpreter,
                        multithreading=self.solver_parameters.multithreading,
                    )
                    self.visited_configurations[conf] = current_solution

                # tighten configuration of added charger
                current_solution, decision = self.tighten_dynamic_station_configuration(
                    decision, charger_to_open, current_solution
                )

                if current_solution + toler < best_solution:
                    # store
                    success_flag = True
                    best_solution = current_solution
                    best_decision = decision.copy()

            except ValueError:
                self.infeasible_configurations.add(decision.configuration_set())

            # reset
            decision.remove_charging_station(charger_to_open)

        logging.info(f"Adding dynamic stations finished ({success_flag})")
        decision.set_configuration(*best_decision)
        return best_solution

    def remove_static_operator(self, decision: DecisionVariables, initial_solution: float) -> float:
        """
        Iterate over all static chargers in configuration, close, compute the best routing if feasible, and compare
        the resulting costs. Then close the worst charger with regard to cost.
        (If closing doesn't improve the solution, then return the parameter solution)
        @param decision: DecisionVariables
        @param initial_solution: Float solution value of current solution to be im improved
        @return: new solution value
        """
        logging.info("Removing static stations started")
        list_of_opened_chargers = decision.set_of_charging_stations_for_sampling(filter_for_constructed=True, filter_for_static=True)

        success_flag = False
        best_solution = initial_solution
        best_decision = decision.copy(retain_route_networks=True)
        #initial_route_networks = best_decision[2]

        # check only subset
        random_chargers_to_check = rd.sample(
            list_of_opened_chargers,
            k=min(self.solver_parameters.ls_num_configs_to_explore, len(list_of_opened_chargers))
        )

        for charger in random_chargers_to_check:
            # close the current charger
            decision.remove_charging_station(charger)
            conf = decision.configuration_set()
            if conf not in self.infeasible_configurations:
                try:
                    if conf in self.visited_configurations:
                        current_solution = self.visited_configurations[conf]
                    else:
                        current_solution = objective_cost_value(
                            decision, time_step=self.solver_parameters.time_step_in_seconds,
                            interpreter=self.solver_parameters.interpreter,
                            multithreading=self.solver_parameters.multithreading,
                        )
                        self.visited_configurations[conf] = current_solution

                    if current_solution < best_solution:
                        # store the best configuration
                        success_flag = True
                        best_solution = current_solution
                        best_decision = decision.copy(retain_route_networks=True)

                except ValueError:
                    self.infeasible_configurations.add(decision.configuration_set())

            # reset to the initial configuration
            decision.add_charging_station(charger)

        logging.info(f"Removing static stations finished ({success_flag})")
        decision.set_configuration(*best_decision)
        return best_solution

    def remove_dynamic_operator(self, decision: DecisionVariables, initial_solution: float) -> float:
        """
        Iterate over all dynamic chargers in configuration, close and tighten, compute the best routing if feasible,
        and compare the resulting costs. Then close the worst charger with regard to cost.
        (If closing doesn't improve the solution, then return the parameter solution)
        @param decision: DecisionVariables
        @param initial_solution: Float solution value of current solution to be im improved
        @return: new solution value
        """
        logging.info("Removing (or deconstructing) dynamic stations started")
        list_of_opened_chargers = decision.set_of_charging_stations_for_sampling(filter_for_constructed=True, filter_for_dynamic=True)
        # current_solution = initial_solution
        success_flag = False
        best_solution = initial_solution
        best_decision = decision.copy(retain_route_networks=True)

        # check only subset
        random_chargers_to_check = rd.sample(
            list_of_opened_chargers,
            k=min(self.solver_parameters.ls_num_configs_to_explore, len(list_of_opened_chargers))
        )

        for charger in random_chargers_to_check:
            current_solution, decision = self.tighten_dynamic_station_configuration(
                decision, charger, initial_solution
            )

            if current_solution + toler < best_solution:
                # store
                success_flag = True
                best_solution = current_solution
                best_decision = decision.copy(retain_route_networks=True)

            # reset to initial configuration
            decision.add_charging_station(charger)

        logging.info(f"Removing (and deconstructing) dynamic stations finished ({success_flag})")
        decision.set_configuration(*best_decision)
        return best_solution

    def destroy(self, decision: DecisionVariables, current_strength: int):
        """
        Destroy the solution by closing the best "strength" chargers.
        @param decision: DecisionVariables
        @param current_strength: Strength of Exploration (depends on # unsuccessful ILS iterations)
        @return: None
        """
        logging.info("Destroying Solution Started")
        n = len(decision.set_of_charging_stations_for_sampling(filter_for_constructed=True))

        # in case the number of opened chargers is lower than the strength, close everything
        if n <= current_strength:
            decision.set_configuration({}, {})
            logging.info("Destroying Solution Finished")
            return
        while len(decision.set_of_charging_stations_for_sampling(filter_for_constructed=True)) > n - current_strength:
            decision.remove_charging_station(
                rd.sample(decision.set_of_charging_stations_for_sampling(filter_for_constructed=True), k=1)[0]
            )
        logging.info("Destroying Solution Finished")

    def tighten_dynamic_station_configuration(
            self, decision: DecisionVariables, charger: tuple[str, str, Charger], current_solution_value: float
    ) -> Tuple[float, DecisionVariables]:
        """
        We call this function to tighten the configuration with respect to one specific dynamic station (i.e., the one
        associated with the segment passed as attribute. For this station we stepwise remove the segments such that
        the overall station remains connected and the configuration feasible.
        @param decision: DecisionVariable object
        @param charger: Representation to identify concrete segment
        @param current_solution_value: Initial solution value
        @return: new solution value (can be equal but never worse) than initial one, and new decision variables
        """
        segment_list_copy = copy.copy(decision.intermediate_rep.charger_edges_by_charger[charger[2]])
        binary_tightener = BinaryTightener(
            decision, segment_list_copy, current_solution_value, charger[2]
        )

        # initial cut, i.e., check adding segments from start to middle, then in later loop adjust end such that
        # we find the tightest configuration (note: we always start with the first element)
        decision = binary_tightener.cut_segments(True)

        while not binary_tightener.has_converged():

            conf = decision.configuration_set()
            if conf not in self.visited_configurations:
                try:
                    if conf in self.infeasible_configurations:
                        raise ValueError
                    new_cost = objective_cost_value(
                        decision, time_step=self.solver_parameters.time_step_in_seconds,
                        interpreter=self.solver_parameters.interpreter,
                        multithreading=self.solver_parameters.multithreading)
                    self.visited_configurations[conf] = new_cost
                    feasible = True
                except ValueError:
                    # if new config is not feasible --> set back and return
                    self.infeasible_configurations.add(conf)
                    feasible = False
            else:
                new_cost = self.visited_configurations[conf]
                feasible = True

            # if new config is feasible but worse --> set back and continue without saving
            if feasible and binary_tightener.best_solution_value < new_cost:
                break

            # update info on best solution (we only reach this line if this is also an improvement)
            if feasible:
                binary_tightener.update_best_solution(decision, new_cost)
            decision = binary_tightener.cut_segments(feasible)

        binary_tightener.decisions.set_configuration(*binary_tightener.best_decisions)
        return binary_tightener.best_solution_value, binary_tightener.decisions

    def structured_perturbation(self, decision: DecisionVariables, current_strength: int, max_config_size: int) -> float:
        """
        Destroy the current solution, compute the distances between the closed chargers and the new infeasible routes,
        open the best ones that renders the configuration feasible.
        @param decision: the current decision to perturbation
        @param current_strength: Strength of Exploration (depends on # of unsuccessfull iterations in ILS loop)
        @param max_config_size: Maximum number of chargers in config (to avoid exploding subproblem)
        @return Solution Value of Perturbed Solution
        """
        logging.info(f"---Perturbation Phase Started---")
        # First, destroy
        self.destroy(decision, current_strength=current_strength)

        # Second, repair
        new_solution_value = self.repair(decision, current_strength=current_strength, max_config_size = max_config_size)

        logging.info(f"---Perturbation Phase Finished with Solution Value: {new_solution_value}---")
        return new_solution_value

    def repair(self, decision: DecisionVariables, current_strength: int, max_config_size: int) -> Optional[float]:
        """
        In place alternation of decision [DecisionVariables] instance such that feasibility is restored
        @param decision: DecisionVariables
        @param current_strength: Strength of Exploration (depends on # of unsuccessfull iterations in ILS loop)
        @param max_config_size: Maximum number of chargers in config (to avoid exploding subproblem
        @return solution value of repaired solution
        """
        logging.info(f"Repairing Solution Started with current strength {current_strength}")

        if current_strength + decision.current_number_of_chargers() > max_config_size:
            max_config_size = decision.current_number_of_chargers() + current_strength

        add_chargers = current_strength
        closed_stations = decision.set_of_charging_stations_for_sampling(filter_for_constructed=False)
        while True:
            try:
                chargers_to_open = rd.sample(
                    closed_stations,
                    k=add_chargers
                )
                for charger_to_open in chargers_to_open:
                    decision.add_charging_station(charger_to_open)
                logging.info(
                    f"Evaluate new configuration with ({decision.current_number_of_stationary_chargers()}, "
                    f"{decision.current_number_of_dynamic_chargers()}) chargers"
                )
                new_solution = objective_cost_value(
                    decision,
                    time_step=self.solver_parameters.time_step_in_seconds,
                    interpreter=self.solver_parameters.interpreter,
                    multithreading=self.solver_parameters.multithreading,
                )
                self.visited_configurations[decision.configuration_set()] = new_solution
                logging.debug(f"Evaluating done (feasible)")
                break
            except ValueError:
                logging.debug(f"Evaluating done (infeasible or timeout)")
                pass

            # check feasibility
            for charger_to_remove in chargers_to_open:
                decision.remove_charging_station(charger_to_remove)

            if add_chargers + decision.current_number_of_chargers() >= max_config_size:
                # reset to current strength and try other sampling - this is necessary because very large configs
                # may be rendered infeasible due to iteration limit in subproblem
                add_chargers = current_strength
            else:
                add_chargers += 1

        logging.info(f"Repairing Solution Finished")
        return new_solution


def nearest_charger(decision: DecisionVariables, target_charger: tuple[str, str, Charger],
                    list_of_chargers: list[tuple[str, str, Charger]]) -> tuple[str, str, Charger]:
    """
    Compute the nearest charger (along the shortest path by distance) to the target among a list of chargers
    @param decision: instance of 'DecisionVariable'
    @param target_charger: a charger to be checked
    @param list_of_chargers: List of chargers objects
    @return: the nearest charger (along shortest paths by distance)
    """
    distances = []
    if target_charger[0]==target_charger[1]:
        start_point = target_charger[0]
    else:
        # ID from first segment of dynamic station (\alpha_1)
        start_point = decision.intermediate_rep.charger_edges_by_charger[target_charger[2]][0][0]
    for charger in list_of_chargers:
        if charger[0]==charger[1]:
            distance = decision.get_distance(VertexID(start_point), VertexID(charger[0]))
        else:
            # ID from first segment of dynamic station (\alpha_1)
            start_point_charger = decision.intermediate_rep.charger_edges_by_charger[charger[2]][0][0]
            distance = decision.get_distance(VertexID(start_point), VertexID(start_point_charger))
        distances += [(charger, distance)]
    return min(distances, key=lambda item: item[1])[0]
