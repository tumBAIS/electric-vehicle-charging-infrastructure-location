import logging
import sys
import os
import json
from typing import Optional
from docplex.mp.solution import SolveSolution, WriteLevel
from dataclasses import asdict
from framework import preprocessing as preprocessor
from iterative_local_search.iterative_local_search import ILSParameters, ILSSolver
from framework.instance_parameter import InstanceParameters
from framework.solution_representation import SolutionRepresentation
from framework.intermediate_representation import IntermediateRepresentation, VertexID, parse_intermediate_representation
from parse_adapted_solomon_instances import read_instance
from iterative_local_search.decision_variables import DecisionVariables
from math_programming_model.math_programming_solver import MathProgrammingParameters, MathProgrammingSolver


print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))
if len(sys.argv) not in [3,4]:
    raise TypeError("The script expects two or three input arguments.")

run_name = sys.argv[1]
solver = sys.argv[2]
speed_adaption = None

if len(sys.argv) > 3:
    speed_adaption = float(sys.argv[3])


def _parse_solomon(instance_name: str):
    """
    Generate a pre-processed IntermediateRepresentation object of the given instance
    @param instance_name: the name of the instance
    @return: the intermediate representation, the chargers used by the BKS, and the parameters of the instance (speed,
    consumption, maximum capacity)
    """
    intermediate_rep, params = read_instance(instance_name, "PR", max_routes=None)

    speed = params['v']
    consumption = params['r']
    max_capacity = params['Q']
    return intermediate_rep, speed, consumption, max_capacity


def _convert_sr_dump_to_decision_variables(
    path: str,
    ir: IntermediateRepresentation,
    i_paras: InstanceParameters,
) -> DecisionVariables:
    dv = DecisionVariables(
        intermediate_rep=ir,
        vertex_chargers = {},
        arc_chargers = {},
        vehicle_max_speed=i_paras.max_speed(ir.routes[0].vehicle_id),
        vehicle_consumption=i_paras.consumption,
        soc_init=i_paras.soc_init,
        max_soc=i_paras.soc_max,
        min_soc=i_paras.soc_min,
        energy_prices=i_paras.energy_prices,
        consumption_cost=i_paras.consumption_cost
    )
    with open(path, 'r') as file:
        data = json.load(file)

    for key, value in data["static_invest"].items():
        # every of the found pairs is a charger in intermediate representation
        for v in ir.charger_nodes:
            charger = next(iter(v.constructible_charger))
            if key == v.id:
                assert key not in dv.vertex_chargers
                dv.vertex_chargers[key] = charger
                break # one vertex per charger

    for key, value in data["dynamic_invest"].items():
        # same logic as above
        for u,v,arc in ir.charger_edges:
            charger = next(iter(arc.constructible_chargers))
            if value["key_0"] == u and value["key_1"] == v:
                assert (value["key_0"], value["key_1"]) not in dv.arc_chargers
                dv.arc_chargers[(value["key_0"], value["key_1"])] = charger
                continue # multiple arcs per charger
    return dv


def main(run_name: str, solver: str, speed_adaption: Optional[float]):
    # necessary
    allow_path_deviations = True
    check_conflicts = False
    interpreter = "cpp" if solver=="ILS" else "python"

    # parse and simplify instance
    logging.info("Start parsing and simplifying instances")
    intermediate_rep, speed, consumption, max_capacity = _parse_solomon(run_name)
    intermediate_rep = preprocessor.simplify_intermediate_repr(intermediate_rep, allow_path_deviations)

    if speed_adaption:
        speed *= speed_adaption

    intermediate_rep = preprocessor.preprocess_time_windows_in_inter_rep(intermediate_rep, vehicle_maxspeed=speed)
    logging.info("Parsing and simplifying finished")

    # initialise instance parameter
    instance_parameters = InstanceParameters(
        velocity=speed,
        consumption=consumption,
        soc_init=max_capacity,
        soc_max=max_capacity,
        soc_min=0.0,
        energy_prices=0.05,
        consumption_cost=0.05,
        allow_path_deviations=allow_path_deviations,
    )

    instance_parameter_string = "_".join(map(str, asdict(instance_parameters).values()))
    result_path = f"results/{run_name}_BidiAStar/{instance_parameter_string}/{solver}"

    if solver=="ILS":
        warmstart_path = f"warmstarts/{run_name}/{instance_parameter_string}".replace("True",
                                                                                      "False")
        solver_parameters = ILSParameters(
            results_path_name=result_path,
            time_step_in_seconds=3600,
            run_time_in_seconds=3600,
            interpreter=interpreter,
            multithreading=False,
            ls_num_configs_to_explore=2,
            check_conflicts=check_conflicts,
            init_opening_stepsize=2,
            strength = 2,
        )

        warmstart_solution = None
        if os.path.exists(warmstart_path+"/solution_representation.json"):
            warmstart_solution = SolutionRepresentation.load_from_json(warmstart_path+"/solution_representation.json")

        solver_machine = ILSSolver(instance_parameters, solver_parameters, warmstart_solution)
        solution_representation = solver_machine.solve(intermediate_rep)
        if solution_representation is not None:
            solution_representation.dump_as_json(solver_parameters.get_full_solver_path+"/solution_representation.json")
            solution_representation.validate(intermediate_rep, (instance_parameters.soc_min, instance_parameters.soc_max))

    if solver=="cplex":
        warmstart_path = f"warmstarts/{run_name}/{instance_parameter_string}".replace("True","False") + "/solve_solution.mst"
        solver_parameters = MathProgrammingParameters(
            results_path_name=result_path,
            time_step_in_seconds=3600,
            run_time_in_seconds=3600,
            num_replicas=3,
        )
        if os.path.exists(warmstart_path):
            logging.info(f"Load warmstart from {warmstart_path}")
            solver_machine = MathProgrammingSolver(instance_parameters, solver_parameters, warmstart_path)
        else:
            solver_machine = MathProgrammingSolver(instance_parameters, solver_parameters)
        solution_representation, solve_solution = solver_machine._solve(intermediate_rep)
        if solution_representation is not None:
            solution_representation.dump_as_json(solver_parameters.get_full_solver_path+"/solution_representation.json")
            solution_representation.validate(intermediate_rep, (instance_parameters.soc_min, instance_parameters.soc_max))

main(run_name, solver, speed_adaption)


