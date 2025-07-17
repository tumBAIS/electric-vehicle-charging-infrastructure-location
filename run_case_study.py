import sys
import os
import json
import random
import shutil

from docplex.mp.solution import SolveSolution, WriteLevel
from dataclasses import asdict
from framework.intermediate_representation import parse_intermediate_representation
from iterative_local_search.iterative_local_search import ILSParameters, ILSSolver
from framework.instance_parameter import InstanceParameters
from data_hof import parse_hof_instance
from math_programming_model.math_programming_solver import MathProgrammingParameters, MathProgrammingSolver

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))
if len(sys.argv) != 8:
    print(sys.argv)
    raise TypeError(f"The script expects 8 input arguments; {sys.argv} were given")

random.seed(9002)

run_name = sys.argv[1]
preis_scenario = sys.argv[2]
consumption = float(sys.argv[3])
static_transformer_cost = float(sys.argv[4])
dynamic_transformer_cost = float(sys.argv[5])
variable_cost = float(sys.argv[6])
allow_path_deviations = str(sys.argv[7])


def main(
        run_name: str,
        preis_scenario: str,
        consumption: float,
        static_transformer_cost: float,
        dynamic_transformer_cost: float,
        variable_cost: float,
        allow_path_deviations: str,
):
    # some fixed parameters
    if allow_path_deviations=="True":
        allow_path_deviations=True
    else:
        allow_path_deviations=False
    check_conflicts = False
    solver = "cplex" if not allow_path_deviations else "ILS"
    interpreter = "cpp"
    city = "hof" if "hof" in run_name else "bad_staffelstein"
    static_charging_rate=30 if city=="hof" else 7
    dynamic_charging_rate=30 if city=="hof" else 7
    speed = 15.0 if city=="hof" else 18.0
    max_capacity = 33.0

    if allow_path_deviations:
        dev_string = "Deviations"
    else:
        dev_string = "noDeviations"

    energy_prices = {
        0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.3, 9: 0.3, 10: 0.3, 11: 0.3,
        12: 0.3, 13: 0.3, 14: 0.3, 15: 0.3, 16: 0.3, 17: 0.3, 18: 0.3, 19: 0.3, 20: 0.3, 21: 0.3, 22: 0.3,
        23: 0.3
    }
    if preis_scenario=="medium":
        energy_prices[8] = 0.2
        energy_prices[9] = 0.2
        energy_prices[10] = 0.2
    elif preis_scenario=="ibc":
        energy_prices[0] = 0.3
        energy_prices[1] = 0.3
        energy_prices[2] = 0.3
        energy_prices[3] = 0.3
        energy_prices[4] = 0.3
        energy_prices[5] = 0.3
        energy_prices[6] = 0.3
        energy_prices[7] = 0.3
        energy_prices[8] = 0.04
        energy_prices[9] = 0.04
        energy_prices[10] = 0.04
        energy_prices[11] = 0.04
        energy_prices[12] = 0.04
        energy_prices[13] = 0.04
        energy_prices[14] = 0.2067
        energy_prices[15] = 0.2067
        energy_prices[16] = 0.2067
        energy_prices[17] = 0.2067
        energy_prices[18] = 0.2067
        energy_prices[19] = 0.2067
        energy_prices[20] = 0.2067
        energy_prices[21] = 0.2067
        energy_prices[21] = 0.2067
        energy_prices[22] = 0.2067
        energy_prices[23] = 0.2067
    elif preis_scenario=="full":
        energy_prices[8] = 0.04
        energy_prices[9] = 0.04
        energy_prices[10] = 0.04

    # in case we want to load from file
    intermediate_rep = parse_intermediate_representation(f"Instance_parser/{city}_instances/{run_name}_{dev_string}.json")

    # update cost in case they changed vs. saved instance
    intermediate_rep.update_static_charger(static_transformer_cost, static_charging_rate)
    intermediate_rep.update_dynamic_chargers(dynamic_transformer_cost, variable_cost, dynamic_charging_rate)
    intermediate_rep.limit_distance_precision(5)
    intermediate_rep.print_descriptive_stats()

    # initialise instance parameter
    instance_parameters = InstanceParameters(
        velocity=speed,
        consumption=consumption,
        soc_init=0.9*max_capacity,
        soc_max=0.9*max_capacity,
        soc_min=0.1*max_capacity,
        energy_prices=energy_prices,
        consumption_cost=0.3,
        allow_path_deviations=allow_path_deviations,
    )

    instance_parameter_string = "_".join([
        s for s in map(str, [round(l, 2) for l in asdict(instance_parameters).values() if (isinstance(l, float) or isinstance(l, int))])
    ])
    result_path = f"results/{run_name}/{instance_parameter_string}/{solver}"

    # Function to clear a directory
    def clear_directory(directory_path):
        if os.path.exists(directory_path):
            for file_or_folder in os.listdir(directory_path):
                file_or_folder_path = os.path.join(directory_path, file_or_folder)
                if os.path.isfile(file_or_folder_path) or os.path.islink(file_or_folder_path):
                    os.unlink(file_or_folder_path)  # Remove file or symlink
                elif os.path.isdir(file_or_folder_path):
                    shutil.rmtree(file_or_folder_path)  # Remove directory

    # save price curve
    # Create the directory if it doesn't exist
    # Clear the folder if it exists
    clear_directory(result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    with open(f"results/{run_name}/{instance_parameter_string}/energy_prices.json", "w") as json_file:
        json.dump(energy_prices, json_file, indent=4)

    if solver=="ILS":
        solver_parameters = ILSParameters(
            results_path_name=result_path,
            time_step_in_seconds=300,
            run_time_in_seconds=7200,
            interpreter=interpreter,
            multithreading=False,
            ls_num_configs_to_explore=2,
            check_conflicts=check_conflicts,
            init_opening_stepsize=5,
            strength=2,
        )
        solver_machine = ILSSolver(instance_parameters, solver_parameters)
        solution_representation = solver_machine.solve(intermediate_rep)
        if solution_representation is not None:
            solution_representation.dump_as_json(solver_parameters.get_full_solver_path+"/solution_representation.json")
            solution_representation.validate(intermediate_rep, (instance_parameters.soc_min, instance_parameters.soc_max))

    if solver=="cplex":
        solver_parameters = MathProgrammingParameters(
            results_path_name=result_path,
            time_step_in_seconds=60,
            run_time_in_seconds=3600,
            num_replicas=20,
        )

        # warm_start_path = "<path to warmstart solution>"
        solver_machine = MathProgrammingSolver(instance_parameters, solver_parameters)#, warm_start_path)
        solution_representation, solve_solution = solver_machine._solve(intermediate_rep)
        if solution_representation is not None:
            solution_representation.dump_as_json(solver_parameters.get_full_solver_path+"/solution_representation.json")
            solution_representation.validate(intermediate_rep,
                                             (instance_parameters.soc_min, instance_parameters.soc_max))
            file_path = os.path.join(solver_parameters.get_full_solver_path, "solve_solution.mst")
            solve_solution.export(file_path, format="mst", write_level=WriteLevel.DiscreteVars)


if __name__ == "__main__":
    main(run_name, preis_scenario, consumption, static_transformer_cost, dynamic_transformer_cost, variable_cost, allow_path_deviations)
