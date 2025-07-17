import sys
import random
import logging
from framework import preprocessing as preprocessor
from framework.solution_representation import SolutionRepresentation
from parse_adapted_solomon_instances import read_instance

from framework.staggered_conflict_resolution import resolve_conflict
from iterative_local_search.conflict import detect_conflict_from_solution


print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))
if len(sys.argv) != 2:
    print(sys.argv)
    raise TypeError(f"The script expects 2 input arguments; {sys.argv} were given")

random.seed(9002)

rn = sys.argv[1]

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


def main(rn):
    sol_rep = SolutionRepresentation.load_from_json(f"./Instances/evrptw_instance_solutions/{rn}.json")
    intermediate_rep, speed, consumption, max_capacity = _parse_solomon(rn)
    intermediate_rep = preprocessor.simplify_intermediate_repr(intermediate_rep, True)
    intermediate_rep = preprocessor.preprocess_time_windows_in_inter_rep(intermediate_rep, vehicle_maxspeed=speed)
    logging.info("Parsing and simplifying finished")
    if len(detect_conflict_from_solution(sol_rep)) == 0:
        logging.info("No conflicts in solution")
        sol_rep.dump_as_json(path=f"./results/{rn}_cf.json")
        return
    sol_free = resolve_conflict(sol_rep, intermediate_rep, speed)
    if not sol_free:
        return
    assert len(detect_conflict_from_solution(sol_free)) == 0, f"This solution should be conflict free"
    sol_free.validate(ir=intermediate_rep, soc_bounds=(0.0, max_capacity),precision=1e-3, check_time_windows=False)
    sol_free.dump_as_json(path=f"./results/{rn}_cf.json")


if __name__ == "__main__":
    main(rn)
