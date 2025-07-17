from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Optional
from framework.instance_parameter import InstanceParameters
from framework.intermediate_representation import IntermediateRepresentation
from framework.solution_representation import SolutionRepresentation


@dataclass(frozen=True)
class SolverParameters:
    results_path_name: Optional[str]
    time_step_in_seconds: int
    run_time_in_seconds: int

    @property
    def get_full_solver_path(self) -> Optional[str]:
        if self.results_path_name is None:
            return None
        attribute_names = [field.name for field in fields(self) if field.name != 'results_path_name']
        attribute_values = [getattr(self, name) for name in attribute_names if not isinstance(getattr(self, name), dict)]
        return f"{self.results_path_name}/" + "_".join(map(str, attribute_values))


class Solver(ABC):
    instance_parameters: InstanceParameters
    solver_parameters: SolverParameters

    def __init__(self, instance_parameters, solver_parameters):
        self.solver_parameters = solver_parameters
        self.instance_parameters = instance_parameters

    @abstractmethod
    def solve(
            self,
            intermediate_rep: IntermediateRepresentation,
    ) -> Optional[SolutionRepresentation]:
        pass
