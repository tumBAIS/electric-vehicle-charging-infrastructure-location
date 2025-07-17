from dataclasses import dataclass
from typing import Union, Dict
import framework.intermediate_representation as ir


@dataclass(frozen=True)
class InstanceParameters:
    velocity: float
    consumption: float
    soc_init: float
    soc_max: float
    soc_min: float
    energy_prices: Union[float, Dict[int, float]]
    consumption_cost: float
    allow_path_deviations: bool

    def max_speed(self, vehicle_id: ir.VehicleID) -> float:
        if callable(self.velocity):
            return self.velocity(vehicle_id)
        return self.velocity

    def max_consumption(self, vehicle_id: ir.VehicleID) -> float:
        if callable(self.consumption):
            return self.consumption(vehicle_id)
        return self.consumption
