import sys
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Set, TextIO, Union, Tuple
import framework.intermediate_representation as ir
from funcy import map
from rich import print as rprint
from rich.table import Table

from math_programming_model.network import Arc, Vehicle, Vertex
import math_programming_model.network as nw
from framework.solution_representation import SolCharger, Itinerary, Point
from framework.intermediate_representation import VehicleID

Route = List[Vertex]


@dataclass
class Solution:
    investments: Set[Union[Vertex, Arc]]
    routes: Dict[Vehicle, Route]
    global_cost: float  # cost unit
    consumed_energy: float  # kwH
    recharged_energy: float  # kwH
    consumed_energy_per_vehicle: Dict[Vehicle, list]
    recharger_energy_per_vehicle: Dict[Vehicle, list]

    def __str__(self):
        output = StringIO("")
        print(len(self.investments), "investments:", file=output)
        for investment in self.investments:
            print("\t", investment, file=output)
        print(len(self.routes), "routes:", file=output)
        for v_id, sol_route in self.routes.items():
            print("Vehicle", v_id, ":", "->".join(map(str, sol_route)), file=output)
        return output.getvalue()

    def _is_charger(self, v: Union[Vertex, Arc]) -> bool:
        return v in self.investments

    def __format__(self, format_spec):
        pass

    def _format_route_rich(self, route: Route, vehicle_id: str, time_step: int) -> str:
        table = Table(title=f"Route of vehicle {vehicle_id}", width=125)
        table.add_column("Vertex", width=30)
        table.add_column("Arrival time")
        table.add_column("Arrival SoC")
        table.add_column("Depart. time")
        table.add_column("Depart. SOC")
        table.add_column("Delta time")
        # table.add_column("Charge time")
        table.add_column("Delta SoC")
        for v in route:
            args = [str(v), round(v.departure_time-v.delta_time*time_step, 2), round(v.arrival_soc, 2), round(v.departure_time, 2),
                    round(v.arrival_soc+v.delta_charge, 2), round(time_step*v.delta_time, 2)]
            if self._is_charger(v):
                # args.append(v.delta_time*time_step)
                args.append(round(v.delta_charge, 2))
            else:
                # args.append(v.delta_time*time_step)
                args.append(round(v.delta_charge,2))
            table.add_row(*map(str, args))
        buf = StringIO("")
        rprint(table, file=buf)
        return buf.getvalue()

    def _format_static_charger_investments(self) -> str:
        output = StringIO("")
        vertex_table = Table(title="Static chargers", width=125)
        vertex_table.add_column("Vertex", width=45)
        vertex_table.add_column("Price")
        vertex_table.add_column("Charging rate")
        for v in filter(lambda x: isinstance(x, Vertex), self.investments):
            if v.is_dummy:
                continue
            vertex_table.add_row(
                str(v),
                f"{v.charger.construction_cost:.2f}",
                f"{v.charger.charging_rate_kwh_per_h:.2f}",
            )
        rprint(vertex_table, file=output)
        return output.getvalue()

    def _format_dynamic_charger_investments(self) -> str:
        output = StringIO("")
        dyn_charger_table = Table(title="Dynamic chargers", width=125)
        dyn_charger_table.add_column("Arc", width=35)
        dyn_charger_table.add_column("Transformer share price")
        dyn_charger_table.add_column("Segment price")
        dyn_charger_table.add_column("Segment length")
        dyn_charger_table.add_column("Charging rate")
        arcs_by_charger = defaultdict(list)
        for v in filter(lambda x: isinstance(x, Arc), self.investments):
            arcs_by_charger[v.charger].append(v)
        for charger, arcs in arcs_by_charger.items():
            print=True
            for i, v in enumerate(arcs):
                if v.is_dummy:
                    continue
                dyn_charger_table.add_row(
                    str(v),
                    f"{charger.transformer_construction_cost:.2f}" if print else "-",
                    f"{charger.construction_cost_per_km * v.distance:.2f}",
                    f"{v.distance:.2f}",
                    f"{charger.charging_rate_kwh_per_h:.2f}",
                )
                print=False
        rprint(dyn_charger_table, file=output)
        return output.getvalue()

    def _format_investments(self) -> str:
        output = StringIO("Investments\n")

        output.write(self._format_static_charger_investments())
        output.write(self._format_dynamic_charger_investments())

        return output.getvalue()

    def report(self, time_step: int, output: TextIO = sys.stdout):
        print(self._format_investments(), file=output)
        for v_id, sol_route in self.routes.items():
            print(self._format_route_rich(sol_route, vehicle_id=v_id, time_step=time_step), file=output)


def construct_itineraries(sol: Solution, time_step: int) -> List[Itinerary]:
    itin: List[Itinerary] = []
    for veh, route in sol.routes.items():
        accumulated_consumed_energy = sol.consumed_energy_per_vehicle[veh]
        accumulated_recharger_energy = sol.recharger_energy_per_vehicle[veh]

        # init soc
        init_soc = route[0].departure_soc

        # make sure that itineraries are always represented in original nodes (dummies mapped back)
        # we re-construct SOC like this because if Init SOC does not be consumed fully, solver can just distribute SOC
        pts = [Point(
            id=vertex.root_node.id,
            arrival_time=int(vertex.departure_time - time_step * vertex.delta_time),
            departure_time=int(vertex.departure_time),
            soc=init_soc - accumulated_consumed_energy[idx] + accumulated_recharger_energy[idx],   #vertex.departure_soc,
            is_depot=vertex.is_depot,
            is_stop=vertex.is_stop,
            is_static_charger=vertex.can_construct_charger,
            accumulated_consumed_energy=accumulated_consumed_energy[idx],
            accumulated_charged_energy=accumulated_recharger_energy[idx],
            is_synthetic_dyn_charger_representation=vertex.is_auxiliary,
        ) for idx, vertex in enumerate(route)]
        itin.append(Itinerary(vehicle=VehicleID(veh), route=pts))
    return itin


def construct_static_invest(sol: Solution) -> Dict[ir.VertexID, SolCharger]:
    stat_invest: Dict[ir.VertexID, SolCharger] = {}
    for vertex in sol.investments:
        if not isinstance(vertex, Vertex):
            continue
        if vertex.is_dummy:
            continue
        # as first element add charger at this vertex to list
        stat_charger = _stat_charger_to_solution_charger(vertex)
        stat_invest[ir.VertexID(vertex.id)] = stat_charger
    return stat_invest


def _group_dynamic_chargers_by_transformer(sol: Solution) -> Dict[nw.DynamicCharger, Set[Arc]]:
    """Group arcs in solution (i.e., with dyn. chargers) by transformer"""
    arcs = [a for a in sol.investments if isinstance(a, nw.Arc) and not a.is_dummy]
    chargers = {a.charger for a in arcs}
    return {c: [a for a in arcs if a.charger==c] for c in chargers}


def construct_dynamic_invest(sol: Solution) -> Dict[Tuple[ir.VertexID, ir.VertexID], SolCharger]:
    dyn_invest: Dict[Tuple[ir.VertexID, ir.VertexID], SolCharger] = {}
    for arc in sol.investments:
        if not isinstance(arc, nw.Arc):
            continue
        if arc.is_dummy:
            continue
        transformer_share = arc.charger.transformer_construction_cost / len(_group_dynamic_chargers_by_transformer(sol)[arc.charger])
        cost = arc.distance * arc.charger.construction_cost_per_km
        dyn_invest[(ir.VertexID(arc.origin.id), ir.VertexID(arc.target.id))] = SolCharger(
            id=ir.ChargerID(f"{arc}_charger"),
            charger_cost=cost,
            transformer_cost_share=transformer_share,
        )
    return dyn_invest


def _stat_charger_to_solution_charger(v: Vertex) -> SolCharger:
    assert isinstance(v.charger, nw.StaticCharger)
    return SolCharger(
        id=ir.ChargerID(f"{v}_charger"),
        charger_cost=0.0,
        transformer_cost_share=v.charger.construction_cost,
    )


def converting_solution_object(sol: Solution, time_step: int) -> Tuple[
    List[Itinerary],
    Dict[Tuple[ir.VertexID, ir.VertexID], SolCharger],
    Dict[ir.VertexID, SolCharger],
    float,
    float,
    float,
]:
    static_invest = construct_static_invest(sol)
    dynamic_invest = construct_dynamic_invest(sol)
    investment_cost = sum([c.charger_cost+c.transformer_cost_share for c in {**dynamic_invest,**static_invest}.values()])
    routing_cost = sol.global_cost - investment_cost
    itineraries = construct_itineraries(sol, time_step)
    return itineraries, dynamic_invest, static_invest, routing_cost, abs(sol.consumed_energy), abs(sol.recharged_energy)



