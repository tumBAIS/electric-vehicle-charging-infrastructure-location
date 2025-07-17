#include "label.h"
#include "route.h"
#include <algorithm>
#include <optional>
#include <thread>
#include <iostream>

bool Label::valid_label(const Vertex &vertex, std::tuple<double, double> soc_bounds) const {
     return departure_time <= std::get<1>(vertex.departure_time_window) && departure_time >= std::get<0>(vertex.departure_time_window) &&
        std::get<1>(soc_bounds) >= energy && energy >= std::get<0>(soc_bounds);
    }

bool Label::operator==(const Label &right) const {
    return (
            std::abs(cost - right.cost)<=1e-6 &&
            std::abs(energy - right.energy)<=1e-3 &&
            std::abs(departure_time - right.departure_time)<=1e-1);
}

bool Label::operator<(const Label &right) const {
    if (cost != right.cost) {
        return cost < right.cost;
    } else {
        if (energy != right.energy) {
            return effective_energy > right.effective_energy;
        } else {
            return effective_time < right.effective_time;
        }
    }
}

bool LabelNode::operator==(const LabelNode &other) const {
    return *(this->current_label) == *(other.current_label);
}

bool LabelNode::operator<(const LabelNode &other) const {
    const Label& left = *this->current_label;
    const Label& right = *other.current_label;

    if (left.cost != right.cost) {
        return left.cost < right.cost;
    }
    if (left.energy != right.energy) {
        return left.effective_energy > right.effective_energy;
    }
    return left.effective_time < right.effective_time;
}

// Constructor of LabelTree (important: reserve memory for vectors with labels to avoid reallocation)
LabelTree::LabelTree(int n_nodes) : labels_by_cost(n_nodes) {
    for (int i = 0; i < n_nodes; ++i) {
        // Reserve memory for the inner vectors with an initial capacity
        labels_by_cost[i].reserve(8192);
    }
};

void LabelTree::add(LabelNode* label_node, int current_vertex, std::vector<LabelTree::LabelNodeWithBounds>::iterator it) {

    std::vector<LabelNodeWithBounds>& current_vertex_labels_by_cost = labels_by_cost[current_vertex];

    // Store index to prevent iterator invalidation
    size_t index = std::distance(current_vertex_labels_by_cost.begin(), it);
    current_vertex_labels_by_cost.emplace(it, label_node,
       label_node->current_label->effective_energy,
       label_node->current_label->effective_time
    );

    // Recalculate iterator in case of reallocation
    it = current_vertex_labels_by_cost.begin() + index;

    // Repair label_bounds if necessary
    if (current_vertex_labels_by_cost.size() > 1 && it != current_vertex_labels_by_cost.begin()) {

        auto prev = std::prev(it); // Get previous safely

        double max_energy = prev->energy_bound;
        double min_time = prev->time_bound;

        // Iterate and update bounds safely
        while (it != current_vertex_labels_by_cost.end()) {
            bool updated = false; // Track if we make updates

            if (it->energy_bound < max_energy) {
                it->energy_bound = max_energy;
                updated = true;
            } else {
                max_energy = it->energy_bound;
            }

            if (it->time_bound > min_time) {
                it->time_bound = min_time;
                updated = true;
            } else {
                min_time = it->time_bound;
            }

            ++it;

            if (!updated) break; // Prevent infinite loop
        }
    }
}

std::tuple<bool, std::vector<LabelTree::LabelNodeWithBounds>::iterator>LabelTree::
is_dominated(LabelNode* label_node, int current_vertex) {
    std::vector<LabelNodeWithBounds>& label_tree = labels_by_cost[current_vertex];

    // define tolerance
    const double energy_tolerance = 1e-3; // comes in kwh
    const double time_tolerance = 1e-1;  // comes in seconds

    // precompute
    const double label_energy = label_node->current_label->effective_energy;
    const double label_time = label_node->current_label->effective_time;

    // Default iterator points to the end of the label tree
    auto it = label_tree.end();

    // if the current vertex has already been visited
    if (!label_tree.empty()) {

        LabelNodeWithBounds t = {
                label_node,
                label_energy,
                label_time
        };

        // returns iterator starting from the first element which is strictly greater than t
        // loop below starts one from the left to t
        it = std::upper_bound(label_tree.begin(), label_tree.end(), t,
                              [](const LabelNodeWithBounds& a, const LabelNodeWithBounds& b) {
                                  return *(a.label_node) < *(b.label_node);
                              });

        // in case it has the least costly label, it is not dominated
        if (it == label_tree.begin()) {
            return std::make_tuple(false, it);
        }

        // check dominance against all the labels that have less cost, in reverse order with tolerance
        for (auto i = std::make_reverse_iterator(it); i != label_tree.rend(); ++i) {
            // Check in each iteration if a dominate label can exist
            if ((*i).energy_bound + energy_tolerance < label_energy ||
                (*i).time_bound - time_tolerance > label_time) {
                return std::make_tuple(false,it);
            }
            Label* settled_label = (*i).label_node->current_label;
            if (settled_label->effective_energy + energy_tolerance >= label_energy &&
                settled_label->effective_time - time_tolerance <= label_time){
                    return std::make_tuple(true,it);
                }
            }
        }
    return std::make_tuple(false, it);
}


std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
LabelTree::forward_propagate_path(Route& route, int cut_vertex, LabelNode* best_scenario) {
    std::vector<std::tuple<int, double, double, double, double, double, int>> path;
    int start = route.depot_start();

    int current = cut_vertex;

    Label* current_label = best_scenario->current_label;
    int precedent = best_scenario->precedent_vertex;
    LabelNode* precedent_node = best_scenario->precedent_label;
    Label* precedent_label = precedent_node -> current_label;

    while (current != start) {
        path.emplace_back(current, current_label->arrival_time, current_label->departure_time,
                          current_label->energy, current_label->consumed_energy,
                          current_label->recharged_energy, current_label->crossed_dyn_station);

        // update references to current
        current = precedent;
        current_label = precedent_label;

        // update reference to precedent of new current
        precedent = precedent_node->precedent_vertex;
        precedent_node = precedent_node->precedent_label;
        precedent_label = precedent_node->current_label;
    }
    // be aware: crossed station index in first depot representation has no meaning and is assigned -1
    path.emplace_back(start, precedent_label->arrival_time, precedent_label->departure_time,
                                   precedent_label->energy, precedent_label->consumed_energy, precedent_label->recharged_energy, -1);

    double cost = best_scenario->current_label->cost;
    std::reverse(path.begin(), path.end());
    return std::make_tuple(path, cost);
}


std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
LabelTree::backward_propagate_path(Route& route, int cut_vertex, LabelNode* best_scenario, double soc_offset) {
    std::vector<std::tuple<int, double, double, double, double, double, int>> path;
    int end = route.depot_end();

    int current = cut_vertex;
    Label* current_label = best_scenario->current_label;
    int precedent = best_scenario->precedent_vertex;
    LabelNode* precedent_node = best_scenario->precedent_label;
    Label* precedent_label = precedent_node -> current_label;

    double consumed_energy_inverter = std::abs(current_label->consumed_energy);
    double recharged_energy_inverter = current_label->recharged_energy;

    while (current != end) {
        path.emplace_back(current, current_label->arrival_time, current_label->departure_time,
                          current_label->energy + soc_offset, -1*(consumed_energy_inverter - std::abs(current_label->consumed_energy)),
                          recharged_energy_inverter - current_label->recharged_energy, current_label->crossed_dyn_station);
        // update reference to current and the current's precedent
        current = precedent;
        current_label = precedent_label;

        // update reference to precedent of new current via label node
        precedent = precedent_node -> precedent_vertex;
        precedent_node = precedent_node->precedent_label;
        precedent_label = precedent_node->current_label;

    }
    // be aware: crossed station index in first depot representation has no meaning and is assigned -1
    path.emplace_back(end, precedent_label->arrival_time, precedent_label->departure_time,
                      precedent_label->energy + soc_offset, -1*consumed_energy_inverter, recharged_energy_inverter, -1);

    // Reverse order of crossed dynamic segments
    for (size_t p = path.size() - 1; p > 0; --p) {
        std::get<6>(path[p]) = std::get<6>(path[p - 1]);
    }

    // remove first element because this is where forward and backward paths meet (backward can never be alone)
    path.erase(path.begin());

    double cost = best_scenario->current_label->cost;
    return std::make_tuple(path, cost);
}
