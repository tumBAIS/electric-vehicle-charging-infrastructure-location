#include "network.h"
#include <stdexcept>
#include <deque>
#include <cmath>
#include <queue>
#include <tuple>
#include <optional>
#include <cmath>
#include <algorithm>
#include <utility>
#include "vertex.h"
#include <iostream>
#include <iomanip>

// only for debugging
#include <thread>
//#include <mutex>

// Network Constructor
Network::Network(
        std::vector<Vertex> vertices_,
        std::vector<std::tuple<int, int, Arc>> arcs_,
        std::vector<double> energy_prices_,
        double consumption_cost_,
        std::tuple<double, double> soc_bounds_
) :
    vertices(std::move(vertices_)),
    arcs(std::move(arcs_)),
    energy_prices(std::move(energy_prices_)),
    consumption_cost(consumption_cost_),
    soc_bounds(std::move(soc_bounds_)) {


    // Initialize outgoing_arcs_indices
    outgoing_arcs_indices.reserve(vertices.size());
    // Initialize incoming_arcs_lists
    incoming_arcs_indices.resize(vertices.size());

    int start = 0;
    for (const auto& vertex : vertices) {
        int end = start;
        while (end < static_cast<int>(arcs.size()) && std::get<0>(arcs[end]) == vertex.id) {
            incoming_arcs_indices[std::get<1>(arcs[end])].push_back(end);
            ++end;
        }
        outgoing_arcs_indices.emplace_back(start, end);
        start = end;
    }
}

// a simple lookup in a vector based on provided index
Vertex Network::get_vertex(int id) {
    return vertices[id];
}

// this function will be replaced based on helper list and direct call to arc
std::pair<std::vector<std::tuple<int, int, Arc>>::const_iterator, std::vector<std::tuple<int, int, Arc>>::const_iterator>
Network::get_outgoing_arcs(int id) {
    // Retrieve the start and end iterators of outgoing arcs for the given vertex id
    auto [start_index, end_index] = outgoing_arcs_indices[id];

    // Create iterators pointing to the start and end positions of the slice
    auto begin_iterator = arcs.cbegin() + start_index;
    auto end_iterator = arcs.cbegin() + end_index;

    // Return the iterator range representing the view of outgoing arcs
    return std::make_pair(begin_iterator, end_iterator);
}

std::vector<int>
Network::get_incoming_arcs(int id) {
    return incoming_arcs_indices[id];
}

// to steer expanded labels into right direction
double Network::heuristic_cost_component_forward(double lower_energy_bound, Label* next_label) {
    int start_time_of_day = static_cast<int>(next_label->departure_time) % (24 * 3600);
    int start_time_in_hours = int(std::floor(start_time_of_day / 3600));
    double recharge_price = *std::min_element(
            energy_prices.begin() + start_time_in_hours,
            energy_prices.end()
    );
    auto soc_min = static_cast<double>(std::get<0>(soc_bounds));
    double energy_to_charge = std::max(soc_min + lower_energy_bound - next_label->energy, 0.0);
    return energy_to_charge * recharge_price + lower_energy_bound * consumption_cost;
}

// to steer expanded labels into right direction
double Network::heuristic_cost_component_backward(double lower_energy_bound, Label* next_label, double init_soc) {
    int end_time_of_day = static_cast<int>(next_label->departure_time) % (24 * 3600);
    int end_time_in_hours = int(std::floor(end_time_of_day / 3600));
    double recharge_price = *std::min_element(
            energy_prices.begin(),
            energy_prices.begin() + end_time_in_hours
    );
    double energy_to_charge = std::max((next_label->energy) + lower_energy_bound - init_soc, 0.0);
    return energy_to_charge * recharge_price + lower_energy_bound * consumption_cost;
}

// Comparator function definition
bool Network::comp(std::tuple<int, int, int, LabelNode*, int>& left, std::tuple<int, int, int, LabelNode*, int>& right) {
    // Compare the absolute difference between the first elements against tolerance
    // First level: cost approx, second level: stop or depot or joint? third level: ID (depth first)
    if (std::get<4>(left) != std::get<4>(right)) {
        return std::get<4>(left) < std::get<4>(right);
    }
    else if (std::get<0>(left) != std::get<0>(right)) {
        return std::get<0>(left) > std::get<0>(right);
    }
    else if (std::get<1>(left) != std::get<1>(right)) {
        return std::get<1>(left) > std::get<1>(right);
    }
    else if (std::get<2>(left) != std::get<2>(right)) {
        return std::get<2>(left) < std::get<2>(right);
    }
    else {
        return std::get<3>(left)->current_label->label_type < std::get<3>(right)->current_label->label_type;
    }
}

std::priority_queue<std::tuple<int, int, int, LabelNode*, int>, std::vector<std::tuple<int, int, int, LabelNode*, int>>, decltype(&Network::comp)>
Network::createPriorityQueue() {
    // Define (and fill with first element) Priority Queue
    std::priority_queue<std::tuple<int, int, int, LabelNode*, int>, std::vector<std::tuple<int, int, int, LabelNode*, int>>, decltype(&comp)> U(
            comp);
    return U;
}

// we need to change this function and directly provide the next arc (extracted based on helper list)
Label* Network::get_next_label(
        std::tuple<int, int, Arc> current_arc,
        Label* current_label,
        const std::vector<double>& energyPrices
) {

    Arc arc = std::get<2>(current_arc);
    double recharged_energy = arc.recharged_energy;
    double consumed_energy = arc.consumed_energy;
    int time_to_dyn_charger = arc.time_to_dyn_charger;

    if (current_label->label_type == 1) {

        // extract vertex information
        int next_vertex_id = std::get<1>(current_arc);
        auto next_vertex_object = get_vertex(next_vertex_id);

        // prep some values
        auto arrival_time = current_label->departure_time + arc.time;
        double departure_time = std::get<0>(next_vertex_object.departure_time_window);

        if (departure_time < arrival_time) {
            departure_time = arrival_time;
        }
        double energy = current_label->energy - consumed_energy + recharged_energy;

        int pricing_time_of_day = (static_cast<int>(current_label->departure_time)+time_to_dyn_charger) % (24 * 3600);
        int pricing_time = int(std::floor(pricing_time_of_day / 3600));

        double cost = current_label->cost + energyPrices[pricing_time] * recharged_energy +
                      consumption_cost * consumed_energy;

        // check if energy will not go out of bounds, if so --> render label infeasible
        double soc_prof;
        double start_soc = current_label->energy;
        if (arc.consumption_sequence.size() > 1) {
            // Iterate from begin to one-before-end
            for (auto it = arc.consumption_sequence.begin(); it != std::prev(arc.consumption_sequence.end()); ++it) {
                soc_prof = start_soc - *it;
                if (soc_prof < std::get<0>(soc_bounds) || soc_prof > std::get<1>(soc_bounds)) {
                    energy = -1.0;
                    break;
                }
            }
        }

        Label* new_label = &label_collection.emplace_back(Label
                {arrival_time, departure_time, energy, cost,
                 static_cast<double>(current_label->consumed_energy - consumed_energy),
                 static_cast<double>(current_label->recharged_energy + recharged_energy),
                 static_cast<int>(1), 0.0,static_cast<int>(arc.crossed_dyn_station),
                });

        return new_label;
    }


    // extract vertex information
    int next_vertex_id = std::get<0>(current_arc);
    auto next_vertex_object = get_vertex(next_vertex_id);

    double departure_time = std::min(current_label->arrival_time - arc.time, std::get<1>(next_vertex_object.departure_time_window));
    double arrival_time = std::get<1>(next_vertex_object.departure_time_window);
    if (departure_time < arrival_time){
        arrival_time = departure_time;
    }

    // no partial recharging; energy field can be negative (in the backwards labels it is a lower bound)
    double energy = current_label->energy + consumed_energy - recharged_energy;

    int pricing_time_of_day = (static_cast<int>(departure_time)+time_to_dyn_charger) % (24 * 3600);

    // negative times will be invalid anyways
    int pricing_time = std::max(int(std::floor(pricing_time_of_day / 3600)),0);
    double cost = current_label->cost + energyPrices[pricing_time] * recharged_energy +
                  consumption_cost * consumed_energy;

    double headspace = std::min(current_label->headspace, std::get<1>(soc_bounds)-energy);

    // check if energy will not go out of bounds, if so --> render label infeasible
    double soc_prof;
    double start_soc = current_label->energy;
    if (arc.consumption_sequence.size() > 1) {
        // Skip the first element (last in reverse)
        for (auto rit = arc.consumption_sequence.rbegin(); rit != std::prev(arc.consumption_sequence.rend()); ++rit) {
            soc_prof = start_soc + *rit;
            if (soc_prof < std::get<0>(soc_bounds) || soc_prof > std::get<1>(soc_bounds)) {
                energy = -1.0;
                break;
            }
        }
    }

    Label* new_label = &label_collection.emplace_back(Label{arrival_time, departure_time, energy, cost,
                       static_cast<double>(current_label->consumed_energy - consumed_energy),
                       static_cast<double>(current_label->recharged_energy + recharged_energy),
                       static_cast<int>(-1), headspace,static_cast<int>(arc.crossed_dyn_station)});
    return new_label;
}

static int convertToInt(double num, int decimalPlaces) {
    double temp = num * std::pow(10.0, decimalPlaces);
    return std::trunc(temp);
}

static std::tuple<bool, std::optional<LabelNode*>> can_be_merged(
        const int v, LabelNode *l_q, const LabelTree &best_labels_forward,
        const LabelTree &best_labels_backward) {
    if (l_q->current_label->label_type == 1) {
        auto &labels = best_labels_backward.labels_by_cost[v];  // Access the vector of LabelWithBounds
        for (size_t idx = 0; idx < labels.size(); ++idx) {
            const auto &label = labels[idx];
            double soc_difference = l_q->current_label->energy - label.label_node->current_label->energy;
            // energy match is implicitly given if req. is met)
            if ((l_q->current_label->departure_time <= label.label_node->current_label->departure_time) &&
                (0 <= soc_difference) && (soc_difference <= label.label_node->current_label->headspace)) {
                return std::make_tuple(true, label.label_node);
            }
        }
        return std::make_tuple(false, std::nullopt);
    } else {
        // Try to find the labels for the given vertex v in the forward direction
        auto &labels = best_labels_forward.labels_by_cost[v];  // Access the vector of LabelWithBounds
        for (size_t idx = 0; idx < labels.size(); ++idx) {
            const auto &label = labels[idx];
            double soc_difference = label.label_node->current_label->energy - l_q->current_label->energy;
            // energy match is implicitly given if req. is met)
            if ((l_q->current_label->departure_time >= label.label_node->current_label->departure_time) &&
                (0 <= soc_difference) && (soc_difference <= l_q->current_label->headspace)) {
                return std::make_tuple(true, label.label_node);
            }
        }
        return std::make_tuple(false, std::nullopt);
    }
}


std::optional<std::tuple<LabelTree, LabelTree, int, LabelNode*>> Network::A_star_intermediate(
        const std::vector<double> &lower_bound,
        const Route &route,
        const double soc_init,
        const std::vector<double> &energyPrices
) {

    int depot_start = route.depot_start();
    int depot_end = route.depot_end();
    Vertex depot_vertex_start = get_vertex(depot_start);
    Vertex depot_vertex_end = get_vertex(depot_end);
    Label* forward_depot_label = &label_collection.emplace_back(Label{
            0,std::get<0>(depot_vertex_start.departure_time_window),soc_init,
            0,0,0,1, 0.0,-1
    });
    Label* backward_depot_label = &label_collection.emplace_back(Label{
            std::get<1>(depot_vertex_end.departure_time_window),
            std::get<1>(depot_vertex_end.departure_time_window),std::get<0>(soc_bounds),
            0,0,0,-1,std::get<1>(soc_bounds)-std::get<0>(soc_bounds),-1
    });

    LabelNode& forward_depot_label_node = label_node_collection.emplace_back(LabelNode{
            forward_depot_label,
            depot_start,
    });
    LabelNode& backward_depot_label_node = label_node_collection.emplace_back(LabelNode{
            backward_depot_label,
            depot_end,
    });

    // correct pointer to self
    forward_depot_label_node.precedent_label = &forward_depot_label_node;
    backward_depot_label_node.precedent_label = &backward_depot_label_node;

    bool heuristic_solution_available = false;

    // Initialise priority queue
    auto U = Network::createPriorityQueue();
    U.emplace(0, 0, 0, &forward_depot_label_node, 0);
    U.emplace(0, 0, 0, &backward_depot_label_node, 0);

    // Initialization of U and best_labels
    auto number_of_vertices = static_cast<int>(vertices.size());
    LabelTree best_labels_forward = LabelTree(number_of_vertices);
    LabelTree best_labels_backward = LabelTree(number_of_vertices);

    // keep track of progress
    int iter = 0;
    int iter_threshold = 2.5e5; //3e5;
    int deepest_forward = depot_start;
    int deepest_backward = depot_end;
    bool queue_resorted = false;
    int num_active_fw_labels = 1;
    int num_active_bw_labels = 1;

    int prioritize_fw = 1;

    while (!U.empty()) {

        iter += 1;

        // If no forward label in queue, problem must be infeasible
        // Does not hold for backward label because there we start with min soc; we conclude infeasibility in heuristic
        // case if no active bw labels are available
        if (num_active_fw_labels == 0 || (num_active_bw_labels == 0 && deepest_backward != depot_start && queue_resorted)){  //
            return std::nullopt;
        }

        // heuristic element: after XXX iterations we return the best known joint label
        if (iter > iter_threshold && heuristic_solution_available){
            std::optional<std::tuple<LabelTree, LabelTree, int, LabelNode*>> heuristic_result = std::nullopt;
            bool loop_exit = false;
            while (!loop_exit && !U.empty()){
                std::tuple<int, int, int, LabelNode*, int> topElement = U.top();
                U.pop(); // remove first element (pop has return type void)
                bool is_joint = std::get<1>(topElement)==-1;
                if (is_joint) {
                    LabelNode* l_q = std::get<3>(topElement);
                    int v_q = std::get<2>(topElement);
                    loop_exit = true;
                    heuristic_result = std::make_tuple(best_labels_forward, best_labels_backward, v_q, l_q);
                }
            }
            return heuristic_result;
        }

        // re-sort queue according to heuristic metric
        if (!queue_resorted && iter > iter_threshold && deepest_forward >= deepest_backward){
            // log that queue has been resorted
            queue_resorted = true;

            if (num_active_bw_labels < num_active_fw_labels && num_active_bw_labels > 0){
                prioritize_fw = -1;
            }

            // Create a temporary container to hold the modified elements
            std::vector<std::tuple<int, int, int, LabelNode*, int>> tempElements;

            // Copy elements from the priority queue to the temporary container
            while (!U.empty()) {
                tempElements.push_back(U.top());
                U.pop();
            }

            // Modify the first dimension (set it to zero for all elements)
            for (auto& elem : tempElements) {
                int node = std::get<2>(elem);
                LabelNode* label_node = std::get<3>(elem);
                if (label_node->current_label->label_type==-1){
                    node -= depot_end;
                    node *= -1;
                }

                // Set the first dimension to SOC - Min SOC - Min. Consumption to prioritize labels with right SOC
                double lb, min_soc;
                if (label_node->current_label->label_type==-1){
                    lb = lower_bound[depot_start] - lower_bound[node];
                    std::get<0>(elem) = convertToInt(std::abs(soc_init - lb - label_node->current_label->energy),3); // - label_node.current_label->propagated_requirement
                    std::get<4>(elem) = -1 * prioritize_fw;
                }
                else {
                    lb = lower_bound[node];
                    min_soc=std::get<0>(soc_bounds);
                    std::get<0>(elem) = convertToInt(std::abs(lb + min_soc - label_node->current_label->energy),3);
                    std::get<4>(elem) = 1 * prioritize_fw;
                }
            }

            // Rebuild the priority queue with modified elements
            for (const auto& elem : tempElements) {
                U.push(elem);
            }
        }

        // get the first element of the queue and remove it afterwards; unfortunately in separated function
        std::tuple<int, int, int, LabelNode*, int> topElement = U.top();
        U.pop(); // remove first element (pop has return type void)

        LabelNode* l_q = std::get<3>(topElement);
        int v_q = std::get<2>(topElement);
        bool is_joint = std::get<1>(topElement)==-1;

        if (is_joint) {
            return std::make_tuple(best_labels_forward, best_labels_backward, v_q, l_q);
        }

        // we put joint labels always with positive v_q into queue
        if (l_q->current_label->label_type==-1){
            v_q -= depot_end;
            v_q *= -1;
            num_active_bw_labels -= 1;
        }
        else if (l_q->current_label->label_type==1){
            num_active_fw_labels -= 1;
        }

        // forward label can yield a feasible path
        // backward labels need to be merged (because of how propagate path function is designed, and soc init is fixed
        // no need to fix this, it is an edge case from the test cases
        if ((v_q == depot_end && l_q->current_label->label_type==1)) {
            return std::make_tuple(best_labels_forward, best_labels_backward, v_q, l_q);
        }

        // Some of these dominance checks can be further delayed or ideally avoided
        std::tuple<bool, std::vector<LabelTree::LabelNodeWithBounds>::iterator> domination_results;
        if (l_q->current_label->label_type == 1) {
            domination_results = best_labels_forward.is_dominated(l_q, v_q);
        }
        else {
            domination_results = best_labels_backward.is_dominated(l_q, v_q);
        }

        // dominated labels are not expanded
        if (std::get<0>(domination_results)) {
            continue;
        }

        // if the end depot is reached, go out of the loop
        if (l_q->current_label->label_type == 1) {

            // we only keep labels that have a valid successor label
            bool has_valid_successor = false;

            // explore the Successors (via iterating the outgoing arcs)
            auto outgoing_arcs_view = get_outgoing_arcs(v_q);

            // Iterate over the outgoing arcs view using the returned iterators
            for (auto it = outgoing_arcs_view.first; it != outgoing_arcs_view.second; ++it) {

                // Dereference the iterator to access each arc tuple
                std::tuple<int, int, Arc> current_arc = *it;

                // create next label
                Label* next_label = get_next_label(current_arc, l_q->current_label, energyPrices);

                // Check if the label is valid
                Vertex successor_vertex = get_vertex(std::get<1>(current_arc));
                bool valid = next_label->valid_label(successor_vertex, soc_bounds);

                if (valid) {
                    has_valid_successor = true;

                    int prioritize_forward = 0;

                    // Compute the key of the successor
                    double next_key;
                    if (!queue_resorted) {
                        next_key = heuristic_cost_component_forward(
                                lower_bound[std::get<1>(current_arc)], next_label
                        ) + next_label->cost;
                    }
                    else {
                        next_key = std::abs(lower_bound[std::get<1>(current_arc)] + std::get<0>(soc_bounds) - next_label->energy);
                        prioritize_forward = 1*prioritize_fw;
                    }

                    // Push to the priority queue U
                    // check if successor vertex is stop or depot and prioritise stops and depots in label setting
                    // Attention: this relies on stops and depots being associated with the smallest IDs
                    int is_on_route = 1;
                    if (std::get<1>(current_arc) <= depot_end) {
                        is_on_route = 0;
                    }
                    LabelNode& next_label_node = label_node_collection.emplace_back(LabelNode{next_label, v_q, l_q});
                    U.emplace(convertToInt(next_key, 3), is_on_route, std::get<1>(current_arc), &next_label_node, prioritize_forward);
                    num_active_fw_labels += 1;
                }
            }
            if (has_valid_successor || v_q == depot_end){
                best_labels_forward.add(l_q, v_q, std::get<1>(domination_results));
                if (v_q <= depot_end){
                    deepest_forward = std::max(v_q, deepest_forward);
                }
            }
        }

        else {

            // explore the Successors (via iterating the outgoing arcs)
            std::vector<int> incoming_arcs = get_incoming_arcs(v_q);

            // Iterate over the outgoing arcs view using the returned iterators
            for (int it: incoming_arcs) {

                // Dereference the iterator to access each arc tuple
                std::tuple<int, int, Arc> current_arc = arcs[it];

                // create next label
                Label* next_label = get_next_label(current_arc, l_q->current_label, energyPrices);

                // Check if the label is valid
                Vertex successor_vertex = get_vertex(std::get<0>(current_arc));
                bool valid = next_label->valid_label(successor_vertex, soc_bounds);

                if (valid) {
                    int prioritize_forward = 0;

                    // Compute the key of the successor
                    double next_key;
                    if (!queue_resorted) {
                        next_key = heuristic_cost_component_backward(
                                lower_bound[0] - lower_bound[std::get<0>(current_arc)], next_label, soc_init
                        ) + next_label->cost;
                    }
                    else {
                        next_key = std::abs(soc_init - lower_bound[0] + lower_bound[std::get<0>(current_arc)] - next_label->energy); //  - next_label->propagated_requirement
                        prioritize_forward = -1 * prioritize_fw;
                    }

                    // Push to the priority queue U
                    // check if successor vertex is stop or depot and prioritise stops and depots in label setting
                    // Attention: this relies on stops and depots being associated with the smallest IDs
                    int is_on_route = 1;
                    if (std::get<0>(current_arc) <= depot_end) {
                        is_on_route = 0;
                    }
                    LabelNode& next_label_node = label_node_collection.emplace_back(LabelNode{next_label, v_q, l_q});
                    U.emplace(convertToInt(next_key, 3), is_on_route, depot_end - std::get<0>(current_arc), &next_label_node, prioritize_forward);
                    num_active_bw_labels += 1;
                }
            }
            // Add to best_labels
            best_labels_backward.add(l_q, v_q, std::get<1>(domination_results));
            if (v_q <= depot_end) {
                deepest_backward = std::min(v_q, deepest_backward);
            }
        }

        // check if this label can be merged
        std::tuple<bool, std::optional<LabelNode*>> merging_result = can_be_merged(
                v_q, l_q, best_labels_forward, best_labels_backward
                );

        if (std::get<0>(merging_result)) {
            // if above holds, this is always true - two separate checks to make sure at runtime only first condition
            // is checked
            if (std::get<1>(merging_result).has_value()) {
                Label* complementary_label = std::get<1>(merging_result).value()->current_label;
                LabelNode& joint_label_node = label_node_collection.emplace_back(LabelNode{l_q->current_label, l_q->precedent_vertex, l_q->precedent_label});
                double next_key;
                if (!queue_resorted) {
                    next_key = l_q->current_label->cost + complementary_label->cost;
                }
                else {
                    // enforce new label to be on top of queue
                    next_key = -1;
                }
                U.emplace(convertToInt(next_key, 3), -1, v_q, &joint_label_node, 0);
                heuristic_solution_available = true;
            }
        }
    }
    return std::nullopt;
}


std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
Network::propagate_path(std::tuple<LabelTree, LabelTree, int, LabelNode*> raw_paths, Route r){

    int cut_vertex = std::get<2>(raw_paths);
    LabelNode* label_node = std::get<3>(raw_paths);
    LabelTree forward_path = std::get<0>(raw_paths);
    LabelTree backward_path = std::get<1>(raw_paths);

    LabelNode *fw_label_node = nullptr;
    LabelNode *bw_label_node = nullptr;

    // save the merging if instance was solved by forward path alone
    if (cut_vertex == r.depot_end()){
        std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
        fw = forward_path.forward_propagate_path(r, cut_vertex, label_node);
        return fw;
    }

    if (label_node->current_label->label_type == 1){
        for (size_t idx = 0; idx < backward_path.labels_by_cost[cut_vertex].size(); ++idx) {
            LabelNode* lb_node = backward_path.labels_by_cost[cut_vertex][idx].label_node;

            double soc_difference = label_node->current_label->energy - lb_node->current_label->energy;
            bool time_matches = lb_node->current_label->departure_time >= label_node->current_label->departure_time;

            // Placeholder for condition check (energy match is implicitly given if req. is met)
            if (time_matches && (0 <= soc_difference) && (soc_difference <= lb_node->current_label->headspace)) {
                bw_label_node = lb_node;
                break;  // Remove this if you want to check all matches
            }
        }
        fw_label_node = label_node;
    }
    else {
        bw_label_node = label_node;
        for (size_t idx = 0; idx < forward_path.labels_by_cost[cut_vertex].size(); ++idx) {

            LabelNode* lb_node = forward_path.labels_by_cost[cut_vertex][idx].label_node;

            double soc_difference = lb_node->current_label->energy - label_node->current_label->energy;
            bool time_matches = lb_node->current_label->departure_time <= label_node->current_label->departure_time;

            // Placeholder for condition check (energy match is implicitly given if req. is met)
            if (time_matches && (0 <= soc_difference) && (soc_difference <= bw_label_node->current_label->headspace)) {
                fw_label_node = lb_node;
                break;  // Remove this if you want to check all matches
            }
        }
    }
    double soc_offset = fw_label_node->current_label->energy - bw_label_node->current_label->energy;
    std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
    fw = forward_path.forward_propagate_path(r, cut_vertex, fw_label_node);
    std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
    bw = backward_path.backward_propagate_path(r, cut_vertex, bw_label_node, soc_offset);

    std::vector<std::tuple<int, double, double, double, double, double, int>> fw_path = std::get<0>(fw);
    std::vector<std::tuple<int, double, double, double, double, double, int>> bw_path = std::get<0>(bw);

    double fw_cost = std::get<1>(fw);
    double bw_cost = std::get<1>(bw);

    // Combine paths
    std::vector<std::tuple<int, double, double, double, double, double, int>> path;
    path.reserve(fw_path.size() + bw_path.size()); // Reserve memory for efficiency
    path.insert(path.end(), fw_path.begin(), fw_path.end());
    path.insert(path.end(), bw_path.begin(), bw_path.end());

    // Add costs
    double cost = fw_cost + bw_cost;

    // Update elements
    for (size_t idx = fw_path.size(); idx < path.size(); ++idx) {
        std::get<4>(path[idx]) += std::get<4>(fw_path.back());
        std::get<5>(path[idx]) += std::get<5>(fw_path.back());
    }
    return std::make_tuple(path, cost);
}
