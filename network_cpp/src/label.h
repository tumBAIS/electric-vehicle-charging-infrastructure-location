#ifndef LABEL_H
#define LABEL_H

#include <vector>
#include <list>
#include <optional>
#include "vertex.h"
#include "route.h"

class Label {
public:
    double arrival_time;
    double departure_time;
    double energy;
    double cost;
    double consumed_energy;
    double recharged_energy;
    int label_type; // 1 forward, -1 backward
    double headspace;
    int crossed_dyn_station;

    // New precomputed values
    double effective_energy;
    double effective_time;

    // Constructor to initialize effective values
    Label(double arrival_time, double departure_time, double energy, double cost,
          double consumed_energy, double recharged_energy, int label_type,
          double headspace, int crossed_dyn_station)
            : arrival_time(arrival_time), departure_time(departure_time), energy(energy),
              cost(cost), consumed_energy(consumed_energy), recharged_energy(recharged_energy),
              label_type(label_type), headspace(headspace), crossed_dyn_station(crossed_dyn_station),
              effective_energy(energy * label_type),
              effective_time(departure_time * label_type) {}

    bool operator==(const Label& other) const;
    bool operator<(const Label& right) const;
    bool valid_label(const Vertex& vertex, std::tuple<double, double> soc_bounds) const;
};

class LabelNode {
public:
    Label* current_label;
    int precedent_vertex;
    LabelNode* precedent_label;

    // Constructor
    LabelNode(Label* label, int vertex)
            : current_label(label), precedent_vertex(vertex), precedent_label(this) {} // Self-reference

    // Alternative constructor where all three arguments are provided
    LabelNode(Label* label, int vertex, LabelNode* prev_label)
            : current_label(label), precedent_vertex(vertex), precedent_label(prev_label) {}

    bool operator==(const LabelNode& other) const;
    bool operator<(const LabelNode& other) const;
};

class LabelTree {
public:
    struct LabelNodeWithBounds {
        LabelNode* label_node;
        double energy_bound;
        double time_bound;

        LabelNodeWithBounds(LabelNode* label_node, double energy, double departure_time)
                : label_node(const_cast<LabelNode *>(label_node)), energy_bound(energy), time_bound(departure_time) {}
    };

    //std::vector<std::vector<LabelNode>> labels;
    std::vector<std::vector<LabelNodeWithBounds>> labels_by_cost;

    // constructor: reserve memory for vectors
    explicit LabelTree(int n_nodes);

    // some public functions
    void add(LabelNode* label_node, int current_vertex, std::vector<LabelNodeWithBounds>::iterator);

    std::tuple<bool, std::vector<LabelNodeWithBounds>::iterator>
    is_dominated(LabelNode* label_node, int current_vertex);

    std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
    backward_propagate_path(Route& route, int cut_vertex, LabelNode* best_scenario, double soc_offset);

    std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
    forward_propagate_path(Route& route, int cut_vertex, LabelNode* best_scenario);
};

#endif /* LABEL_H */

