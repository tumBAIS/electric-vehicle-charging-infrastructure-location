#include <vector>
#include <string>
#include <tuple>
#include <set>
#include <queue>
#include <optional>


#include "vertex.h"
#include "arc.h"
#include "route.h"
#include "label.h"

class Network {

private:
    // Comparator function for priority queue
    static bool comp(std::tuple<int, int, int, LabelNode*, int>& left, std::tuple<int, int, int, LabelNode*, int>& right);

    double heuristic_cost_component_forward(double lower_energy_bound, Label* next_label);
    double heuristic_cost_component_backward(double lower_energy_bound, Label* next_label, double soc_init);

    Label* get_next_label(
                std::tuple<int, int, Arc> current_arc,
                Label* current_label,
                const std::vector<double>& energyPrices
            );

public:
    const std::vector<Vertex> vertices;
    const std::vector<std::tuple<int, int, Arc>> arcs;
    std::vector<std::tuple<int, int>> outgoing_arcs_indices;
    std::vector<std::vector<int>> incoming_arcs_indices;

    const std::vector<double> energy_prices;
    const double consumption_cost;
    const std::tuple<double, double> soc_bounds;

    std::deque<Label> label_collection;
    std::deque<LabelNode> label_node_collection;

    // Member function for defining the priority queue
    static std::priority_queue<std::tuple<int, int, int, LabelNode*, int>, std::vector<std::tuple<int, int, int, LabelNode*, int>>, decltype(&comp)>
    createPriorityQueue();

    Network(std::vector<Vertex> vertices_,
            std::vector<std::tuple<int, int, Arc>> arcs_,
            std::vector<double> energy_prices_,
            double consumption_cost_,
            std::tuple<double, double> soc_bounds_);

    Vertex get_vertex(int id);

    std::pair<std::vector<std::tuple<int, int, Arc>>::const_iterator, std::vector<std::tuple<int, int, Arc>>::const_iterator>
    get_outgoing_arcs(int id);

    // Declaration for get_incoming_arcs
    std::vector<int>
    get_incoming_arcs(int id);

    std::optional<std::tuple<LabelTree, LabelTree, int, LabelNode*>> A_star_intermediate(
            const std::vector<double> &lower_bound,
            const Route &route,
            double soc_init,
            const std::vector<double> &energyPrices);

    static std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>
    propagate_path(std::tuple<LabelTree, LabelTree, int, LabelNode*> raw_paths, Route r);
};

