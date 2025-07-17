#include "catch.hpp"
#include "label.h"
#include <iostream>
#include "network.h"

TEST_CASE("Label") {
    LabelTree t = static_cast<LabelTree>(LabelTree(3));
    int i = 0;


    Label l1 = Label{1.0f, 5.f, 120, 16.0f, 2.0f, 1.0f, 1, 0, 0};
    Label l2 = Label{2.0f, 8.f, 130, 17.0f, 2.0f, 1.0f,1, 0, 0};
    Label l3 = Label{3.0f, 2.f, 80, 18.0f, 0.0f, 0.0f, 1, 0, 0};

    // ToDo Last element add

    LabelNode ln1 = {&l1, 0};
    LabelNode ln2 = {&l2, 0};
    LabelNode ln3 = {&l3, 0};

    // dominance check required before adding
    auto dominance_result1 = t.is_dominated(&ln1, i);
    t.add(&ln1, i, std::get<1>(dominance_result1));

    // dominance check required before adding
    auto dominance_result2 = t.is_dominated(&ln2, i);
    t.add(&ln2, i, std::get<1>(dominance_result2));

    // dominance check required before adding
    auto dominance_result3 = t.is_dominated(&ln3, i);
    t.add(&ln3, i, std::get<1>(dominance_result3));


    SECTION("add") {
        std::vector<LabelTree::LabelNodeWithBounds> label_bounds = t.labels_by_cost[i];

        double energy_bound = 0.0;
        double time_bound = 100.0;

        for (auto elem: label_bounds) {
            REQUIRE(elem.energy_bound>energy_bound);
            REQUIRE(elem.time_bound<time_bound);
            REQUIRE(elem.label_node->current_label->energy<=elem.energy_bound);
            REQUIRE(elem.label_node->current_label->departure_time>=elem.time_bound);
        }
    }

    // all labels have predecent vertex 0 and point to themselves because not relevant in test case
    SECTION("is_dominated") {
        auto label = Label{1.0f, 10.f, 20, 16.5f, 2.0f, 1.0f, 1, 0.0, 0};
        LabelNode dominated_label = {
                &label,
                0,
        }; // should be inserted between l1 and l2

        //Energy is not dominated, time is dominated. Expected result: False
        auto label1 = Label{1.0f, 10.f, 200, 16.5f, 2.0f, 1.0f,1, 0.0, 0};
        LabelNode non_dominated_label_1 = {
                &label1,
                0,
        };

        //Both not dominated
        Label label2 = Label{1.0f, 10.f, 200, 16.5f, 2.0f, 1.0f,1, 0.0, 0};
        LabelNode non_dominated_label_2 = {
                &label2,
                0,
        };

        //Is inserted at the beginning
        Label label3 = Label{1.0f, 50.f, 10, 1.0f, 2.0f, 1.0f,1, 0.0, 0};
        LabelNode non_dominated_label_3 = {
                &label3,
                0,
        };

        REQUIRE(true == std::get<0>(t.is_dominated(&dominated_label, 0)));
        REQUIRE(false == std::get<0>(t.is_dominated(&non_dominated_label_1, 0)));
        REQUIRE(false == std::get<0>(t.is_dominated(&non_dominated_label_2, 0)));
        REQUIRE(false == std::get<0>(t.is_dominated(&non_dominated_label_3, 0)));
    }
}


TEST_CASE("Network") {///*
    // initialize network
    std::vector<Vertex> vertices;
    vertices.push_back({0, 0, 0, 0, 1e9});
    vertices.push_back({1, 250, 300, 0, 1e9});
    vertices.push_back({2, 250, 1200, 0, 1e9});
    vertices.push_back({3, 1000, 1200, 0, 1e9});
    vertices.push_back({4, 2000, 2270, 0, 1e9});
    vertices.push_back({5, 2000, 2700, 0, 1e9});

    vertices.push_back({6, 0, 1097.142857142857, 0, 1e9});
    vertices.push_back({7, 0, 1097.142857142857, 0, 1e9});
    vertices.push_back({8, 0, 1097.142857142857, 0, 1e9});


    int number_of_vertices = vertices.size();
    std::vector<std::tuple<int, int, Arc>> arcs;

    arcs.push_back({0,1,Arc(288.0, 28.0, 0.0, 0, 0, {0})});
    arcs.push_back({1,6,Arc(102.85714285714285, 14.0, 0.0, 0, 0, {0})});
    arcs.push_back({2,3,Arc(102.85714285714285, 14.0, 0.0, 0, 0, {0})});
    arcs.push_back({3,4,Arc(1080.0, 42.0, 0.0, 0, 0, {0})});
    arcs.push_back({4,5,Arc(60.0, 7.0, 0.0, 0, 0, {0})});
    arcs.push_back({6,2,Arc(0.0, 0.0, 0.0, 0, 0, {0})});
    arcs.push_back({6,7,Arc(5, 0.0, 41.666666666666664, 0, 0, {0})});
    arcs.push_back({7,2,Arc(0.0, 0.0, 0.0, 0, 0, {0})});
    arcs.push_back({7,8,Arc( 5, 0.0, 41.666666666666664, 0, 0, {0})});
    arcs.push_back({8,2,Arc( 0.0, 0.0, 0.0, 0, 0, {0})});

    std::vector<double> energy_prices(24, 1.0);
    double consumption_cost = 0.15;
    double soc_init = 100;
    std::tuple<double,double> soc_bounds(30, 100);

    Network n = {vertices, arcs, energy_prices, consumption_cost, soc_bounds};

    std::vector<int> raw_route = {0,1,2,3,4,5};
    Route r = {raw_route};

    std::vector<double> lower_bound = {105.0, 77.0, 63.0, 49.0, 7.0, 0, 63.0, 63.0, 63.0};
    float upper_bound = 100;//41.66666666666;

    bool restrict_arrival_time = false;

    SECTION("A_star");
    std::optional<std::tuple<LabelTree, LabelTree, int, LabelNode*>> lt = n.A_star_intermediate(
            lower_bound,r, soc_init, energy_prices
            );

    // declare return type
    std::optional<std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>> result;
    if (lt.has_value()) {
        std::tuple<LabelTree, LabelTree, int, LabelNode*> lt_value = lt.value();
        result = n.propagate_path(lt_value, r);
    }
    else {
        result = std::nullopt;
    };

    result;
//*/
}

TEST_CASE("PriorityQueue") {///*
    LabelTree t = static_cast<LabelTree>(LabelTree(3));
    int i = 0;

    Label l1 = Label{1.0f, 5.f, 120, 16.0f, 2.0f, 1.0f, 1,0, 0};
    Label l2 = Label{2.0f, 8.f, 130, 17.0f, 2.0f, 1.0f, 1,0, 0};
    Label l3 = Label{3.0f, 2.f, 80, 18.0f, 0.0f, 0.0f, 1,0, 0};
    Label l4 = Label{3.0f, 2.f, 80, 17.0f, 0.0f, 0.0f, 1,0, 0};

    // ToDo Last element add

    LabelNode ln1 = {&l1, 5};
    LabelNode ln2 = {&l2, 5};
    LabelNode ln3 = {&l3, 5};
    LabelNode ln4 = {&l4, 5};

    auto U = Network::createPriorityQueue();
    U.emplace(0, 0 ,0, &ln1, 0);
    U.emplace(4263460, 0, 1, &ln2, 0);
    U.emplace(4263458, 0, 1, &ln3, 0);
    U.emplace(4263458, 1, 1, &ln4, 0);
    U.emplace(4263458, 0 ,2, &ln4, 0);

    // top on by one
    auto r1 = U.top();
    U.pop();
    REQUIRE(std::get<0>(r1)==0);

    auto r2 = U.top();
    U.pop();
    REQUIRE(std::get<0>(r2)==4263458);
    REQUIRE(std::get<2>(r2)==2);

    auto r3 = U.top();
    U.pop();
    REQUIRE(std::get<0>(r3)==4263458);
    REQUIRE(std::get<2>(r3)==1);

    auto r4 = U.top();
    U.pop();
    REQUIRE(std::get<0>(r4)==4263458);
    REQUIRE(std::get<1>(r4)==1);

    auto r5 = U.top();
    U.pop();
    REQUIRE(std::get<0>(r5)==4263460);
}