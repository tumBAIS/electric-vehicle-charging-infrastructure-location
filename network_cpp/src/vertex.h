//
// Created by sikob on 25/02/2024.
//

#ifndef STRATEGIC_PROBLEM_VERTEX_H
#define STRATEGIC_PROBLEM_VERTEX_H

#include <tuple>
#include <vector>

class Vertex {
public:
    Vertex(int i, double d, double d1, double d2, double d3);

    int id;
    std::tuple<double, double> departure_time_window;
    std::tuple<double, double> arrival_time_window;
};

#endif //STRATEGIC_PROBLEM_VERTEX_H
