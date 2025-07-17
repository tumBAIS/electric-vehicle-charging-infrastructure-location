//
// Created by sikob on 25/02/2024.
//

#ifndef ROUTE_H
#define ROUTE_H

#include <vector>

#include "vertex.h"

class Route {
public:
    const std::vector<int> stops;
    const int length;

    Route(const std::vector<int> stops_);

    int depot_start() const;

    int depot_end() const;
};


#endif //ROUTE_H
