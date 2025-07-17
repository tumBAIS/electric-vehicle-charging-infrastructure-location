#include "vertex.h"

#include <utility>
Vertex::Vertex(int i, double d, double d1, double d2, double d3):
    id(i), departure_time_window(std::make_tuple(d, d1)), arrival_time_window(std::make_tuple(d2, d3)) {

}