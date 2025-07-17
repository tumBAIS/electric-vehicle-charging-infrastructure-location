#include "route.h"

Route::Route(const std::vector<int> stops_) : stops(stops_), length(stops_.size()){}

int Route::depot_start() const {
    return stops.front();
}

int Route::depot_end() const {
    return stops.back();
}