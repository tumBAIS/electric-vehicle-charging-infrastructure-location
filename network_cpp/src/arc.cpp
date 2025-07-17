#include <string>
#include <optional>
#include <vector>
#include "arc.h"


Arc::Arc(double d, double d1, double d2, int i, int j, std::vector<double> cons):
time(d), consumed_energy(d1), recharged_energy(d2), crossed_dyn_station(i), time_to_dyn_charger(j),
consumption_sequence(cons) {

}
