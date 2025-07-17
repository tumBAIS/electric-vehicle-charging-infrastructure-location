//
// Created by sikob on 25/02/2024.
//

#ifndef STRATEGIC_PROBLEM_ARC_H
#define STRATEGIC_PROBLEM_ARC_H

class Arc {
public:
    Arc(double d, double d1, double d2, int i, int j, std::vector<double> cons);

    double time;
    double consumed_energy;
    double recharged_energy;
    int crossed_dyn_station;
    int time_to_dyn_charger;
    std::vector<double> consumption_sequence;
};


#endif //STRATEGIC_PROBLEM_ARC_H
