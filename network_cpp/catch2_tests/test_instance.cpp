#include "catch.hpp"
#include "label.h"
#include "network.h"
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>

TEST_CASE("RunInstance") {
    //SECTION("ReadData");

    std::vector<double> lower_bound;
    std::vector<int> raw_route;
    std::vector<double> energy_prices;
    double soc_init;
    double consumption_cost;
    std::vector<std::tuple<int, double, double, double, double>> raw_vertices; // Assuming 3D vertices raw_vertices;
    std::vector<std::tuple<int, int, double, double, double, int, std::vector<double>, int>> raw_arcs; // Assuming 3D vertices raw_arcs;
    bool restrict_arrival_time;
    std::tuple<double,double> soc_bounds;

    std::string instance_name = "08_42_20_866";
    std::string subdirectory = "../../instances/" + instance_name;

    std::string filename = subdirectory + "/config.txt";
    std::ifstream configFile(filename);
    if (!configFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for reading." << std::endl;
        return;
    }

    std::cout << "Test instance: " << subdirectory << std::endl;


    std::string line;
    while (std::getline(configFile, line)) {
        std::istringstream iss(line);
        std::string key;
        if (!(iss >> key)) {
            continue;
        } else if (key == "soc_init") {
            iss >> soc_init;
        } else if (key == "soc_min") {
            double soc_min_value;
            iss >> soc_min_value;
            std::get<0>(soc_bounds) = soc_min_value;
        } else if (key == "soc_max") {
            double soc_max_value;
            iss >> soc_max_value;
            std::get<1>(soc_bounds) = soc_max_value;
        } else if (key == "consumption_cost") {
            iss >> consumption_cost;
        } else if (key == "restrict_arrival_time") {
            iss >> std::boolalpha >> restrict_arrival_time;
        } else if (key == "energy_prices") {
            double value;
            while (iss >> value) {
                energy_prices.push_back(value);
            }
        } else if (key == "lower_bound") {
            double value;
            while (iss >> value) {
                lower_bound.push_back(value);
            }
        }
    }
    configFile.close();

    filename = subdirectory + "/vertices.txt";
    std::ifstream verticesFile(filename);
    if (!verticesFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for reading." << std::endl;
        return;
    }

    while (std::getline(verticesFile, line)) {
        std::istringstream iss(line);
        std::string stringValue;
        std::vector<double> vertexValues;

        while (iss >> stringValue) {
            double value;
            if (stringValue == "inf") {
                value = std::numeric_limits<double>::infinity();
            } else {
                value = std::stod(stringValue);
            }
            vertexValues.push_back(value);
        }

        // Assuming each vertex has exactly 3 values (x, y, z)
        if (vertexValues.size() == 5) {
            raw_vertices.push_back(std::make_tuple(
                    static_cast<int>(vertexValues[0]),
                    static_cast<double>(vertexValues[1]),
                    static_cast<double>(vertexValues[2]),
                    static_cast<double>(vertexValues[3]),
                    static_cast<double>(vertexValues[4])));
        } else {
            std::cerr << "Error: Each vertex should have exactly 3 values." << std::endl;
        }
    }

    verticesFile.close();

    filename = subdirectory + "/arcs.txt";
    std::ifstream arcsFile(filename);
    if (!arcsFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for reading." << std::endl;
        return;
    }

    while (std::getline(arcsFile, line)) {
        std::istringstream iss(line);
        std::string stringValue;
        std::vector<double> arcValues;

        // Parse the first 6 values of the arc
        for (int i = 0; i < 6; ++i) {
            if (!(iss >> stringValue)) {
                std::cerr << "Error: Unexpected end of line when reading arc values." << std::endl;
                return;
            }

            double value;
            if (stringValue == "inf") {
                value = std::numeric_limits<double>::infinity();
            } else {
                value = std::stod(stringValue); // Use std::stod to convert string to double
            }
            arcValues.push_back(value);
        }

        // The next value is the size of the vector (not the vector itself)
        size_t vectorSize;
        if (!(iss >> vectorSize)) {
            std::cerr << "Error: Could not read vector size for arc." << std::endl;
            return;
        }

        // Now, parse the vector elements based on the vector size
        std::vector<double> dynamicVector;
        for (size_t i = 0; i < vectorSize; ++i) {
            if (!(iss >> stringValue)) {
                std::cerr << "Error: Not enough vector elements in arc." << std::endl;
                return;
            }

            double value;
            if (stringValue == "inf") {
                value = std::numeric_limits<double>::infinity();
            } else {
                value = std::stod(stringValue); // Use std::stod to convert string to double
            }
            dynamicVector.push_back(value);
        }

        // Now, parse the last value (after the vector)
        double lastValue;
        if (!(iss >> stringValue)) {
            std::cerr << "Error: Missing final value after vector elements." << std::endl;
            return;
        }
        if (stringValue == "inf") {
            lastValue = std::numeric_limits<double>::infinity();
        } else {
            lastValue = std::stod(stringValue);
        }

        // Now check that the line has finished correctly
        if (iss >> stringValue) {
            std::cerr << "Error: Extra data at the end of the line after parsing arc." << std::endl;
            return;
        }

        // Reconstruct the arc tuple and push it into the list
        raw_arcs.push_back(std::make_tuple(
                static_cast<int>(arcValues[0]),
                static_cast<int>(arcValues[1]),
                static_cast<double>(arcValues[2]),
                static_cast<double>(arcValues[3]),
                static_cast<double>(arcValues[4]),
                static_cast<int>(arcValues[5]),
                dynamicVector,
                static_cast<int>(lastValue)  // Now use lastValue here
        ));
    }

    arcsFile.close();

    filename = subdirectory + "/route.txt";
    std::ifstream routeFile(filename);
    if (!routeFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for reading." << std::endl;
        return;
    }
    raw_route.clear();
    int value;
    while (routeFile >> value) {
        raw_route.push_back(value);
    }
    routeFile.close();

    //SECTION("BuildNetwork");
    int number_of_vertices = static_cast<int>(raw_vertices.size());

    // declare vector of vertices and arcs and allocate memory address immediately
    std::vector<Vertex> vertices;
    std::vector<std::tuple<int, int, Arc>> arcs;


    // initialise vertex objects (index in vector corresponds to ID of vertex)
    for (auto raw_vertex: raw_vertices) {
        Vertex vertex = {
                std::get<0>(raw_vertex),
                std::get<1>(raw_vertex),
                std::get<2>(raw_vertex),
                std::get<3>(raw_vertex),
                std::get<4>(raw_vertex),
        };
        vertices.push_back(vertex);
    }

    auto end_vertices = std::chrono::steady_clock::now();

    for (auto raw_arc: raw_arcs) {
        auto start = std::get<0>(raw_arc);
        auto end = std::get<1>(raw_arc);
        auto time = std::get<2>(raw_arc);
        auto consumed_energy = std::get<3>(raw_arc);
        auto recharged_energy = std::get<4>(raw_arc);
        auto crossed_dyn_station = std::get<5>(raw_arc);
        auto consumption_sequence = std::get<6>(raw_arc);
        auto time_to_dyn_charger = std::get<7>(raw_arc);

        // Create an instance of Arc
        Arc arc = {time, consumed_energy, recharged_energy, crossed_dyn_station, time_to_dyn_charger, consumption_sequence};

        // Create a tuple representing the arc and add it to the vector
        arcs.emplace_back(start, end, arc);
    }

    // Sort the vector of tuples based on ID origin and then ID target
    std::sort(
            arcs.begin(), arcs.end(), [](const auto &a, const auto &b
            ) {
                // Compare IDs of origin vertices
                if (std::get<0>(a) != std::get<0>(b)) {
                    return std::get<0>(a) < std::get<0>(b);
                }
                // If origin IDs are equal, compare IDs of target vertices
                return std::get<1>(a) < std::get<1>(b);
            });

    // Initialise network
    Network n = {
            vertices, arcs, energy_prices, consumption_cost, soc_bounds
    };
    Route r = {raw_route};

    //SECTION("RunAStar");

    // declare return type
    std::optional<std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>> result;
    std::optional<std::tuple<LabelTree, LabelTree, int, LabelNode*>> lt = n.A_star_intermediate(
            lower_bound, r, soc_init, energy_prices
    );

    if (lt.has_value()) {
        std::tuple<LabelTree, LabelTree, int, LabelNode*> lt_value = lt.value();
        std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double> result_ = n.propagate_path(lt_value, r);
        std::cout << "Final value: "  << std::fixed << std::setprecision(6) << std::get<1>(result_) << std::endl;
    }
    else {
        result = std::nullopt;
    };

}