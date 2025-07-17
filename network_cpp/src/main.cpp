#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "network.h"
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include "vertex.h"

// for parallel execution
#include <future>
#include <atomic>

// Threadpool
#include "../CTPL/ctpl_stl.h"
#include <tuple>
//#include <mutex>
#include <optional>


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)



// namespace
namespace py = pybind11;

// Define the function signature for `spprc`
using InputType = std::vector<std::tuple<
        std::vector<double>, std::vector<int>, std::vector<double>,
        double, double,
        std::vector<std::tuple<int, double, double, double, double>>,
        std::vector<std::tuple<int, int, double, double, double, int, std::vector<double>, int>>,
        std::tuple<double, double>>>;
using ResultType = std::optional<std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>>;


// Function to get the current time as a formatted string
std::string current_time() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%H:%M:%S")
       << "." << std::setw(3) << std::setfill('0') << milliseconds.count();
    return ss.str();
}

std::string get_time() {
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

    std::string oss_s = oss.str();
    // Replace colons and periods with underscores
    for (char& c : oss_s) {
        if (c == ':' || c == '.') {
            c = '_';
        }
    }
    return oss_s;
}

void save_instance(const std::vector<double>& lower_bound, const std::vector<int>& raw_route,
                   const std::vector<double>& energy_prices, double soc_init, double consumption_cost,
                   const std::vector<std::tuple<int, double, double, double, double>>& raw_vertices,
                   std::vector<std::tuple<int, int, double, double, double, int, std::vector<double>, int>> raw_arcs,
                   std::tuple<double,double> soc_bounds){

    // Print the current working directory
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    // Extract folder naming based on current time
    std::string instance_name = get_time();

    // Subdirectory name
    std::string subdirectory = "./network_cpp/instances/"+instance_name;

    //std::cout << "Save instance to " << subdirectory << "\n";

    //std::cout << "Debug: ";
    //for (const auto& element : raw_route) std::cout << element << " ";
    //std::cout << "\n";

    // Create the subdirectory if it doesn't exist
    if (!std::filesystem::exists(subdirectory)) {
        std::filesystem::create_directories(subdirectory);
    }

    std::string filename = subdirectory + "/config.txt";
    std::ofstream configFile(filename);
    configFile << "soc_init " << soc_init << "\n";
    configFile << "consumption_cost " << consumption_cost << "\n";
    configFile << "soc_min " << std::get<0>(soc_bounds) << "\n";
    configFile << "soc_max " << std::get<1>(soc_bounds) << "\n";
    configFile << "energy_prices ";
    for (const auto value : energy_prices) {
        configFile << value << " ";
    }
    configFile << "\n" << "lower_bound ";
    for (const auto value : lower_bound) {
        configFile << value << " ";
    }

    configFile.close();

    // Save vertices
    filename = subdirectory + "/vertices.txt";
    std::ofstream verticesFile(filename);
    for (const auto& vertex : raw_vertices) {
        verticesFile << std::get<0>(vertex) << " "
                     << std::get<1>(vertex) << " "
                     << std::get<2>(vertex) << " "
                     << std::get<3>(vertex) << " "
                     << std::get<4>(vertex) << std::endl; // Add newline after each tuple
    }
    verticesFile.close();

    // Save arcs
    filename = subdirectory + "/arcs.txt";
    std::ofstream arcsFile(filename);
    for (const auto& arc : raw_arcs) {
        arcsFile << std::get<0>(arc) << " "
                 << std::get<1>(arc) << " "
                 << std::get<2>(arc) << " "
                 << std::get<3>(arc) << " "
                 << std::get<4>(arc) << " "
                 << std::get<5>(arc) << " "
                 << std::get<6>(arc).size() << " ";  // Print the size of the vector in place of the vector itself

        // Now print the elements of the vector (std::get<5>(arc) is a vector)
        for (const double& value : std::get<6>(arc)) {
            arcsFile << value << " ";
        }

        arcsFile << std::get<7>(arc) << std::endl; // Add newline after each tuple
    }

    filename = subdirectory + "/route.txt";
    std::ofstream routeFile(filename);
    for (const auto value : raw_route) {
        routeFile << value << " ";
    }
    routeFile.close();

    std::cout << "Saved Instance with  " << raw_vertices.size() << " vertices " << " to " << subdirectory << "\n";
}



ResultType spprc(const std::vector<double>& lower_bound, const std::vector<int>& raw_route,
                 const std::vector<double>& energy_prices, const double soc_init, const double consumption_cost,
                 const std::vector<std::tuple<int, double, double, double, double>>& raw_vertices,
                 const std::vector<std::tuple<int, int, double, double, double, int, std::vector<double>, int>>& raw_arcs,
                 const std::tuple<double,double>& soc_bounds) {

    // for debugging
    //bool save_instance_for_debug = true;
    //if (save_instance_for_debug) {
    //    save_instance(lower_bound, raw_route, energy_prices, soc_init, consumption_cost, raw_vertices, raw_arcs, soc_bounds);
    //}

    // Record the start time
    //auto start = std::chrono::steady_clock::now();

    // declare vector of vertices and arcs and allocate memory address immediately
    std::vector<Vertex> vertices;
    std::vector<std::tuple<int, int, Arc>> arcs;

    // declare return type
    ResultType result;

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

    // Initialise network
    Network n = {
            vertices, arcs, energy_prices, consumption_cost, soc_bounds
    };

    Route r = {raw_route};



    // try to run instance
    auto raw_paths = n.A_star_intermediate(
            lower_bound, r, soc_init, energy_prices
    );

    if (raw_paths.has_value()) {
        std::tuple<LabelTree, LabelTree, int, LabelNode*> raw_paths_value = raw_paths.value();
        result = n.propagate_path(raw_paths_value, r);
        //save_instance(lower_bound, raw_route, energy_prices, soc_init, consumption_cost, raw_vertices, raw_arcs, soc_bounds);
    } else {
        //save_instance(lower_bound, raw_route, energy_prices, soc_init, consumption_cost, raw_vertices, raw_arcs, soc_bounds);
        result = std::nullopt;
    }

    // Record the end time and compute duration (for debugging)
    //auto end = std::chrono::steady_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    //if (duration > std::chrono::seconds(30)) {
    //    save_instance(lower_bound, raw_route, energy_prices, soc_init, consumption_cost, raw_vertices, raw_arcs, soc_bounds);
    //}
    return result;
}


std::vector<ResultType>
solve_partial_subproblem(InputType inputs) {

    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), inputs.size());
    std::vector<std::optional<std::tuple<std::vector<std::tuple<int, double, double, double, double, double, int>>, double>>> results(inputs.size());

    std::atomic<bool> terminate_flag(false);
    std::mutex results_mutex;

    size_t total_tasks = inputs.size();
    size_t batch_size = num_threads;  // Each batch consists of `num_threads` tasks
    size_t batch_start = 0;

    while (batch_start < total_tasks) {
        size_t batch_end = std::min(batch_start + batch_size, total_tasks);
        std::vector<std::future<void>> futures;  // Store async task results

        // Launch batch of threads
        for (size_t i = batch_start; i < batch_end; ++i) {
            futures.push_back(std::async(std::launch::async, [&, i]() {
                if (terminate_flag.load(std::memory_order_relaxed)) {
                    return;
                }

            auto result = std::apply(spprc,inputs[i]);
            if (!result) {
                terminate_flag.store(true, std::memory_order_relaxed);
                return;
            }

            // Store result safely
            std::lock_guard<std::mutex> lock(results_mutex);
            results[i] = *result;
        }));
    }

    // Wait for all futures in this batch to complete
    for (auto& f : futures) {
        f.get();
    }

    batch_start = batch_end;
}

return results;
}



PYBIND11_MODULE(network_cpp, m) {
    m.doc() = R"pbdoc(
        Expose Label Setting Algorithm via Pybind11
        -----------------------

        .. currentmodule:: network_cpp

        .. autosummary::
           :toctree: _generate

           spprc

    )pbdoc";

    m.def("spprc", &spprc, py::return_value_policy::move,
          R"pbdoc(
             Solve decomposed subproblem (RCSSP) for a single vehicle route k in K.
          )pbdoc"
          );


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
