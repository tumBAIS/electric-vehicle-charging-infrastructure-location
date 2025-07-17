# MILAS: Charging station location for stationary and dynamic charging of electric shuttles in rural areas

## Getting started (for Windows)
Create a virtual environment with Python 3.10+ and activate it

```bash
python -m venv venv
./venv/Scripts/activate
```


Install the required packages

```bash
pip install -r requirements.txt
```

## C++ with Pybind11
1. Download and install CMake (https://cmake.org/download/). Make sure that CMake is added to the path variables.  

2. Download and install Visual Studio Build Tools (https://visualstudio.microsoft.com/de/downloads/).

3. Installing Pybind11 into the project
- Navigate into network_cpp/lib
- Clone pybind11 Repo (https://github.com/pybind/pybind11) into this location 
- Install pybind11 as a python package 

```bash
cd ./network_cpp/lib
git clone https://github.com/pybind/pybind11
cd pybind
python setup.py install
```

4. [Optional] Download and install catch2 (https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md)

5. Build the project with CMake in release mode

```bash
cd ./network_cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE="RELEASE" -DPython_ROOT_DIR="<path/to/venv>" 
cmake --build .
```

Note: You may have to add the index to the archive of ```./network_cpp_catch2.cpython-310-x86_64-linux-gnu.so``` by running ranlib on this file: 

```bash
ranlib network_cpp_catch2.cpython-310-x86_64-linux-gnu.so
```

6. Install as python package

```bash
pip install wheel
pip install ./network_cpp
```

## Testing
We use pytest for testing. All unit tests can be found under ```./tests/*``` and the full test suite incl. coverage report is started via

```bash
cd ./tests
pytest --cov=charging-station-location
```

## Running experiments
Running experiments requires a problem instances parsed as ```IntermediateRepresentation```. 
Instead of collecting customized data and parsing it into an instance of ```IntermediateRepresentation```, it is straightforward to use our Parser(s). We differentiate between artifical benchmark instances (to evaluate the computational performance of our algorithmic framework and provide a study for an alternative use case in logistics), and instances based on OSM street networks:

To run the artificial EVRPTW instances (Based on well-known "Solomon Instances"), run

```bash
python3 run_solomon.py <name> <solver> <speed-adaption>
```

where the parameters define the name of the instance (e.g., "C101"), the solver (i.e., "ILS", or "cplex"), and speed adaption scales the vehicle velocity (i.e., "1.0" uses the original velocity)

To run the instaces based on the OSM street network, run

```bash
python3 run_case_study.py "hof_scenario_0" <price-curve> <consumption> <stat-cost> <dyn-cost> <var-cost> <allow-deviations>
```

where the parameters define the price-curve (e.g., "base" for a constant base price), the consumption [kWh/km], the fixed cost for stationary charging stations and dynamic charging stations [EUR], the variable cost per meter segment [EUR/m], and a flag indicating if vehicles are allowed to deviate from their shortest path between consecutive stops ["True" / "False"]

