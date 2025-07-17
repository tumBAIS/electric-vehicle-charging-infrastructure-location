cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(network_cpp)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found python binary: ${Python_EXECUTABLE} (Version ${Python_VERSION})")

# Set Python variables for pybind11
set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_LIBS ${Python_LIBS})

# Include pybind11
add_subdirectory(lib/pybind11)

# Add project_options as an interface library
add_library(project_options INTERFACE)
target_include_directories(project_options INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
target_compile_features(project_options INTERFACE cxx_std_17)

# Compiler optimization for release mode
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

# Define targets for different build configurations
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create Python module targets using pybind11
pybind11_add_module(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PRIVATE project_options)

pybind11_add_module(${PROJECT_NAME}_catch2 STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME}_catch2 PRIVATE include)
target_link_libraries(${PROJECT_NAME}_catch2 PRIVATE project_options)

# Compiler options for targets
if (MSVC)
    # Set compile options for MSVC
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:/Od>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RelWithDebInfo>:/O2>)
    target_compile_options(${PROJECT_NAME}_catch2 PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(${PROJECT_NAME}_catch2 PRIVATE $<$<CONFIG:Debug>:/Od>)
    target_compile_options(${PROJECT_NAME}_catch2 PRIVATE $<$<CONFIG:RelWithDebInfo>:/O2>)
else ()
    # Set compile options for non-MSVC
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:-O3 -flto -frename-registers -march=native -DNDEBUG -s>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:-O0 -ggdb -march=native>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3 -flto -frename-registers -ggdb -march=native>)
    target_compile_options(${PROJECT_NAME}_catch2 PRIVATE $<$<CONFIG:Release>:-O3 -flto -frename-registers -march=native -DNDEBUG -s>)
    target_compile_options(${PROJECT_NAME}_catch2 PRIVATE $<$<CONFIG:Debug>:-O0 -ggdb -march=native>)
    target_compile_options(${PROJECT_NAME}_catch2 PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3 -flto -frename-registers -ggdb -march=native>)

    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-flto")
    set_target_properties(${PROJECT_NAME}_catch2 PROPERTIES LINK_FLAGS "-flto")
endif ()

# Add Catch2 tests subdirectory
add_subdirectory(catch2_tests)
