file(GLOB_RECURSE CATCH_SOURCES "${PROJECT_SOURCE_DIR}/catch2_tests/*.cpp")
add_executable(catch2_tests ${CATCH_SOURCES})

# Ensure Python development files are included
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
link_directories(${Python_LIBRARY_DIRS})

target_include_directories(catch2_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(catch2_tests PRIVATE ${PROJECT_SOURCE_DIR}/lib/catch2/include)
target_include_directories(catch2_tests PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(catch2_tests PRIVATE ${PROJECT_SOURCE_DIR}/lib/pybind11/include)

target_link_libraries(catch2_tests PRIVATE network_cpp_catch2)
target_link_libraries(catch2_tests PRIVATE project_options)
target_link_libraries(catch2_tests PRIVATE ${Python_LIBRARIES})

