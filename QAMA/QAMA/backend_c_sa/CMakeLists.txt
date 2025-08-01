cmake_minimum_required(VERSION 3.18)
project(simulated_annealing_qubo)

set(CMAKE_CXX_STANDARD 17)

find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 CONFIG REQUIRED)

# set(Torch_DIR "$ENV{LIBTORCH_HOME}/share/cmake/Torch")
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE Torch_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Torch_DIR "${Torch_DIR}/Torch")
message(STATUS "LibTorch directory: ${Torch_DIR}")
find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

pybind11_add_module(simulated_annealing_qubo simulated_annealing_qubo.cpp)
target_link_libraries(simulated_annealing_qubo PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Python)

set_property(TARGET simulated_annealing_qubo PROPERTY CXX_STANDARD 17)
set_target_properties(simulated_annealing_qubo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    PREFIX ""
    SUFFIX ".so"
)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type.")
