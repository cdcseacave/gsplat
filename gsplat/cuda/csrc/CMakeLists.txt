# Needs g++ >= 11, which set C++17 as default standard.

cmake_minimum_required(VERSION 3.12)
project(gsplat LANGUAGES CXX CUDA)

set(PYTHON_EXECUTABLE "/home/paul/bin/anaconda3/envs/nerfstudio-gsplat2/bin/python")

# Set version
set(GSPLAT_VERSION "0.1")

# Ensure `-fPIC` is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define CUDA flags
#set(NVCC_FLAGS "-O3 --use_fast_math")
set(NVCC_FLAGS "-G -g -lineinfo")

execute_process(
        COMMAND python -c "import sys; print(sys.executable)"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _python_prefix_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("Result: ${_result}")
message("Python path: ${_python_prefix_path}")

# Get the Torch path from Python
execute_process(
    COMMAND python -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _python_prefix_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if the command was successful
if (_result EQUAL 0)
    # Set the CMAKE_PREFIX_PATH
    set(CMAKE_PREFIX_PATH "${_python_prefix_path}" CACHE PATH "Path to Torch")
else()
    message(FATAL_ERROR "Failed to get CMAKE_PREFIX_PATH from Python")
endif()

# Find packages
find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)

# Include directories
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third_party/glm)

include_directories(${TORCH_INCLUDE_DIRS})
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

# Set up CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

# Gather source files
file(GLOB_RECURSE SOURCES
    ${CMAKE_SOURCE_DIR}/*.cu
)

# Create the library
add_library(gsplat STATIC ${SOURCES})

target_include_directories(gsplat PUBLIC ${TORCH_INCLUDE_DIRS})

# Link libraries
target_link_libraries(gsplat ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

# Set additional properties
set_target_properties(gsplat PROPERTIES
    VERSION ${GSPLAT_VERSION}
    SOVERSION 1
)
target_link_libraries(gsplat ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

# Install targets
install(TARGETS gsplat
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

add_executable(test_splat test.cpp)
target_link_libraries(test_splat ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES} gsplat)
target_include_directories(test_splat PUBLIC ${TORCH_INCLUDE_DIRS})

message(STATUS "CMake configuration done!")

