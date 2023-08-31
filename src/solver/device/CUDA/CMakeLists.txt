cmake_minimum_required(VERSION 3.24)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB device_solver_files "*")

add_library(CUDA_device_solver ${device_solver_files})

target_include_directories(CUDA_device_solver PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(CUDA_device_solver PROPERTIES CUDA_ARCHITECTURES "86")

set(DEVICE_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
