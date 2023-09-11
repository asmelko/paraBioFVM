cmake_minimum_required(VERSION 3.24)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB device_solver_files "*")

add_library(device_solver ${device_solver_files})

target_include_directories(device_solver PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${noarr_structures_SOURCE_DIR}/include)

set_target_properties(device_solver PROPERTIES CUDA_ARCHITECTURES native)

# target_compile_options(device_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
