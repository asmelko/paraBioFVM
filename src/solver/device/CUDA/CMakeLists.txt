cmake_minimum_required(VERSION 3.24)

enable_language(CUDA)

# Find CC of installed GPUs
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
message (STATUS "CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB device_solver_files "*")

add_library(device_solver OBJECT ${device_solver_files})

target_include_directories(
  device_solver PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                       ${noarr_structures_SOURCE_DIR}/include)

# target_compile_options(device_solver PRIVATE
# $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
