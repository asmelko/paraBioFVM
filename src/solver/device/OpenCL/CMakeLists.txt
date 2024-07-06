cmake_minimum_required(VERSION 3.18)

file(GLOB device_solver_files "*")

add_library(device_solver OBJECT ${device_solver_files})

FetchContent_Declare(
  opencl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
  GIT_TAG main)
FetchContent_MakeAvailable(opencl_headers)

if(${APPLE})
  file(CREATE_LINK ${opencl_headers_SOURCE_DIR}/CL
       ${opencl_headers_SOURCE_DIR}/OpenCL SYMBOLIC)
endif()

FetchContent_Declare(
  opencl_icd
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
  GIT_TAG main)
FetchContent_MakeAvailable(opencl_icd)

FetchContent_Declare(
  opencl_hpp
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
  GIT_TAG main)
FetchContent_MakeAvailable(opencl_hpp)

target_include_directories(device_solver
                           PUBLIC ${opencl_hpp_SOURCE_DIR}/include)

target_link_libraries(device_solver PUBLIC OpenCL::OpenCL)

target_compile_definitions(
  device_solver PUBLIC BioFVM_OPENCL_SOURCES="${CMAKE_CURRENT_SOURCE_DIR}/")
