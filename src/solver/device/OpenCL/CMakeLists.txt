cmake_minimum_required(VERSION 3.18)

file(GLOB device_solver_files "*")

add_library(device_solver OBJECT ${device_solver_files})

FetchContent_Declare(
  opencl_headers
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_SOURCE_DIR}/submodules/OpenCL-Headers)
FetchContent_MakeAvailable(opencl_headers)

if(${APPLE})
  file(CREATE_LINK ${opencl_headers_SOURCE_DIR}/CL
       ${opencl_headers_SOURCE_DIR}/OpenCL SYMBOLIC)
endif()

FetchContent_Declare(
  opencl_icd
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_SOURCE_DIR}/submodules/OpenCL-ICD-Loader)
FetchContent_MakeAvailable(opencl_icd)

FetchContent_Declare(
  opencl_hpp
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_SOURCE_DIR}/submodules/OpenCL-CLHPP)
FetchContent_MakeAvailable(opencl_hpp)

target_include_directories(device_solver
                           PUBLIC ${opencl_hpp_SOURCE_DIR}/include)

target_link_libraries(device_solver PUBLIC OpenCL::OpenCL)

target_compile_definitions(
  device_solver PUBLIC BioFVM_OPENCL_SOURCES="${CMAKE_CURRENT_SOURCE_DIR}/")
