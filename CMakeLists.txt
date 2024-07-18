cmake_minimum_required(VERSION 3.22)

project(
  paraBioFVM
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_FOR_DEVICE "Build opencl device code" OFF)
option(DOUBLE_PRECISION "Use double instead of float" OFF)

if(MSVC)
  set(BIOFVM_CPP_COMPILE_OPTIONS /W4 /bigobj)
else()
  set(BIOFVM_CPP_COMPILE_OPTIONS -Wall -Wextra -pedantic)
endif()

set(DEVICE_IMPL
    "opencl"
    CACHE STRING "Which device implementation to choose.")

set_property(CACHE DEVICE_IMPL PROPERTY STRINGS opencl cuda)

include(FetchContent)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND BUILD_TESTS)
  # Fetch Googletest
  FetchContent_Declare(
    googletest URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/googletest
                   DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Fetch noarr
FetchContent_Declare(
  noarr_structures
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/noarr-structures
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(noarr_structures)

# Target BioFVMCore
file(GLOB src_files "src/*")
list(FILTER src_files EXCLUDE REGEX ".*main.*")
file(GLOB common_solver_files "src/solver/*")
file(GLOB host_solver_files "src/solver/host/*")

add_library(BioFVMCore ${src_files} ${common_solver_files} ${host_solver_files})
target_compile_options(BioFVMCore PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${BIOFVM_CPP_COMPILE_OPTIONS}>)

include_directories(include/BioFVM src)
target_include_directories(BioFVMCore
                           PUBLIC ${noarr_structures_SOURCE_DIR}/include)

if(DOUBLE_PRECISION)
  target_compile_definitions(BioFVMCore PUBLIC USE_DOUBLES)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(BioFVMCore PUBLIC OpenMP::OpenMP_CXX)
endif()

if(BUILD_FOR_DEVICE)
  if(DEVICE_IMPL STREQUAL "opencl")
    target_compile_definitions(BioFVMCore PUBLIC BioFVM_SOLVER_IMPL_OPENCL)
    add_subdirectory(src/solver/device/OpenCL)
  else()
    enable_language(CUDA)
    target_compile_definitions(BioFVMCore PUBLIC BioFVM_SOLVER_IMPL_CUDA)
    add_subdirectory(src/solver/device/CUDA)
  endif()

  target_compile_options(device_solver PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${BIOFVM_CPP_COMPILE_OPTIONS}>)
  target_link_libraries(BioFVMCore PUBLIC device_solver)

  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    # Target BioFVM_device
    add_executable(BioFVM_device src/main_device.cpp)
    target_link_libraries(BioFVM_device BioFVMCore)
  endif()
else()
  target_compile_definitions(BioFVMCore PUBLIC BioFVM_SOLVER_IMPL_HOST)
endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  if(BUILD_TESTS)
    # Target unit_BioFVM
    file(GLOB_RECURSE test_files "test/*")
    if(NOT BUILD_FOR_DEVICE)
      list(FILTER test_files EXCLUDE REGEX ".*test_device.cpp")
    endif()
    add_executable(unit_BioFVM ${test_files})

    target_link_libraries(unit_BioFVM PRIVATE gtest_main gmock_main BioFVMCore)

    target_include_directories(unit_BioFVM PUBLIC "src")

    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(unit_BioFVM)
  endif()

  # Target BioFVM
  add_executable(BioFVM src/main.cpp)
  target_link_libraries(BioFVM BioFVMCore)
endif()
