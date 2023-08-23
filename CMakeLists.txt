cmake_minimum_required(VERSION 3.18)

project(paraBioFVM VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_FOR_DEVICE "Build opencl device code" OFF)
option(DOUBLE_PRECISION "Use double instead of float" OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	# Fetch Googletest
	include(FetchContent)
	FetchContent_Declare(
	  googletest
	  GIT_REPOSITORY https://github.com/google/googletest.git
	  GIT_TAG v1.14.0
	)
	# For Windows: Prevent overriding the parent project's compiler/linker settings
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)
endif()

# Fetch noarr
FetchContent_Declare(
	noarr_structures
	GIT_REPOSITORY https://github.com/ParaCoToUl/noarr-structures.git
	GIT_TAG        c++17
)
FetchContent_Populate(noarr_structures)

# Target BioFVMCore
file(GLOB src_files "src/*")
list(FILTER src_files EXCLUDE REGEX ".*main.*")
file(GLOB common_solver_files "src/solver/*")
file(GLOB host_solver_files "src/solver/host/*")
file(GLOB device_solver_files "src/solver/device/*")

if (BUILD_FOR_DEVICE)
	add_library(BioFVMCore OBJECT ${src_files} ${common_solver_files} ${host_solver_files} ${device_solver_files})
else()
	add_library(BioFVMCore OBJECT ${src_files} ${common_solver_files} ${host_solver_files})
endif()

target_include_directories(BioFVMCore PUBLIC ${noarr_structures_SOURCE_DIR}/include include/BioFVM)

if (DOUBLE_PRECISION)
	target_compile_definitions(BioFVMCore PUBLIC USE_DOUBLES)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(BioFVMCore PUBLIC OpenMP::OpenMP_CXX)
endif()

if (BUILD_FOR_DEVICE)

	target_compile_definitions(BioFVMCore PUBLIC BioFVM_SOLVER_IMPL=1)

	FetchContent_Declare(
		opencl_hpp
		GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
		GIT_TAG        v2023.04.17
	)
	FetchContent_Populate(opencl_hpp)

	FetchContent_Declare(
		opencl_headers
		GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
		GIT_TAG        e049b16b5f157e2f28e7b5c301e71e1ccb3fe288
	)
	FetchContent_Populate(opencl_headers)

	set(OPENCL_ICD_LOADER_HEADERS_DIR ${opencl_headers_SOURCE_DIR} CACHE PATH "Path to OpenCL Headers")

	FetchContent_Declare(
		opencl_icd
		GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
		GIT_TAG        229410f86a8c8c9e0f86f195409e5481a2bae067
	)
	FetchContent_MakeAvailable(opencl_icd)

	target_include_directories(BioFVMCore PUBLIC ${opencl_hpp_SOURCE_DIR}/include)

	target_link_libraries(BioFVMCore PUBLIC OpenCL::OpenCL)

	target_compile_definitions(BioFVMCore PUBLIC BioFVM_OPENCL_SOURCES="${CMAKE_CURRENT_SOURCE_DIR}/src/solver/device/")

	if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
		# Target BioFVM_device
		add_executable(BioFVM_device src/main_device.cpp)
		target_link_libraries(BioFVM_device BioFVMCore)
	endif()

endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	
	# Target unit_BioFVM
	if (BUILD_FOR_DEVICE)
		file(GLOB_RECURSE test_files "test/*")	
		add_executable(unit_BioFVM ${test_files})
	else()
		file(GLOB_RECURSE test_files "test/*")
		list(FILTER test_files EXCLUDE REGEX ".*test_device.cpp")
		add_executable(unit_BioFVM ${test_files})
	endif()

	# Target BioFVM
	add_executable(BioFVM src/main.cpp)
	target_link_libraries(BioFVM BioFVMCore)

	target_link_libraries(unit_BioFVM PRIVATE
		gtest_main gmock_main BioFVMCore)

	target_include_directories(unit_BioFVM PUBLIC
		"src")

	enable_testing()
	include(GoogleTest)
	gtest_discover_tests(unit_BioFVM)
endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	set(all_targets BioFVM unit_BioFVM BioFVMCore)
else()
	set(all_targets BioFVMCore)
endif()

foreach(target ${all_targets})
	if(MSVC)
		target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /bigobj>)
	else()
		target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
	endif()
endforeach()
