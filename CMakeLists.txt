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

add_library(BioFVMCore ${src_files} ${common_solver_files} ${host_solver_files})

include_directories(include/BioFVM src)
target_include_directories(BioFVMCore PUBLIC ${noarr_structures_SOURCE_DIR}/include)

if (DOUBLE_PRECISION)
	target_compile_definitions(BioFVMCore PUBLIC USE_DOUBLES)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(BioFVMCore PUBLIC OpenMP::OpenMP_CXX)
endif()

if (BUILD_FOR_DEVICE)
	target_compile_definitions(BioFVMCore PUBLIC BioFVM_SOLVER_IMPL=1)

	add_subdirectory(src/solver/device/OpenCL)

	target_link_libraries(BioFVMCore PUBLIC OpenCL_device_solver)

	if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
		# Target BioFVM_device
		add_executable(BioFVM_device src/main_device.cpp)
		target_link_libraries(BioFVM_device BioFVMCore)
	endif()
endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	# Target unit_BioFVM
	file(GLOB_RECURSE test_files "test/*")	
	if (NOT BUILD_FOR_DEVICE)
		list(FILTER test_files EXCLUDE REGEX ".*test_device.cpp")
	endif()
	add_executable(unit_BioFVM ${test_files})

	target_link_libraries(unit_BioFVM PRIVATE
		gtest_main gmock_main BioFVMCore)

	target_include_directories(unit_BioFVM PUBLIC
		"src")

	enable_testing()
	include(GoogleTest)
	gtest_discover_tests(unit_BioFVM)

	# Target BioFVM
	add_executable(BioFVM src/main.cpp)
	target_link_libraries(BioFVM BioFVMCore)
endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	set(all_targets BioFVM unit_BioFVM BioFVMCore)
else()
	set(all_targets BioFVMCore)
endif()

if(MSVC)
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4 /bigobj>)
else()
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
endif()
