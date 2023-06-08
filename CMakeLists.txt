cmake_minimum_required(VERSION 3.18)

project(paraBioFVM VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fetch Googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch noarr
FetchContent_Declare(
  noarr_structures
  GIT_REPOSITORY https://github.com/ParaCoToUl/noarr-structures.git
  GIT_TAG        HEAD
)
FetchContent_Populate(noarr_structures)

# Target BioFVMCore
file(GLOB_RECURSE src_files "src/*")
list(FILTER src_files EXCLUDE REGEX ".*main\\..*")

add_library(BioFVMCore ${src_files})

target_include_directories(BioFVMCore PUBLIC ${noarr_structures_SOURCE_DIR}/include include/BioFVM)

# Target unit_BioFVM
file(GLOB_RECURSE test_files "test/*")
add_executable(unit_BioFVM ${test_files})

# Target BioFVM
add_executable(BioFVM src/main.cpp)
target_link_libraries(BioFVM BioFVMCore)

target_link_libraries(unit_BioFVM PRIVATE
	gtest_main gmock_main BioFVMCore)

target_include_directories(unit_BioFVM PUBLIC
	"src")

foreach(target BioFVM unit_BioFVM BioFVMCore)
	if(MSVC)
		target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /bigobj>)
	else()
		target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
	endif()
endforeach()

enable_testing()
include(GoogleTest)
gtest_discover_tests(unit_BioFVM)
