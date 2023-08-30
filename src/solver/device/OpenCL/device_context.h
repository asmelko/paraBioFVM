#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.hpp>

#include "microenvironment.h"
#include "solver/common_solver.h"

namespace biofvm {
namespace solvers {
namespace device {

struct device_context : common_solver
{
private:
	void resize(std::size_t new_capacity, microenvironment& m);

public:
	cl::Context context;
	cl::CommandQueue substrates_queue;
	cl::CommandQueue cell_data_queue;

	cl::Buffer diffusion_substrates;

	// required agent data:
	std::size_t capacity;

	cl::Buffer secretion_rates;
	cl::Buffer saturation_densities;
	cl::Buffer uptake_rates;
	cl::Buffer net_export_rates;

	cl::Buffer internalized_substrates;

	cl::Buffer volumes;
	cl::Buffer positions;

	device_context(bool print_device_info = true);

	void initialize(microenvironment& m);

	void copy_to_device(microenvironment& m);
	void copy_to_host(microenvironment& m);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
