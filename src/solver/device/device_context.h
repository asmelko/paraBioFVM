#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>

#include "microenvironment.h"

namespace biofvm {
namespace solvers {
namespace device {

struct device_context
{
	cl::Context context;
	cl::CommandQueue queue;

	cl::Buffer diffusion_substrates;

	device_context();

	void initialize(microenvironment& m);

	void copy_to_device(microenvironment& m);
	void copy_to_host(microenvironment& m);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
