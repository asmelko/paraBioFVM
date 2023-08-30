#pragma once

#include "device_context.h"

namespace biofvm {
namespace solvers {
namespace device {

class opencl_solver
{
protected:
	device_context& ctx_;

	cl::Program program_;

	opencl_solver(device_context& ctx, const std::string& file_name);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
