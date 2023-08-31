#pragma once

#include "device_context.h"

namespace biofvm {
namespace solvers {
namespace device {

class cuda_solver
{
protected:
	device_context& ctx_;

	cuda_solver(device_context& ctx);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
