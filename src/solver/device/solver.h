#pragma once

#include "../host/gradient_solver.h"
#include "cell_solver.h"
#include "device_context.h"
#include "diffusion_solver.h"

namespace biofvm {
namespace solvers {
namespace device {

class solver
{
	device_context ctx;

public:
	solver(bool print_device_info = false);

	static constexpr bool is_device_solver = true;

	cell_solver cell;
	diffusion_solver diffusion;
	host::gradient_solver gradient;

	void initialize(microenvironment& m);

	void store_data_to_solver(microenvironment& m);
	void load_data_from_solver(microenvironment& m);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
