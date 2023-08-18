#pragma once

#include "bulk_solver.h"
#include "cell_solver.h"
#include "diffusion_solver.h"
#include "gradient_solver.h"

namespace biofvm {
namespace solvers {
namespace host {

class solver
{
public:
	static constexpr bool is_device_solver = false;

	bulk_solver bulk;
	cell_solver cell;
	diffusion_solver diffusion;
	gradient_solver gradient;

	void initialize(microenvironment& m);

	void store_data_to_solver(microenvironment& m);
	void load_data_from_solver(microenvironment& m);
};

} // namespace host
} // namespace solvers
} // namespace biofvm
