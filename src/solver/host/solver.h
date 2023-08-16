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
	bulk_solver bulk;
	cell_solver cell;
	diffusion_solver diffusion;
	gradient_solver gradient;

	void initialize(microenvironment& m);
};

} // namespace host
} // namespace solvers
} // namespace biofvm
