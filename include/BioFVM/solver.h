#pragma once

#define impl host

#if impl == host
#	include "../../src/solver/host/bulk_solver.h"
#	include "../../src/solver/host/cell_solver.h"
#	include "../../src/solver/host/diffusion_solver.h"
#	include "../../src/solver/host/dirichlet_solver.h"
#	include "../../src/solver/host/gradient_solver.h"
#endif

namespace biofvm {

class solver
{
public:
	bulk_solver bulk;
	cell_solver cell;
	diffusion_solver diffusion;
	dirichlet_solver dirichlet;
	gradient_solver gradient;

	void initialize(microenvironment& m);
};

} // namespace biofvm
