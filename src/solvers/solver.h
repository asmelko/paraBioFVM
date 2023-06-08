#pragma once

#define impl host

#if impl == host
#	include "host/bulk/bulk_solver.h"
#	include "host/cell/cell_solver.h"
#	include "host/diffusion/diffusion_solver.h"
#	include "host/dirichlet/dirichlet_solver.h"
#	include "host/gradient/gradient_solver.h"
#endif

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
