#include "solver.h"

void solver::initialize(microenvironment& m)
{
	bulk.initialize(m);
	cell.initialize(m);
	diffusion.initialize(m);
	dirichlet.initialize(m);
	gradient.initialize(m);
}
