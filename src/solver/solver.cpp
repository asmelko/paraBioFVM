#include "solver.h"

using namespace biofvm;

void solver::initialize(microenvironment& m)
{
	bulk.initialize(m);
	cell.initialize(m);
	dirichlet.initialize(m);
	diffusion.initialize(m, dirichlet);
	gradient.initialize(m);
}
