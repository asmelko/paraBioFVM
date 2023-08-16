#include "solver.h"

using namespace biofvm::solvers::host;

void solver::initialize(microenvironment& m)
{
	bulk.initialize(m);
	cell.initialize(m);
	diffusion.initialize(m);
}
