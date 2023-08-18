#include "solver.h"

using namespace biofvm::solvers::host;

void solver::initialize(microenvironment& m)
{
	bulk.initialize(m);
	cell.initialize(m);
	diffusion.initialize(m);
}

void solver::store_data_to_solver(microenvironment&) {}

void solver::load_data_from_solver(microenvironment&) {}
