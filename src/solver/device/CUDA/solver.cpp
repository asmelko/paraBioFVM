#include "solver.h"

using namespace biofvm::solvers::device;

solver::solver() : cell(ctx), diffusion(ctx) {}

void solver::initialize(microenvironment& m)
{
	ctx.initialize(m);

	cell.initialize(m);
	diffusion.initialize(m);
}

void solver::store_data_to_solver(microenvironment& m) { ctx.copy_to_device(m); }

void solver::load_data_from_solver(microenvironment& m) { ctx.copy_to_host(m); }

void solver::wait_for_all()
{
	CUCH(cudaStreamSynchronize(ctx.substrates_queue));
	CUCH(cudaStreamSynchronize(ctx.cell_data_queue));
}
