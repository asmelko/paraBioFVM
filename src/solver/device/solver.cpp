#include "solver.h"

using namespace biofvm::solvers::device;

solver::solver(bool print_device_info) : ctx(print_device_info), cell(ctx), diffusion(ctx) {}

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
	ctx.substrates_queue.finish();
	ctx.cell_data_queue.finish();
}
