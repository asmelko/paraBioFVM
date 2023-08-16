#include "dirichlet_solver.h"

using namespace biofvm;
using namespace solvers::device;

dirichlet_solver::dirichlet_solver(device_context& context)
	: opencl_solver(context, "dirichlet_solver.cl"), solve_boundary_2d_(this->program_, "solve_boundary_2d")
{}

void dirichlet_solver::initialize(microenvironment& m)
{
	for (index_t i = 0; i < m.mesh.dims; i++)
	{
		if (m.dirichlet_min_boundary_values[i])
		{
			dirichlet_min_boundary_values_[i] =
				cl::Buffer(ctx_.context, m.dirichlet_min_boundary_values[i].get(),
						   m.dirichlet_min_boundary_values[i].get() + m.substrates_count, true);

			dirichlet_min_boundary_conditions_[i] =
				cl::Buffer(ctx_.context, m.dirichlet_min_boundary_conditions[i].get(),
						   m.dirichlet_min_boundary_conditions[i].get() + m.substrates_count, true);
		}

		if (m.dirichlet_max_boundary_values[i])
		{
			dirichlet_max_boundary_conditions_[i] =
				cl::Buffer(ctx_.context, m.dirichlet_max_boundary_conditions[i].get(),
						   m.dirichlet_max_boundary_conditions[i].get() + m.substrates_count, true);

			dirichlet_max_boundary_values_[i] =
				cl::Buffer(ctx_.context, m.dirichlet_max_boundary_values[i].get(),
						   m.dirichlet_max_boundary_values[i].get() + m.substrates_count, true);
		}
	}
}

void dirichlet_solver::solve_2d(microenvironment& m)
{
	if (m.dirichlet_min_boundary_values[0])
	{
		solve_boundary_2d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0])), ctx_.diffusion_substrates,
						   dirichlet_min_boundary_values_[0], dirichlet_min_boundary_conditions_[0], m.substrates_count,
						   0, 1);
	}

	if (m.dirichlet_max_boundary_values[0])
	{
		solve_boundary_2d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0])), ctx_.diffusion_substrates,
						   dirichlet_max_boundary_values_[0], dirichlet_max_boundary_conditions_[0], m.substrates_count,
						   0, 1);
	}
}
