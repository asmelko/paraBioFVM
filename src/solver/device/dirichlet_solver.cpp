#include "dirichlet_solver.h"

#include "microenvironment.h"

#define ROUND_UP(a, b) ((((a) + (b)-1) / (b)) * (b))

using namespace biofvm;
using namespace solvers::device;

dirichlet_solver::dirichlet_solver(device_context& context)
	: opencl_solver(context, "dirichlet_solver.cl"),
	  solve_interior_2d_(this->program_, "solve_interior_2d"),
	  solve_interior_3d_(this->program_, "solve_interior_3d")
{}

void dirichlet_solver::initialize(microenvironment& m)
{
	std::unique_ptr<bool[]> conditions_default = std::make_unique<bool[]>(m.substrates_count);
	for (index_t i = 0; i < m.substrates_count; i++)
		conditions_default[i] = false;

	for (index_t i = 0; i < m.mesh.dims; i++)
	{
		if (m.dirichlet_min_boundary_values[i])
		{
			dirichlet_min_boundary_values[i] =
				cl::Buffer(ctx_.context, m.dirichlet_min_boundary_values[i].get(),
						   m.dirichlet_min_boundary_values[i].get() + m.substrates_count, true);

			dirichlet_min_boundary_conditions[i] =
				cl::Buffer(ctx_.context, m.dirichlet_min_boundary_conditions[i].get(),
						   m.dirichlet_min_boundary_conditions[i].get() + m.substrates_count, true);
		}
		else
		{
			dirichlet_min_boundary_conditions[i] =
				cl::Buffer(ctx_.context, conditions_default.get(), conditions_default.get() + m.substrates_count, true);
			dirichlet_min_boundary_values[i] = nullptr;
		}

		if (m.dirichlet_max_boundary_values[i])
		{
			dirichlet_max_boundary_conditions[i] =
				cl::Buffer(ctx_.context, m.dirichlet_max_boundary_conditions[i].get(),
						   m.dirichlet_max_boundary_conditions[i].get() + m.substrates_count, true);

			dirichlet_max_boundary_values[i] =
				cl::Buffer(ctx_.context, m.dirichlet_max_boundary_values[i].get(),
						   m.dirichlet_max_boundary_values[i].get() + m.substrates_count, true);
		}
		else
		{
			dirichlet_max_boundary_conditions[i] =
				cl::Buffer(ctx_.context, conditions_default.get(), conditions_default.get() + m.substrates_count, true);
			dirichlet_max_boundary_values[i] = nullptr;
		}
	}

	dirichlet_interior_voxels_count = m.dirichlet_interior_voxels_count;

	if (dirichlet_interior_voxels_count)
	{
		dirichlet_interior_voxels =
			cl::Buffer(ctx_.context, m.dirichlet_interior_voxels.get(),
					   m.dirichlet_interior_voxels.get() + dirichlet_interior_voxels_count * m.mesh.dims, true);
		dirichlet_interior_conditions = cl::Buffer(
			ctx_.context, m.dirichlet_interior_conditions.get(),
			m.dirichlet_interior_conditions.get() + dirichlet_interior_voxels_count * m.substrates_count, true);
		dirichlet_interior_values =
			cl::Buffer(ctx_.context, m.dirichlet_interior_values.get(),
					   m.dirichlet_interior_values.get() + dirichlet_interior_voxels_count * m.substrates_count, true);
	}
}

void dirichlet_solver::solve_2d(microenvironment& m)
{
	if (dirichlet_interior_voxels_count)
		solve_interior_2d_(
			cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(dirichlet_interior_voxels_count * m.substrates_count, 256)), cl::NDRange(256)),
			ctx_.diffusion_substrates, dirichlet_interior_voxels, dirichlet_interior_values,
			dirichlet_interior_conditions, m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], dirichlet_interior_voxels_count);
}

void dirichlet_solver::solve_3d(microenvironment& m)
{
	if (dirichlet_interior_voxels_count)
		solve_interior_3d_(
			cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(dirichlet_interior_voxels_count * m.substrates_count, 256)), cl::NDRange(256)),
			ctx_.diffusion_substrates, dirichlet_interior_voxels, dirichlet_interior_values,
			dirichlet_interior_conditions, m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1],
			m.mesh.grid_shape[2], dirichlet_interior_voxels_count);
}

void dirichlet_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}
