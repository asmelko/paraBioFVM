#include "dirichlet_solver.h"

#include "microenvironment.h"

using namespace biofvm;
using namespace solvers::device;

dirichlet_solver::dirichlet_solver(device_context& context)
	: opencl_solver(context, "dirichlet_solver.cl"),
	  solve_boundary_2d_x_(this->program_, "solve_boundary_2d_x"),
	  solve_boundary_2d_y_(this->program_, "solve_boundary_2d_y"),
	  solve_interior_2d_(this->program_, "solve_interior_2d"),
	  solve_boundary_3d_x_(this->program_, "solve_boundary_3d_x"),
	  solve_boundary_3d_y_(this->program_, "solve_boundary_3d_y"),
	  solve_boundary_3d_z_(this->program_, "solve_boundary_3d_z"),
	  solve_interior_3d_(this->program_, "solve_interior_3d")
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

	dirichlet_interior_voxels_count = m.dirichlet_interior_voxels_count;

	if (dirichlet_interior_voxels_count)
	{
		dirichlet_interior_voxels =
			cl::Buffer(ctx_.context, m.dirichlet_interior_voxels.get(),
					   m.dirichlet_interior_voxels.get() + dirichlet_interior_voxels_count, true);
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
	if (m.dirichlet_min_boundary_values[0])
	{
		solve_boundary_2d_x_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[1] * m.substrates_count)),
							 ctx_.diffusion_substrates, dirichlet_min_boundary_values_[0],
							 dirichlet_min_boundary_conditions_[0], m.substrates_count, m.mesh.grid_shape[0],
							 m.mesh.grid_shape[1], 0);
	}

	if (m.dirichlet_max_boundary_values[0])
	{
		solve_boundary_2d_x_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[1] * m.substrates_count)),
							 ctx_.diffusion_substrates, dirichlet_max_boundary_values_[0],
							 dirichlet_max_boundary_conditions_[0], m.substrates_count, m.mesh.grid_shape[0],
							 m.mesh.grid_shape[1], m.mesh.grid_shape[0] - 1);
	}

	if (m.dirichlet_min_boundary_values[1])
	{
		solve_boundary_2d_y_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.substrates_count)),
							 ctx_.diffusion_substrates, dirichlet_min_boundary_values_[1],
							 dirichlet_min_boundary_conditions_[1], m.substrates_count, m.mesh.grid_shape[0],
							 m.mesh.grid_shape[1], 0);
	}

	if (m.dirichlet_max_boundary_values[1])
	{
		solve_boundary_2d_y_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.substrates_count)),
							 ctx_.diffusion_substrates, dirichlet_max_boundary_values_[1],
							 dirichlet_max_boundary_conditions_[1], m.substrates_count, m.mesh.grid_shape[0],
							 m.mesh.grid_shape[1], m.mesh.grid_shape[1] - 1);
	}

	if (dirichlet_interior_voxels_count)
		solve_interior_2d_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(dirichlet_interior_voxels_count * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_interior_voxels, dirichlet_interior_values,
			dirichlet_interior_conditions, m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1]);
}

void dirichlet_solver::solve_3d(microenvironment& m)
{
	if (m.dirichlet_min_boundary_values[0])
	{
		solve_boundary_3d_x_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[1] * m.mesh.grid_shape[2] * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_min_boundary_values_[0], dirichlet_min_boundary_conditions_[0],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2], 0);
	}

	if (m.dirichlet_max_boundary_values[0])
	{
		solve_boundary_3d_x_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[1] * m.mesh.grid_shape[2] * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_max_boundary_values_[0], dirichlet_max_boundary_conditions_[0],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2],
			m.mesh.grid_shape[0] - 1);
	}

	if (m.dirichlet_min_boundary_values[1])
	{
		solve_boundary_3d_y_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.mesh.grid_shape[2] * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_min_boundary_values_[1], dirichlet_min_boundary_conditions_[1],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2], 0);
	}

	if (m.dirichlet_max_boundary_values[1])
	{
		solve_boundary_3d_y_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.mesh.grid_shape[2] * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_max_boundary_values_[1], dirichlet_max_boundary_conditions_[1],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2],
			m.mesh.grid_shape[1] - 1);
	}

	if (m.dirichlet_min_boundary_values[2])
	{
		solve_boundary_3d_z_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.mesh.grid_shape[1] * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_min_boundary_values_[2], dirichlet_min_boundary_conditions_[2],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2], 0);
	}

	if (m.dirichlet_max_boundary_values[2])
	{
		solve_boundary_3d_z_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.mesh.grid_shape[1] * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_max_boundary_values_[2], dirichlet_max_boundary_conditions_[2],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2],
			m.mesh.grid_shape[2] - 1);
	}

	if (dirichlet_interior_voxels_count)
		solve_interior_3d_(
			cl::EnqueueArgs(ctx_.queue, cl::NDRange(dirichlet_interior_voxels_count * m.substrates_count)),
			ctx_.diffusion_substrates, dirichlet_interior_voxels, dirichlet_interior_values,
			dirichlet_interior_conditions, m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1],
			m.mesh.grid_shape[2]);
}
