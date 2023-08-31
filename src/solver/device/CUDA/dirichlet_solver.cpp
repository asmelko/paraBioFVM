#include "dirichlet_solver.h"

#include "microenvironment.h"

using namespace biofvm;
using namespace solvers::device;

void run_solve_interior_2d(real_t* substrate_densities, const index_t* dirichlet_voxels, const real_t* dirichlet_values,
						   const bool* dirichlet_conditions, index_t n, index_t substrates_count, index_t x_size,
						   index_t y_size, cudaStream_t& stream);
void run_solve_interior_3d(real_t* substrate_densities, const index_t* dirichlet_voxels, const real_t* dirichlet_values,
						   const bool* dirichlet_conditions, index_t n, index_t substrates_count, index_t x_size,
						   index_t y_size, index_t z_size, cudaStream_t& stream);

dirichlet_solver::dirichlet_solver(device_context& context) : cuda_solver(context) {}

void dirichlet_solver::initialize(microenvironment& m)
{
	std::unique_ptr<bool[]> conditions_default = std::make_unique<bool[]>(m.substrates_count);
	for (index_t i = 0; i < m.substrates_count; i++)
		conditions_default[i] = false;

	for (index_t i = 0; i < m.mesh.dims; i++)
	{
		if (m.dirichlet_min_boundary_values[i])
		{
			CUCH(cudaMalloc(&dirichlet_min_boundary_values[i], m.substrates_count * sizeof(real_t)));

			CUCH(cudaMemcpy(dirichlet_min_boundary_values[i], m.dirichlet_min_boundary_values[i].get(),
							m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));

			CUCH(cudaMalloc(&dirichlet_min_boundary_conditions[i], m.substrates_count * sizeof(bool)));

			CUCH(cudaMemcpy(dirichlet_min_boundary_conditions[i], m.dirichlet_min_boundary_conditions[i].get(),
							m.substrates_count * sizeof(bool), cudaMemcpyHostToDevice));
		}
		else
		{
			CUCH(cudaMalloc(&dirichlet_min_boundary_conditions[i], m.substrates_count * sizeof(bool)));

			CUCH(cudaMemcpy(dirichlet_min_boundary_conditions[i], conditions_default.get(),
							m.substrates_count * sizeof(bool), cudaMemcpyHostToDevice));

			dirichlet_min_boundary_values[i] = nullptr;
		}

		if (m.dirichlet_max_boundary_values[i])
		{
			CUCH(cudaMalloc(&dirichlet_max_boundary_values[i], m.substrates_count * sizeof(real_t)));

			CUCH(cudaMemcpy(dirichlet_max_boundary_values[i], m.dirichlet_max_boundary_values[i].get(),
							m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));

			CUCH(cudaMalloc(&dirichlet_max_boundary_conditions[i], m.substrates_count * sizeof(bool)));

			CUCH(cudaMemcpy(dirichlet_max_boundary_conditions[i], m.dirichlet_max_boundary_conditions[i].get(),
							m.substrates_count * sizeof(bool), cudaMemcpyHostToDevice));
		}
		else
		{
			CUCH(cudaMalloc(&dirichlet_max_boundary_conditions[i], m.substrates_count * sizeof(bool)));

			CUCH(cudaMemcpy(dirichlet_max_boundary_conditions[i], conditions_default.get(),
							m.substrates_count * sizeof(bool), cudaMemcpyHostToDevice));

			dirichlet_max_boundary_values[i] = nullptr;
		}
	}

	dirichlet_interior_voxels_count = m.dirichlet_interior_voxels_count;

	if (dirichlet_interior_voxels_count)
	{
		CUCH(cudaMalloc(&dirichlet_interior_voxels, dirichlet_interior_voxels_count * m.mesh.dims * sizeof(index_t)));
		CUCH(cudaMemcpy(dirichlet_interior_voxels, m.dirichlet_interior_voxels.get(),
						dirichlet_interior_voxels_count * m.mesh.dims * sizeof(index_t), cudaMemcpyHostToDevice));

		CUCH(cudaMalloc(&dirichlet_interior_conditions,
						dirichlet_interior_voxels_count * m.substrates_count * sizeof(bool)));
		CUCH(cudaMemcpy(dirichlet_interior_conditions, m.dirichlet_interior_conditions.get(),
						dirichlet_interior_voxels_count * m.substrates_count * sizeof(bool), cudaMemcpyHostToDevice));

		CUCH(cudaMalloc(&dirichlet_interior_values,
						dirichlet_interior_voxels_count * m.substrates_count * sizeof(real_t)));
		CUCH(cudaMemcpy(dirichlet_interior_values, m.dirichlet_interior_values.get(),
						dirichlet_interior_voxels_count * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
	}
}

void dirichlet_solver::solve_2d(microenvironment& m)
{
	if (dirichlet_interior_voxels_count)
		run_solve_interior_2d(ctx_.diffusion_substrates, dirichlet_interior_voxels, dirichlet_interior_values,
							  dirichlet_interior_conditions, dirichlet_interior_voxels_count, m.substrates_count,
							  m.mesh.grid_shape[0], m.mesh.grid_shape[1], ctx_.substrates_queue);
}

void dirichlet_solver::solve_3d(microenvironment& m)
{
	if (dirichlet_interior_voxels_count)
		run_solve_interior_3d(ctx_.diffusion_substrates, dirichlet_interior_voxels, dirichlet_interior_values,
							  dirichlet_interior_conditions, dirichlet_interior_voxels_count, m.substrates_count,
							  m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2], ctx_.substrates_queue);
}

void dirichlet_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}
