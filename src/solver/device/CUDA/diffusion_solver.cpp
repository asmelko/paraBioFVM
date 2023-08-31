#include "diffusion_solver.h"

#include <stdexcept>

#include "microenvironment.h"
#include "solver/host/diffusion_solver.h"

using namespace biofvm;
using namespace solvers::device;

void run_solve_slice_2d_x(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  cudaStream_t& stream);
void run_solve_slice_2d_y(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  cudaStream_t& stream);

void run_solve_slice_3d_x(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  index_t z_size, cudaStream_t& stream);
void run_solve_slice_3d_y(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  index_t z_size, cudaStream_t& stream);
void run_solve_slice_3d_z(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  index_t z_size, cudaStream_t& stream);

diffusion_solver::diffusion_solver(device_context& ctx) : cuda_solver(ctx), dirichlet(ctx) {}

void diffusion_solver::initialize(microenvironment& m)
{
	if (m.mesh.dims >= 1)
		precompute_values(bx_, cx_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);
	if (m.mesh.dims >= 2)
		precompute_values(by_, cy_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
	if (m.mesh.dims >= 3)
		precompute_values(bz_, cz_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);

	dirichlet.initialize(m);
}

void diffusion_solver::precompute_values(real_t*& b, real_t*& c, index_t shape, index_t dims, index_t n,
										 const microenvironment& m)
{
	std::unique_ptr<real_t[]> bx, cx, ex;

	if (n == 1)
	{
		throw std::runtime_error("This implementation of diffusion solver does not support dimensions with size 1.");
	}

	solvers::host::diffusion_solver::precompute_values(bx, cx, ex, shape, dims, n, m, 1);

	CUCH(cudaMalloc(&b, m.substrates_count * n * sizeof(real_t)));
	CUCH(cudaMalloc(&c, m.substrates_count * sizeof(real_t)));

	CUCH(cudaMemcpy(b, bx.get(), m.substrates_count * n * sizeof(real_t), cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(c, cx.get(), m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
}

void diffusion_solver::solve_2d(microenvironment& m)
{
	dirichlet.solve_2d(m);

	run_solve_slice_2d_x(ctx_.diffusion_substrates, bx_, cx_, dirichlet.dirichlet_min_boundary_conditions[0],
						 dirichlet.dirichlet_min_boundary_values[0], dirichlet.dirichlet_max_boundary_conditions[0],
						 dirichlet.dirichlet_max_boundary_values[0], m.substrates_count, m.mesh.grid_shape[0],
						 m.mesh.grid_shape[1], ctx_.substrates_queue);

	dirichlet.solve_2d(m);

	run_solve_slice_2d_y(ctx_.diffusion_substrates, by_, cy_, dirichlet.dirichlet_min_boundary_conditions[1],
						 dirichlet.dirichlet_min_boundary_values[1], dirichlet.dirichlet_max_boundary_conditions[1],
						 dirichlet.dirichlet_max_boundary_values[1], m.substrates_count, m.mesh.grid_shape[0],
						 m.mesh.grid_shape[1], ctx_.substrates_queue);

	dirichlet.solve_2d(m);
}

void diffusion_solver::solve_3d(microenvironment& m)
{
	dirichlet.solve_3d(m);

	run_solve_slice_3d_x(ctx_.diffusion_substrates, bx_, cx_, dirichlet.dirichlet_min_boundary_conditions[0],
						 dirichlet.dirichlet_min_boundary_values[0], dirichlet.dirichlet_max_boundary_conditions[0],
						 dirichlet.dirichlet_max_boundary_values[0], m.substrates_count, m.mesh.grid_shape[0],
						 m.mesh.grid_shape[1], m.mesh.grid_shape[2], ctx_.substrates_queue);

	dirichlet.solve_3d(m);

	run_solve_slice_3d_y(ctx_.diffusion_substrates, by_, cy_, dirichlet.dirichlet_min_boundary_conditions[1],
						 dirichlet.dirichlet_min_boundary_values[1], dirichlet.dirichlet_max_boundary_conditions[1],
						 dirichlet.dirichlet_max_boundary_values[1], m.substrates_count, m.mesh.grid_shape[0],
						 m.mesh.grid_shape[1], m.mesh.grid_shape[2], ctx_.substrates_queue);

	dirichlet.solve_3d(m);

	run_solve_slice_3d_z(ctx_.diffusion_substrates, bz_, cz_, dirichlet.dirichlet_min_boundary_conditions[2],
						 dirichlet.dirichlet_min_boundary_values[2], dirichlet.dirichlet_max_boundary_conditions[2],
						 dirichlet.dirichlet_max_boundary_values[2], m.substrates_count, m.mesh.grid_shape[0],
						 m.mesh.grid_shape[1], m.mesh.grid_shape[2], ctx_.substrates_queue);

	dirichlet.solve_3d(m);
}

void diffusion_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}
