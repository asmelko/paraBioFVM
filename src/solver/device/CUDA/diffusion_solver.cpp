#include "diffusion_solver.h"

#include <stdexcept>

#include "microenvironment.h"
#include "solver/host/diffusion_solver.h"

using namespace biofvm;
using namespace solvers::device;

constexpr index_t work_block_size = 30;

void run_solve_slice_2d_x(real_t* densities, const real_t* b, const real_t* c, const real_t* a, const real_t* r_fwd,
						  const real_t* c_fwd, const real_t* a_bck, const real_t* c_bck, const real_t* c_rdc,
						  const real_t* r_rdc, const bool* dirichlet_conditions_min, const real_t* dirichlet_values_min,
						  const bool* dirichlet_conditions_max, const real_t* dirichlet_values_max,
						  index_t substrates_count, index_t x_size, index_t y_size, index_t work_block_size,
						  cudaStream_t& stream, index_t shared_mem_limit);
void run_solve_slice_2d_y(real_t* densities, const real_t* b, const real_t* c, const real_t* a, const real_t* r_fwd,
						  const real_t* c_fwd, const real_t* a_bck, const real_t* c_bck, const real_t* c_rdc,
						  const real_t* r_rdc, const bool* dirichlet_conditions_min, const real_t* dirichlet_values_min,
						  const bool* dirichlet_conditions_max, const real_t* dirichlet_values_max,
						  index_t substrates_count, index_t x_size, index_t y_size, index_t work_block_size,
						  cudaStream_t& stream, index_t shared_mem_limit);

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
	{
		precompute_values(bx_, cx_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);
		precompute_values_mod(a_def_x_, r_fwd_x_, c_fwd_x_, a_bck_x_, c_bck_x_, c_rdc_x_, r_rdc_x_,
							  m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], work_block_size, m);
	}
	if (m.mesh.dims >= 2)
	{
		precompute_values(by_, cy_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
		precompute_values_mod(a_def_y_, r_fwd_y_, c_fwd_y_, a_bck_y_, c_bck_y_, c_rdc_y_, r_rdc_y_,
							  m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], work_block_size, m);
	}
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
void diffusion_solver::precompute_values_mod(real_t*& a_def_, real_t*& r_fwd_, real_t*& c_fwd_, real_t*& a_bck_,
											 real_t*& c_bck_, real_t*& c_rdc_, real_t*& r_rdc_, index_t shape,
											 index_t dims, index_t n, index_t block_size, const microenvironment& m)
{
	std::unique_ptr<real_t[]> a, r_fwd, c_fwd, a_bck, c_bck, c_rdc, r_rdc;

	auto blocks = solvers::host::diffusion_solver::precompute_values_modified_thomas(
		a, r_fwd, c_fwd, a_bck, c_bck, c_rdc, r_rdc, shape, dims, n, block_size, m);

	CUCH(cudaMalloc(&a_def_, m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&r_fwd_, n * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&c_fwd_, n * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&a_bck_, n * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&c_bck_, n * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&c_rdc_, blocks * 2 * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&r_rdc_, blocks * 2 * m.substrates_count * sizeof(real_t)));

	CUCH(cudaMemcpy(a_def_, a.get(), m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(r_fwd_, r_fwd.get(), n * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(c_fwd_, c_fwd.get(), n * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(a_bck_, a_bck.get(), n * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(c_bck_, c_bck.get(), n * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(c_rdc_, c_rdc.get(), blocks * 2 * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(r_rdc_, r_rdc.get(), blocks * 2 * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice));
}

void diffusion_solver::solve_2d(microenvironment& m)
{
	dirichlet.solve_2d(m);

	run_solve_slice_2d_x(ctx_.diffusion_substrates, bx_, cx_, a_def_x_, r_fwd_x_, c_fwd_x_, a_bck_x_, c_bck_x_,
						 c_rdc_x_, r_rdc_x_, dirichlet.dirichlet_min_boundary_conditions[0],
						 dirichlet.dirichlet_min_boundary_values[0], dirichlet.dirichlet_max_boundary_conditions[0],
						 dirichlet.dirichlet_max_boundary_values[0], m.substrates_count, m.mesh.grid_shape[0],
						 m.mesh.grid_shape[1], work_block_size, ctx_.substrates_queue,
						 ctx_.device_properties.sharedMemPerBlock);

	dirichlet.solve_2d(m);

	run_solve_slice_2d_y(ctx_.diffusion_substrates, by_, cy_, a_def_y_, r_fwd_y_, c_fwd_y_, a_bck_y_, c_bck_y_,
						 c_rdc_y_, r_rdc_y_, dirichlet.dirichlet_min_boundary_conditions[1],
						 dirichlet.dirichlet_min_boundary_values[1], dirichlet.dirichlet_max_boundary_conditions[1],
						 dirichlet.dirichlet_max_boundary_values[1], m.substrates_count, m.mesh.grid_shape[0],
						 m.mesh.grid_shape[1], work_block_size, ctx_.substrates_queue,
						 ctx_.device_properties.sharedMemPerBlock);

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
