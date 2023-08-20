#include "diffusion_solver.h"

#include "../host/diffusion_solver.h"
#include "microenvironment.h"

using namespace biofvm;
using namespace solvers::device;

diffusion_solver::diffusion_solver(device_context& ctx)
	: opencl_solver(ctx, "diffusion_solver.cl"),
	  solve_slice_2d_x_(this->program_, "solve_slice_2d_x"),
	  solve_slice_2d_y_(this->program_, "solve_slice_2d_y"),
	  solve_slice_3d_x_(this->program_, "solve_slice_3d_x"),
	  solve_slice_3d_y_(this->program_, "solve_slice_3d_y"),
	  solve_slice_3d_z_(this->program_, "solve_slice_3d_z"),
	  dirichlet(ctx)
{}

void diffusion_solver::initialize(microenvironment& m)
{
	if (m.mesh.dims >= 1)
		precompute_values(bx_, cx_, ex_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);
	if (m.mesh.dims >= 2)
		precompute_values(by_, cy_, ey_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
	if (m.mesh.dims >= 3)
		precompute_values(bz_, cz_, ez_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);

	dirichlet.initialize(m);
}

void diffusion_solver::precompute_values(cl::Buffer& b, cl::Buffer& c, cl::Buffer& e, index_t shape, index_t dims,
										 index_t n, const microenvironment& m)
{
	std::unique_ptr<real_t[]> bx, cx, ex;

	solvers::host::diffusion_solver::precompute_values(bx, cx, ex, shape, dims, n, m, 1);

	if (n == 1)
	{
		b = cl::Buffer(ctx_.context, bx.get(), bx.get() + m.substrates_count, true);
	}
	else
	{
		b = cl::Buffer(ctx_.context, bx.get(), bx.get() + m.substrates_count * n, true);
		e = cl::Buffer(ctx_.context, ex.get(), ex.get() + m.substrates_count * (n - 1), true);
		c = cl::Buffer(ctx_.context, cx.get(), cx.get() + m.substrates_count, true);
	}
}

void diffusion_solver::solve_2d(microenvironment& m)
{
	dirichlet.solve_2d(m);

	solve_slice_2d_x_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[1] * m.substrates_count)),
					  ctx_.diffusion_substrates, bx_, cx_, m.substrates_count, m.mesh.grid_shape[0],
					  m.mesh.grid_shape[1]);

	dirichlet.solve_2d(m);

	solve_slice_2d_y_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.substrates_count)),
					  ctx_.diffusion_substrates, by_, cy_, m.substrates_count, m.mesh.grid_shape[0],
					  m.mesh.grid_shape[1]);

	dirichlet.solve_2d(m);
}

void diffusion_solver::solve_3d(microenvironment& m)
{
	dirichlet.solve_3d(m);

	solve_slice_3d_x_(
		cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[1] * m.mesh.grid_shape[2] * m.substrates_count)),
		ctx_.diffusion_substrates, bx_, cx_, m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1],
		m.mesh.grid_shape[2]);

	dirichlet.solve_3d(m);

	solve_slice_3d_y_(
		cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.mesh.grid_shape[2] * m.substrates_count)),
		ctx_.diffusion_substrates, by_, cy_, m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1],
		m.mesh.grid_shape[2]);

	dirichlet.solve_3d(m);

	solve_slice_3d_z_(
		cl::EnqueueArgs(ctx_.queue, cl::NDRange(m.mesh.grid_shape[0] * m.mesh.grid_shape[1] * m.substrates_count)),
		ctx_.diffusion_substrates, bz_, cz_, m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1],
		m.mesh.grid_shape[2]);

	dirichlet.solve_3d(m);
}

void diffusion_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}
