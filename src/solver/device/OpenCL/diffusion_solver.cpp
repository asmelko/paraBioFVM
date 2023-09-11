#include "diffusion_solver.h"

#include "microenvironment.h"
#include "solver/host/diffusion_solver.h"

using namespace biofvm;
using namespace solvers::device;

constexpr index_t work_block_size = 30;

diffusion_solver::diffusion_solver(device_context& ctx)
	: opencl_solver(ctx, "diffusion_solver.cl"),
	  solve_slice_2d_x_(this->program_, "solve_slice_2d_x"),
	  solve_slice_2d_y_(this->program_, "solve_slice_2d_y"),
	  solve_slice_2d_x_block_(this->program_, "solve_slice_2d_x_block"),
	  solve_slice_2d_y_block_(this->program_, "solve_slice_2d_y_block"),
	  solve_slice_2d_x_shared_(this->program_, "solve_slice_2d_x_shared_full"),
	  solve_slice_2d_y_shared_(this->program_, "solve_slice_2d_y_shared_full"),
	  solve_slice_3d_x_(this->program_, "solve_slice_3d_x"),
	  solve_slice_3d_y_(this->program_, "solve_slice_3d_y"),
	  solve_slice_3d_z_(this->program_, "solve_slice_3d_z"),
	  solve_slice_3d_x_block_(this->program_, "solve_slice_3d_x_block"),
	  solve_slice_3d_y_block_(this->program_, "solve_slice_3d_y_block"),
	  solve_slice_3d_z_block_(this->program_, "solve_slice_3d_z_block"),
	  dirichlet(ctx)
{}

void diffusion_solver::initialize(microenvironment& m)
{
	if (m.mesh.dims >= 1)
	{
		precompute_values(bx_, cx_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);
		precompute_values_modified_thomas(a_def_x_, r_fwd_x_, c_fwd_x_, a_bck_x_, c_bck_x_, c_rdc_x_, r_rdc_x_,
										  m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], work_block_size, m);
	}
	if (m.mesh.dims >= 2)
	{
		precompute_values(by_, cy_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
		precompute_values_modified_thomas(a_def_y_, r_fwd_y_, c_fwd_y_, a_bck_y_, c_bck_y_, c_rdc_y_, r_rdc_y_,
										  m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], work_block_size, m);
	}
	if (m.mesh.dims >= 3)
	{
		precompute_values(bz_, cz_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);
		precompute_values_modified_thomas(a_def_z_, r_fwd_z_, c_fwd_z_, a_bck_z_, c_bck_z_, c_rdc_z_, r_rdc_z_,
										  m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], work_block_size, m);
	}

	dirichlet.initialize(m);
}

void diffusion_solver::precompute_values(cl::Buffer& b, cl::Buffer& c, index_t shape, index_t dims, index_t n,
										 const microenvironment& m)
{
	std::unique_ptr<real_t[]> bx, cx, ex;

	if (n == 1)
	{
		throw std::runtime_error("This implementation of diffusion solver does not support dimensions with size 1.");
	}

	solvers::host::diffusion_solver::precompute_values(bx, cx, ex, shape, dims, n, m, 1);

	b = cl::Buffer(ctx_.context, bx.get(), bx.get() + m.substrates_count * n, true);
	c = cl::Buffer(ctx_.context, cx.get(), cx.get() + m.substrates_count, true);
}

void diffusion_solver::precompute_values_modified_thomas(cl::Buffer& a, cl::Buffer& r_fwd, cl::Buffer& c_fwd,
														 cl::Buffer& a_bck, cl::Buffer& c_bck, cl::Buffer& c_rdc,
														 cl::Buffer& r_rdc, index_t shape, index_t dims, index_t n,
														 index_t block_size, const microenvironment& m)
{
	std::unique_ptr<real_t[]> a_, r_fwd_, c_fwd_, a_bck_, c_bck_, c_rdc_, r_rdc_;

	auto blocks = solvers::host::diffusion_solver::precompute_values_modified_thomas(
		a_, r_fwd_, c_fwd_, a_bck_, c_bck_, c_rdc_, r_rdc_, shape, dims, n, block_size, m);

	a = cl::Buffer(ctx_.context, a_.get(), a_.get() + m.substrates_count, true);
	r_fwd = cl::Buffer(ctx_.context, r_fwd_.get(), r_fwd_.get() + m.substrates_count * n, true);
	c_fwd = cl::Buffer(ctx_.context, c_fwd_.get(), c_fwd_.get() + m.substrates_count * n, true);
	a_bck = cl::Buffer(ctx_.context, a_bck_.get(), a_bck_.get() + m.substrates_count * n, true);
	c_bck = cl::Buffer(ctx_.context, c_bck_.get(), c_bck_.get() + m.substrates_count * n, true);
	c_rdc = cl::Buffer(ctx_.context, c_rdc_.get(), c_rdc_.get() + blocks * 2 * m.substrates_count, true);
	r_rdc = cl::Buffer(ctx_.context, r_rdc_.get(), r_rdc_.get() + blocks * 2 * m.substrates_count, true);
}

void diffusion_solver::solve_2d(microenvironment& m)
{
	dirichlet.solve_2d(m);

	{
		index_t shared_mem_limit = ctx_.local_mem_limit;

		size_t b_size = m.mesh.grid_shape[0] * m.substrates_count * sizeof(real_t);

		int slice_remainder = (m.mesh.grid_shape[0] * m.substrates_count * sizeof(real_t)) % 128;
		int s_remainder = (m.substrates_count * sizeof(real_t)) % 128;

		int padding;
		if (slice_remainder <= s_remainder)
		{
			padding = s_remainder - slice_remainder;
		}
		else
		{
			padding = 128 - slice_remainder + s_remainder;
		}

		index_t slice_size = m.mesh.grid_shape[0] * m.substrates_count * sizeof(real_t) + padding;

		shared_mem_limit -= b_size;
		int max_slices = shared_mem_limit / slice_size;
		int slices_in_warp = (32 + m.substrates_count - 1) / m.substrates_count;
		int max_warps = max_slices / slices_in_warp;

		int work_items = m.mesh.grid_shape[1] * m.substrates_count;

		if (max_warps <= 0)
		{
			int block_size = 256;
			int blocks = (work_items + 32 - 1) / 32;

			solve_slice_2d_x_block_(
				cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
				ctx_.diffusion_substrates, a_def_x_, r_fwd_x_, c_fwd_x_, a_bck_x_, c_bck_x_, c_rdc_x_, r_rdc_x_,
				dirichlet.dirichlet_min_boundary_conditions[0], dirichlet.dirichlet_min_boundary_values[0],
				dirichlet.dirichlet_max_boundary_conditions[0], dirichlet.dirichlet_max_boundary_values[0],
				m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], work_block_size);
		}
		else
		{
			int block_size = std::min(512, max_warps * 32);
			int blocks = (work_items + block_size - 1) / block_size;
			int shmem = ((block_size + m.substrates_count - 1) / m.substrates_count) * slice_size + b_size;

			solve_slice_2d_x_shared_.getKernel().setArg(11, shmem, NULL);

			solve_slice_2d_x_shared_(
				cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
				ctx_.diffusion_substrates, bx_, cx_, dirichlet.dirichlet_min_boundary_conditions[0],
				dirichlet.dirichlet_min_boundary_values[0], dirichlet.dirichlet_max_boundary_conditions[0],
				dirichlet.dirichlet_max_boundary_values[0], m.substrates_count, m.mesh.grid_shape[0],
				m.mesh.grid_shape[1], padding / sizeof(real_t));
		}
	}

	dirichlet.solve_2d(m);

	{
		index_t shared_mem_limit = ctx_.local_mem_limit;

		index_t b_size = m.mesh.grid_shape[1] * m.substrates_count * sizeof(real_t);
		index_t slice_size = m.mesh.grid_shape[1] * sizeof(real_t);

		shared_mem_limit -= b_size;
		int max_slices = shared_mem_limit / slice_size;
		int max_warps = max_slices / 32;

		int work_items = m.mesh.grid_shape[0] * m.substrates_count;

		if (max_warps <= 0)
		{
			int block_size = 256;
			int blocks = (work_items + 32 - 1) / 32;

			solve_slice_2d_y_block_(
				cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
				ctx_.diffusion_substrates, a_def_y_, r_fwd_y_, c_fwd_y_, a_bck_y_, c_bck_y_, c_rdc_y_, r_rdc_y_,
				dirichlet.dirichlet_min_boundary_conditions[1], dirichlet.dirichlet_min_boundary_values[1],
				dirichlet.dirichlet_max_boundary_conditions[1], dirichlet.dirichlet_max_boundary_values[1],
				m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], work_block_size);
		}
		else
		{
			int block_size = std::min(512, max_warps * 32);
			int blocks = (work_items + block_size - 1) / block_size;
			int shmem = block_size * slice_size + b_size;

			solve_slice_2d_y_shared_.getKernel().setArg(11, shmem, NULL);

			solve_slice_2d_y_shared_(
				cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
				ctx_.diffusion_substrates, by_, cy_, dirichlet.dirichlet_min_boundary_conditions[1],
				dirichlet.dirichlet_min_boundary_values[1], dirichlet.dirichlet_max_boundary_conditions[1],
				dirichlet.dirichlet_max_boundary_values[1], m.substrates_count, m.mesh.grid_shape[0],
				m.mesh.grid_shape[1], 0);
		}
	}

	dirichlet.solve_2d(m);
}

void diffusion_solver::solve_3d(microenvironment& m)
{
	dirichlet.solve_3d(m);

	{
		int block_size = 256;
		int work_items = m.mesh.grid_shape[1] * m.mesh.grid_shape[2] * m.substrates_count;
		int blocks = (work_items + 32 - 1) / 32;

		solve_slice_3d_x_block_(
			cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
			ctx_.diffusion_substrates, a_def_x_, r_fwd_x_, c_fwd_x_, a_bck_x_, c_bck_x_, c_rdc_x_, r_rdc_x_,
			dirichlet.dirichlet_min_boundary_conditions[0], dirichlet.dirichlet_min_boundary_values[0],
			dirichlet.dirichlet_max_boundary_conditions[0], dirichlet.dirichlet_max_boundary_values[0],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2], work_block_size);
	}

	dirichlet.solve_3d(m);

	{
		int block_size = 256;
		int work_items = m.mesh.grid_shape[0] * m.mesh.grid_shape[2] * m.substrates_count;
		int blocks = (work_items + 32 - 1) / 32;

		solve_slice_3d_y_block_(
			cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
			ctx_.diffusion_substrates, a_def_y_, r_fwd_y_, c_fwd_y_, a_bck_y_, c_bck_y_, c_rdc_y_, r_rdc_y_,
			dirichlet.dirichlet_min_boundary_conditions[1], dirichlet.dirichlet_min_boundary_values[1],
			dirichlet.dirichlet_max_boundary_conditions[1], dirichlet.dirichlet_max_boundary_values[1],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2], work_block_size);
	}


	dirichlet.solve_3d(m);

	{
		int block_size = 256;
		int work_items = m.mesh.grid_shape[0] * m.mesh.grid_shape[1] * m.substrates_count;
		int blocks = (work_items + 32 - 1) / 32;

		solve_slice_3d_z_block_(
			cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
			ctx_.diffusion_substrates, a_def_z_, r_fwd_z_, c_fwd_z_, a_bck_z_, c_bck_z_, c_rdc_z_, r_rdc_z_,
			dirichlet.dirichlet_min_boundary_conditions[2], dirichlet.dirichlet_min_boundary_values[2],
			dirichlet.dirichlet_max_boundary_conditions[2], dirichlet.dirichlet_max_boundary_values[2],
			m.substrates_count, m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2], work_block_size);
	}

	dirichlet.solve_3d(m);
}

void diffusion_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}
