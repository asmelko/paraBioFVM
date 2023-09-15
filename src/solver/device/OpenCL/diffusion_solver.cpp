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
		precompute_values(bz_, cz_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);

	dirichlet.initialize(m);

	if (m.mesh.dims == 2)
		prepare_2d_kernels(m);
	else
	{
		prepare_3d_kernel(m, bx_, cx_, solve_slice_3d_x_.getKernel(), x_global_, x_local_, 0);
		prepare_3d_kernel(m, by_, cy_, solve_slice_3d_y_.getKernel(), y_global_, y_local_, 1);
		prepare_3d_kernel(m, bz_, cz_, solve_slice_3d_z_.getKernel(), z_global_, z_local_, 2);
	}
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
		if (!x_shared_optim_)
		{
			ctx_.substrates_queue.enqueueNDRangeKernel(solve_slice_2d_x_block_.getKernel(), cl::NullRange, x_global_,
													   x_local_);
		}
		else
		{
			ctx_.substrates_queue.enqueueNDRangeKernel(solve_slice_2d_x_shared_.getKernel(), cl::NullRange, x_global_,
													   x_local_);
		}
	}

	dirichlet.solve_2d(m);

	{
		if (!y_shared_optim_)
		{
			ctx_.substrates_queue.enqueueNDRangeKernel(solve_slice_2d_y_block_.getKernel(), cl::NullRange, y_global_,
													   y_local_);
		}
		else
		{
			ctx_.substrates_queue.enqueueNDRangeKernel(solve_slice_2d_y_shared_.getKernel(), cl::NullRange, y_global_,
													   y_local_);
		}
	}

	dirichlet.solve_2d(m);
}

void diffusion_solver::solve_3d(microenvironment& m)
{
	dirichlet.solve_3d(m);

	ctx_.substrates_queue.enqueueNDRangeKernel(solve_slice_3d_x_.getKernel(), cl::NullRange, x_global_);

	dirichlet.solve_3d(m);

	ctx_.substrates_queue.enqueueNDRangeKernel(solve_slice_3d_y_.getKernel(), cl::NullRange, y_global_);

	dirichlet.solve_3d(m);

	ctx_.substrates_queue.enqueueNDRangeKernel(solve_slice_3d_z_.getKernel(), cl::NullRange, z_global_);

	dirichlet.solve_3d(m);
}

void diffusion_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}

void diffusion_solver::prepare_2d_kernels(microenvironment& m)
{
	// x
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

			x_shared_optim_ = false;
			x_global_ = cl::NDRange(blocks * block_size);
			x_local_ = cl::NDRange(block_size);

			auto kernel = solve_slice_2d_x_block_.getKernel();

			kernel.setArg(0, ctx_.diffusion_substrates);
			kernel.setArg(1, a_def_x_);
			kernel.setArg(2, r_fwd_x_);
			kernel.setArg(3, c_fwd_x_);
			kernel.setArg(4, a_bck_x_);
			kernel.setArg(5, c_bck_x_);
			kernel.setArg(6, c_rdc_x_);
			kernel.setArg(7, r_rdc_x_);
			kernel.setArg(8, dirichlet.dirichlet_min_boundary_conditions[0]);
			kernel.setArg(9, dirichlet.dirichlet_min_boundary_values[0]);
			kernel.setArg(10, dirichlet.dirichlet_max_boundary_conditions[0]);
			kernel.setArg(11, dirichlet.dirichlet_max_boundary_values[0]);
			kernel.setArg(12, m.substrates_count);
			kernel.setArg(13, m.mesh.grid_shape[0]);
			kernel.setArg(14, m.mesh.grid_shape[1]);
			kernel.setArg(15, work_block_size);
		}
		else
		{
			int block_size = std::min(512, max_warps * 32);
			int blocks = (work_items + block_size - 1) / block_size;
			int shmem = ((block_size + m.substrates_count - 1) / m.substrates_count) * slice_size + b_size;

			x_shared_optim_ = true;
			x_global_ = cl::NDRange(blocks * block_size);
			x_local_ = cl::NDRange(block_size);

			auto kernel = solve_slice_2d_x_shared_.getKernel();

			kernel.setArg(0, ctx_.diffusion_substrates);
			kernel.setArg(1, bx_);
			kernel.setArg(2, cx_);
			kernel.setArg(3, dirichlet.dirichlet_min_boundary_conditions[0]);
			kernel.setArg(4, dirichlet.dirichlet_min_boundary_values[0]);
			kernel.setArg(5, dirichlet.dirichlet_max_boundary_conditions[0]);
			kernel.setArg(6, dirichlet.dirichlet_max_boundary_values[0]);
			kernel.setArg(7, m.substrates_count);
			kernel.setArg(8, m.mesh.grid_shape[0]);
			kernel.setArg(9, m.mesh.grid_shape[1]);
			kernel.setArg<index_t>(10, padding / sizeof(real_t));
			kernel.setArg(11, shmem, nullptr);
		}
	}

	// y
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

			y_shared_optim_ = false;
			y_global_ = cl::NDRange(blocks * block_size);
			y_local_ = cl::NDRange(block_size);

			auto kernel = solve_slice_2d_y_block_.getKernel();

			kernel.setArg(0, ctx_.diffusion_substrates);
			kernel.setArg(1, a_def_y_);
			kernel.setArg(2, r_fwd_y_);
			kernel.setArg(3, c_fwd_y_);
			kernel.setArg(4, a_bck_y_);
			kernel.setArg(5, c_bck_y_);
			kernel.setArg(6, c_rdc_y_);
			kernel.setArg(7, r_rdc_y_);
			kernel.setArg(8, dirichlet.dirichlet_min_boundary_conditions[1]);
			kernel.setArg(9, dirichlet.dirichlet_min_boundary_values[1]);
			kernel.setArg(10, dirichlet.dirichlet_max_boundary_conditions[1]);
			kernel.setArg(11, dirichlet.dirichlet_max_boundary_values[1]);
			kernel.setArg(12, m.substrates_count);
			kernel.setArg(13, m.mesh.grid_shape[0]);
			kernel.setArg(14, m.mesh.grid_shape[1]);
			kernel.setArg(15, work_block_size);
		}
		else
		{
			int block_size = std::min(512, max_warps * 32);
			int blocks = (work_items + block_size - 1) / block_size;
			int shmem = block_size * slice_size + b_size;

			y_shared_optim_ = true;
			y_global_ = cl::NDRange(blocks * block_size);
			y_local_ = cl::NDRange(block_size);

			auto kernel = solve_slice_2d_y_shared_.getKernel();

			kernel.setArg(0, ctx_.diffusion_substrates);
			kernel.setArg(1, by_);
			kernel.setArg(2, cy_);
			kernel.setArg(3, dirichlet.dirichlet_min_boundary_conditions[1]);
			kernel.setArg(4, dirichlet.dirichlet_min_boundary_values[1]);
			kernel.setArg(5, dirichlet.dirichlet_max_boundary_conditions[1]);
			kernel.setArg(6, dirichlet.dirichlet_max_boundary_values[1]);
			kernel.setArg(7, m.substrates_count);
			kernel.setArg(8, m.mesh.grid_shape[0]);
			kernel.setArg(9, m.mesh.grid_shape[1]);
			kernel.setArg(10, 0);
			kernel.setArg(11, shmem, NULL);
		}
	}
}

void diffusion_solver::prepare_3d_kernel(microenvironment& m, cl::Buffer& b, cl::Buffer& c, cl::Kernel kernel,
										 cl::NDRange& global, cl::NDRange& local, index_t dim)
{
	index_t work = m.substrates_count;
	for (index_t d = 0; d < m.mesh.dims; d++)
	{
		if (d != dim)
			work *= m.mesh.grid_shape[d];
	}

	global = cl::NDRange(work);
	local = cl::NullRange;

	kernel.setArg(0, ctx_.diffusion_substrates);
	kernel.setArg(1, b);
	kernel.setArg(2, c);
	kernel.setArg(3, dirichlet.dirichlet_min_boundary_conditions[dim]);
	kernel.setArg(4, dirichlet.dirichlet_min_boundary_values[dim]);
	kernel.setArg(5, dirichlet.dirichlet_max_boundary_conditions[dim]);
	kernel.setArg(6, dirichlet.dirichlet_max_boundary_values[dim]);
	kernel.setArg(7, m.substrates_count);
	kernel.setArg(8, m.mesh.grid_shape[0]);
	kernel.setArg(9, m.mesh.grid_shape[1]);
	kernel.setArg(10, m.mesh.grid_shape[2]);
}