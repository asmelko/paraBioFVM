#include "cell_solver.h"

#include <iostream>

#include "solver/host/cell_solver.h"

using namespace biofvm;
using namespace solvers::device;

void cell_solver::simulate_1d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		compute_fused_1d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(agents_count * m.substrates_count)),
						  ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_, factors_,
						  reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions, ballots_,
						  conflicts_, conflicts_wrk_, m.mesh.voxel_volume(), m.substrates_count,
						  m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0]);
	}
	else
	{
		compute_densities_1d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(agents_count * m.substrates_count)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0]);
	}
}

void cell_solver::simulate_2d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_fused_2d_.getKernel(), cl::NullRange,
												   cl::NDRange(agents_count * m.substrates_count));
	}
	else
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_densities_2d_.getKernel(), cl::NullRange,
												   cl::NDRange(agents_count * m.substrates_count));
	}
}

void cell_solver::simulate_3d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_fused_3d_.getKernel(), cl::NullRange,
												   cl::NDRange(agents_count * m.substrates_count));
	}
	else
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_densities_3d_.getKernel(), cl::NullRange,
												   cl::NDRange(agents_count * m.substrates_count));
	}
}

void cell_solver::simulate_secretion_and_uptake(microenvironment& m, bool recompute)
{
	ctx_.cell_data_queue.finish();

	index_t agents_count = get_agent_data(m).agents_count;

	if (recompute)
	{
		resize(m);

		if (m.mesh.dims == 2)
		{
			modify_kernel_2d(m, compute_fused_2d_.getKernel(), compute_densities_2d_.getKernel());
		}
		else if (m.mesh.dims == 3)
		{
			modify_kernel_2d(m, compute_fused_3d_.getKernel(), compute_densities_3d_.getKernel());
		}

		clear_and_ballot_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(agents_count * m.substrates_count)),
						  ctx_.positions, ballots_, reduced_numerators_, reduced_denominators_, reduced_factors_,
						  conflicts_, conflicts_wrk_, m.substrates_count, m.mesh.bounding_box_mins[0],
						  m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0],
						  m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1],
						  m.mesh.grid_shape[2], m.mesh.dims);

		ballot_and_sum_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(agents_count * m.substrates_count)),
						reduced_numerators_, reduced_denominators_, reduced_factors_, numerators_, denominators_,
						factors_, ctx_.secretion_rates, ctx_.uptake_rates, ctx_.saturation_densities,
						ctx_.net_export_rates, ctx_.volumes, ctx_.positions, ballots_, conflicts_, conflicts_wrk_,
						m.mesh.voxel_volume(), m.diffusion_time_step, m.substrates_count, m.mesh.bounding_box_mins[0],
						m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0],
						m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1],
						m.mesh.grid_shape[2], m.mesh.dims);
	}

	if (m.mesh.dims == 1)
	{
		simulate_1d(m);
	}
	else if (m.mesh.dims == 2)
	{
		simulate_2d(m);
	}
	else if (m.mesh.dims == 3)
	{
		simulate_3d(m);
	}
}

void cell_solver::release_internalized_substrates(microenvironment& m, index_t index)
{
	if (!compute_internalized_substrates_)
		return;

	host::cell_solver::release_internalized_substrates(get_agent_data(m), index);
}

void cell_solver::resize(microenvironment& m)
{
	if (get_agent_data(m).volumes.size() != capacity_)
	{
		capacity_ = get_agent_data(m).volumes.size();

		numerators_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * m.substrates_count * sizeof(real_t));
		denominators_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * m.substrates_count * sizeof(real_t));
		factors_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * m.substrates_count * sizeof(real_t));

		reduced_numerators_ =
			cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * m.substrates_count * sizeof(real_t));
		reduced_denominators_ =
			cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * m.substrates_count * sizeof(real_t));
		reduced_factors_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * m.substrates_count * sizeof(real_t));

		conflicts_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * sizeof(index_t));
		conflicts_wrk_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, capacity_ * m.substrates_count * sizeof(index_t));
	}
}

void cell_solver::initialize(microenvironment& m)
{
	capacity_ = 0;
	compute_internalized_substrates_ = m.compute_internalized_substrates;

	resize(m);

	ballots_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, m.mesh.voxel_count() * sizeof(index_t));

	if (m.mesh.dims == 2)
	{
		prepare_kernel_2d(m, compute_fused_2d_.getKernel(), compute_densities_2d_.getKernel());
	}
	else if (m.mesh.dims == 3)
	{
		prepare_kernel_3d(m, compute_fused_3d_.getKernel(), compute_densities_3d_.getKernel());
	}
}

cell_solver::cell_solver(device_context& ctx)
	: opencl_solver(ctx, "cell_solver.cl"),
	  clear_and_ballot_(this->program_, "clear_and_ballot"),
	  ballot_and_sum_(this->program_, "ballot_and_sum"),
	  compute_densities_1d_(this->program_, "compute_densities_1d"),
	  compute_fused_1d_(this->program_, "compute_fused_1d"),
	  compute_densities_2d_(this->program_, "compute_densities_2d"),
	  compute_fused_2d_(this->program_, "compute_fused_2d"),
	  compute_densities_3d_(this->program_, "compute_densities_3d"),
	  compute_fused_3d_(this->program_, "compute_fused_3d")
{}

void cell_solver::prepare_kernel_2d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel)
{
	fused_kernel.setArg(0, ctx_.internalized_substrates);
	fused_kernel.setArg(1, ctx_.diffusion_substrates);
	fused_kernel.setArg(2, numerators_);
	fused_kernel.setArg(3, denominators_);
	fused_kernel.setArg(4, factors_);
	fused_kernel.setArg(5, reduced_numerators_);
	fused_kernel.setArg(6, reduced_denominators_);
	fused_kernel.setArg(7, reduced_factors_);
	fused_kernel.setArg(8, ctx_.positions);
	fused_kernel.setArg(9, ballots_);
	fused_kernel.setArg(10, conflicts_);
	fused_kernel.setArg(11, conflicts_wrk_);
	fused_kernel.setArg<real_t>(12, m.mesh.voxel_volume());
	fused_kernel.setArg(13, m.substrates_count);
	fused_kernel.setArg(14, m.mesh.bounding_box_mins[0]);
	fused_kernel.setArg(15, m.mesh.bounding_box_mins[1]);
	fused_kernel.setArg(16, m.mesh.voxel_shape[0]);
	fused_kernel.setArg(17, m.mesh.voxel_shape[1]);
	fused_kernel.setArg(18, m.mesh.grid_shape[0]);
	fused_kernel.setArg(19, m.mesh.grid_shape[1]);

	dens_kernel.setArg(0, ctx_.diffusion_substrates);
	dens_kernel.setArg(1, reduced_numerators_);
	dens_kernel.setArg(2, reduced_denominators_);
	dens_kernel.setArg(3, reduced_factors_);
	dens_kernel.setArg(4, ctx_.positions);
	dens_kernel.setArg(5, ballots_);
	dens_kernel.setArg<real_t>(6, m.mesh.voxel_volume());
	dens_kernel.setArg(7, m.substrates_count);
	dens_kernel.setArg(8, m.mesh.bounding_box_mins[0]);
	dens_kernel.setArg(9, m.mesh.bounding_box_mins[1]);
	dens_kernel.setArg(10, m.mesh.voxel_shape[0]);
	dens_kernel.setArg(11, m.mesh.voxel_shape[1]);
	dens_kernel.setArg(12, m.mesh.grid_shape[0]);
	dens_kernel.setArg(13, m.mesh.grid_shape[1]);
}

void cell_solver::prepare_kernel_3d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel)
{
	fused_kernel.setArg(0, ctx_.internalized_substrates);
	fused_kernel.setArg(1, ctx_.diffusion_substrates);
	fused_kernel.setArg(2, numerators_);
	fused_kernel.setArg(3, denominators_);
	fused_kernel.setArg(4, factors_);
	fused_kernel.setArg(5, reduced_numerators_);
	fused_kernel.setArg(6, reduced_denominators_);
	fused_kernel.setArg(7, reduced_factors_);
	fused_kernel.setArg(8, ctx_.positions);
	fused_kernel.setArg(9, ballots_);
	fused_kernel.setArg(10, conflicts_);
	fused_kernel.setArg(11, conflicts_wrk_);
	fused_kernel.setArg<real_t>(12, m.mesh.voxel_volume());
	fused_kernel.setArg(13, m.substrates_count);
	fused_kernel.setArg(14, m.mesh.bounding_box_mins[0]);
	fused_kernel.setArg(15, m.mesh.bounding_box_mins[1]);
	fused_kernel.setArg(16, m.mesh.bounding_box_mins[2]);
	fused_kernel.setArg(17, m.mesh.voxel_shape[0]);
	fused_kernel.setArg(18, m.mesh.voxel_shape[1]);
	fused_kernel.setArg(19, m.mesh.voxel_shape[2]);
	fused_kernel.setArg(20, m.mesh.grid_shape[0]);
	fused_kernel.setArg(21, m.mesh.grid_shape[1]);
	fused_kernel.setArg(22, m.mesh.grid_shape[2]);

	dens_kernel.setArg(0, ctx_.diffusion_substrates);
	dens_kernel.setArg(1, reduced_numerators_);
	dens_kernel.setArg(2, reduced_denominators_);
	dens_kernel.setArg(3, reduced_factors_);
	dens_kernel.setArg(4, ctx_.positions);
	dens_kernel.setArg(5, ballots_);
	dens_kernel.setArg<real_t>(6, m.mesh.voxel_volume());
	dens_kernel.setArg(7, m.substrates_count);
	dens_kernel.setArg(8, m.mesh.bounding_box_mins[0]);
	dens_kernel.setArg(9, m.mesh.bounding_box_mins[1]);
	dens_kernel.setArg(10, m.mesh.bounding_box_mins[2]);
	dens_kernel.setArg(11, m.mesh.voxel_shape[0]);
	dens_kernel.setArg(12, m.mesh.voxel_shape[1]);
	dens_kernel.setArg(13, m.mesh.voxel_shape[2]);
	dens_kernel.setArg(14, m.mesh.grid_shape[0]);
	dens_kernel.setArg(15, m.mesh.grid_shape[1]);
	dens_kernel.setArg(16, m.mesh.grid_shape[2]);
}

void cell_solver::modify_kernel_2d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel)
{
	fused_kernel.setArg(0, ctx_.internalized_substrates);
	fused_kernel.setArg(2, numerators_);
	fused_kernel.setArg(3, denominators_);
	fused_kernel.setArg(4, factors_);
	fused_kernel.setArg(5, reduced_numerators_);
	fused_kernel.setArg(6, reduced_denominators_);
	fused_kernel.setArg(7, reduced_factors_);
	fused_kernel.setArg(8, ctx_.positions);
	fused_kernel.setArg(10, conflicts_);
	fused_kernel.setArg(11, conflicts_wrk_);

	dens_kernel.setArg(1, reduced_numerators_);
	dens_kernel.setArg(2, reduced_denominators_);
	dens_kernel.setArg(3, reduced_factors_);
	dens_kernel.setArg(4, ctx_.positions);
}

void cell_solver::modify_kernel_3d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel)
{
	fused_kernel.setArg(0, ctx_.internalized_substrates);
	fused_kernel.setArg(2, numerators_);
	fused_kernel.setArg(3, denominators_);
	fused_kernel.setArg(4, factors_);
	fused_kernel.setArg(5, reduced_numerators_);
	fused_kernel.setArg(6, reduced_denominators_);
	fused_kernel.setArg(7, reduced_factors_);
	fused_kernel.setArg(8, ctx_.positions);
	fused_kernel.setArg(10, conflicts_);
	fused_kernel.setArg(11, conflicts_wrk_);

	dens_kernel.setArg(1, reduced_numerators_);
	dens_kernel.setArg(2, reduced_denominators_);
	dens_kernel.setArg(3, reduced_factors_);
	dens_kernel.setArg(4, ctx_.positions);
}

void cell_solver::set_kernel_args(microenvironment& m, bool update)
{
	if (m.mesh.dims == 2)
	{
		prepare_kernel_2d(m, compute_fused_2d_.getKernel(), compute_densities_2d_.getKernel());
	}
	else if (m.mesh.dims == 3)
	{
		prepare_kernel_3d(m, compute_fused_3d_.getKernel(), compute_densities_3d_.getKernel());
	}
}
