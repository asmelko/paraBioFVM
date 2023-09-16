#include "cell_solver.h"

#include <iostream>

#include "solver/host/cell_solver.h"

using namespace biofvm;
using namespace solvers::device;

void cell_solver::simulate_1d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	int work_items = agents_count * m.substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	if (compute_internalized_substrates_)
	{
		compute_fused_1d_(
			cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(blocks * block_size), cl::NDRange(block_size)),
			ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_, factors_,
			reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions, ballots_, conflicts_,
			conflicts_wrk_, m.mesh.voxel_volume(), m.substrates_count, m.mesh.bounding_box_mins[0],
			m.mesh.voxel_shape[0], m.mesh.grid_shape[0], agents_count);
	}
	else
	{
		compute_densities_1d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(agents_count * m.substrates_count)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0], agents_count);
	}
}

void cell_solver::simulate_2d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	int work_items = agents_count * m.substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	if (compute_internalized_substrates_)
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_fused_2d_, cl::NullRange, cl::NDRange(blocks * block_size),
												   cl::NDRange(block_size));
	}
	else
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_densities_2d_, cl::NullRange,
												   cl::NDRange(blocks * block_size), cl::NDRange(block_size));
	}
}

void cell_solver::simulate_3d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	int work_items = agents_count * m.substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	if (compute_internalized_substrates_)
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_fused_3d_, cl::NullRange, cl::NDRange(blocks * block_size),
												   cl::NDRange(block_size));
	}
	else
	{
		ctx_.substrates_queue.enqueueNDRangeKernel(compute_densities_3d_, cl::NullRange,
												   cl::NDRange(blocks * block_size), cl::NDRange(block_size));
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
			modify_kernel_2d(m);
		}
		else if (m.mesh.dims == 3)
		{
			modify_kernel_2d(m);
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
		prepare_kernel_2d(m);
	}
	else if (m.mesh.dims == 3)
	{
		prepare_kernel_3d(m);
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

void cell_solver::prepare_kernel_2d(microenvironment& m)
{
	index_t agents_size = get_agent_data(m).volumes.size();

	compute_fused_2d_.setArg(0, ctx_.internalized_substrates);
	compute_fused_2d_.setArg(1, ctx_.diffusion_substrates);
	compute_fused_2d_.setArg(2, numerators_);
	compute_fused_2d_.setArg(3, denominators_);
	compute_fused_2d_.setArg(4, factors_);
	compute_fused_2d_.setArg(5, reduced_numerators_);
	compute_fused_2d_.setArg(6, reduced_denominators_);
	compute_fused_2d_.setArg(7, reduced_factors_);
	compute_fused_2d_.setArg(8, ctx_.positions);
	compute_fused_2d_.setArg(9, ballots_);
	compute_fused_2d_.setArg(10, conflicts_);
	compute_fused_2d_.setArg(11, conflicts_wrk_);
	compute_fused_2d_.setArg<real_t>(12, m.mesh.voxel_volume());
	compute_fused_2d_.setArg(13, m.substrates_count);
	compute_fused_2d_.setArg(14, m.mesh.bounding_box_mins[0]);
	compute_fused_2d_.setArg(15, m.mesh.bounding_box_mins[1]);
	compute_fused_2d_.setArg(16, m.mesh.voxel_shape[0]);
	compute_fused_2d_.setArg(17, m.mesh.voxel_shape[1]);
	compute_fused_2d_.setArg(18, m.mesh.grid_shape[0]);
	compute_fused_2d_.setArg(19, m.mesh.grid_shape[1]);
	compute_fused_2d_.setArg(20, agents_size);

	compute_densities_2d_.setArg(0, ctx_.diffusion_substrates);
	compute_densities_2d_.setArg(1, reduced_numerators_);
	compute_densities_2d_.setArg(2, reduced_denominators_);
	compute_densities_2d_.setArg(3, reduced_factors_);
	compute_densities_2d_.setArg(4, ctx_.positions);
	compute_densities_2d_.setArg(5, ballots_);
	compute_densities_2d_.setArg<real_t>(6, m.mesh.voxel_volume());
	compute_densities_2d_.setArg(7, m.substrates_count);
	compute_densities_2d_.setArg(8, m.mesh.bounding_box_mins[0]);
	compute_densities_2d_.setArg(9, m.mesh.bounding_box_mins[1]);
	compute_densities_2d_.setArg(10, m.mesh.voxel_shape[0]);
	compute_densities_2d_.setArg(11, m.mesh.voxel_shape[1]);
	compute_densities_2d_.setArg(12, m.mesh.grid_shape[0]);
	compute_densities_2d_.setArg(13, m.mesh.grid_shape[1]);
	compute_densities_2d_.setArg(14, agents_size);
}

void cell_solver::prepare_kernel_3d(microenvironment& m)
{
	index_t agents_size = get_agent_data(m).volumes.size();

	compute_fused_3d_.setArg(0, ctx_.internalized_substrates);
	compute_fused_3d_.setArg(1, ctx_.diffusion_substrates);
	compute_fused_3d_.setArg(2, numerators_);
	compute_fused_3d_.setArg(3, denominators_);
	compute_fused_3d_.setArg(4, factors_);
	compute_fused_3d_.setArg(5, reduced_numerators_);
	compute_fused_3d_.setArg(6, reduced_denominators_);
	compute_fused_3d_.setArg(7, reduced_factors_);
	compute_fused_3d_.setArg(8, ctx_.positions);
	compute_fused_3d_.setArg(9, ballots_);
	compute_fused_3d_.setArg(10, conflicts_);
	compute_fused_3d_.setArg(11, conflicts_wrk_);
	compute_fused_3d_.setArg<real_t>(12, m.mesh.voxel_volume());
	compute_fused_3d_.setArg(13, m.substrates_count);
	compute_fused_3d_.setArg(14, m.mesh.bounding_box_mins[0]);
	compute_fused_3d_.setArg(15, m.mesh.bounding_box_mins[1]);
	compute_fused_3d_.setArg(16, m.mesh.bounding_box_mins[2]);
	compute_fused_3d_.setArg(17, m.mesh.voxel_shape[0]);
	compute_fused_3d_.setArg(18, m.mesh.voxel_shape[1]);
	compute_fused_3d_.setArg(19, m.mesh.voxel_shape[2]);
	compute_fused_3d_.setArg(20, m.mesh.grid_shape[0]);
	compute_fused_3d_.setArg(21, m.mesh.grid_shape[1]);
	compute_fused_3d_.setArg(22, m.mesh.grid_shape[2]);
	compute_fused_3d_.setArg(23, agents_size);

	compute_densities_3d_.setArg(0, ctx_.diffusion_substrates);
	compute_densities_3d_.setArg(1, reduced_numerators_);
	compute_densities_3d_.setArg(2, reduced_denominators_);
	compute_densities_3d_.setArg(3, reduced_factors_);
	compute_densities_3d_.setArg(4, ctx_.positions);
	compute_densities_3d_.setArg(5, ballots_);
	compute_densities_3d_.setArg<real_t>(6, m.mesh.voxel_volume());
	compute_densities_3d_.setArg(7, m.substrates_count);
	compute_densities_3d_.setArg(8, m.mesh.bounding_box_mins[0]);
	compute_densities_3d_.setArg(9, m.mesh.bounding_box_mins[1]);
	compute_densities_3d_.setArg(10, m.mesh.bounding_box_mins[2]);
	compute_densities_3d_.setArg(11, m.mesh.voxel_shape[0]);
	compute_densities_3d_.setArg(12, m.mesh.voxel_shape[1]);
	compute_densities_3d_.setArg(13, m.mesh.voxel_shape[2]);
	compute_densities_3d_.setArg(14, m.mesh.grid_shape[0]);
	compute_densities_3d_.setArg(15, m.mesh.grid_shape[1]);
	compute_densities_3d_.setArg(16, m.mesh.grid_shape[2]);
	compute_densities_3d_.setArg(17, agents_size);
}

void cell_solver::modify_kernel_2d(microenvironment& m)
{
	index_t agents_size = get_agent_data(m).volumes.size();

	compute_fused_2d_.setArg(0, ctx_.internalized_substrates);
	compute_fused_2d_.setArg(2, numerators_);
	compute_fused_2d_.setArg(3, denominators_);
	compute_fused_2d_.setArg(4, factors_);
	compute_fused_2d_.setArg(5, reduced_numerators_);
	compute_fused_2d_.setArg(6, reduced_denominators_);
	compute_fused_2d_.setArg(7, reduced_factors_);
	compute_fused_2d_.setArg(8, ctx_.positions);
	compute_fused_2d_.setArg(10, conflicts_);
	compute_fused_2d_.setArg(11, conflicts_wrk_);
	compute_fused_2d_.setArg(20, agents_size);

	compute_densities_2d_.setArg(1, reduced_numerators_);
	compute_densities_2d_.setArg(2, reduced_denominators_);
	compute_densities_2d_.setArg(3, reduced_factors_);
	compute_densities_2d_.setArg(4, ctx_.positions);
	compute_densities_2d_.setArg(14, agents_size);
}

void cell_solver::modify_kernel_3d(microenvironment& m)
{
	index_t agents_size = get_agent_data(m).volumes.size();

	compute_fused_3d_.setArg(0, ctx_.internalized_substrates);
	compute_fused_3d_.setArg(2, numerators_);
	compute_fused_3d_.setArg(3, denominators_);
	compute_fused_3d_.setArg(4, factors_);
	compute_fused_3d_.setArg(5, reduced_numerators_);
	compute_fused_3d_.setArg(6, reduced_denominators_);
	compute_fused_3d_.setArg(7, reduced_factors_);
	compute_fused_3d_.setArg(8, ctx_.positions);
	compute_fused_3d_.setArg(10, conflicts_);
	compute_fused_3d_.setArg(11, conflicts_wrk_);
	compute_fused_3d_.setArg(23, agents_size);

	compute_densities_3d_.setArg(1, reduced_numerators_);
	compute_densities_3d_.setArg(2, reduced_denominators_);
	compute_densities_3d_.setArg(3, reduced_factors_);
	compute_densities_3d_.setArg(4, ctx_.positions);
	compute_densities_3d_.setArg(17, agents_size);
}
