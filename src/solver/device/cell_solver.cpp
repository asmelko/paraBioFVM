#include "cell_solver.h"

#include <iostream>

#include "../host/cell_solver.h"

#define ROUND_UP(a, b) ((((a) + (b)-1) / (b)) * (b))

using namespace biofvm;
using namespace solvers::device;

void cell_solver::simulate_1d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		compute_fused_1d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
						  ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_, factors_,
						  reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions, ballots_,
						  conflicts_, conflicts_wrk_, m.mesh.voxel_volume(), m.substrates_count,
						  m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0],agents_count);
	}
	else
	{
		compute_densities_1d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0],agents_count);
	}
}

void cell_solver::simulate_2d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		compute_fused_2d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
						  ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_, factors_,
						  reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions, ballots_,
						  conflicts_, conflicts_wrk_, m.mesh.voxel_volume(), m.substrates_count,
						  m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.voxel_shape[0],
						  m.mesh.voxel_shape[1], m.mesh.grid_shape[0], m.mesh.grid_shape[1],agents_count);
	}
	else
	{
		compute_densities_2d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.voxel_shape[0],
							  m.mesh.voxel_shape[1], m.mesh.grid_shape[0], m.mesh.grid_shape[1],agents_count);
	}
}

void cell_solver::simulate_3d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		compute_fused_3d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
						  ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_, factors_,
						  reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions, ballots_,
						  conflicts_, conflicts_wrk_, m.mesh.voxel_volume(), m.substrates_count,
						  m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2],
						  m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0],
						  m.mesh.grid_shape[1], m.mesh.grid_shape[2],agents_count);
	}
	else
	{
		compute_densities_3d_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2],
							  m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0],
							  m.mesh.grid_shape[1], m.mesh.grid_shape[2],agents_count);
	}
}

void cell_solver::simulate_secretion_and_uptake(microenvironment& m, bool recompute)
{
	ctx_.cell_data_queue.finish();

	index_t agents_count = get_agent_data(m).agents_count;

	if (recompute)
	{
		resize(m);

		clear_and_ballot_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
						  ctx_.positions, ballots_, reduced_numerators_, reduced_denominators_, reduced_factors_,
						  conflicts_, conflicts_wrk_, m.substrates_count, m.mesh.bounding_box_mins[0],
						  m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0],
						  m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1],
						  m.mesh.grid_shape[2], m.mesh.dims,agents_count);

		ballot_and_sum_(cl::EnqueueArgs(ctx_.substrates_queue, cl::NDRange(ROUND_UP(agents_count * m.substrates_count, 256)), cl::NDRange(256)),
						reduced_numerators_, reduced_denominators_, reduced_factors_, numerators_, denominators_,
						factors_, ctx_.secretion_rates, ctx_.uptake_rates, ctx_.saturation_densities,
						ctx_.net_export_rates, ctx_.volumes, ctx_.positions, ballots_, conflicts_, conflicts_wrk_,
						m.mesh.voxel_volume(), m.diffusion_time_step, m.substrates_count, m.mesh.bounding_box_mins[0],
						m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0],
						m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1],
						m.mesh.grid_shape[2], m.mesh.dims,agents_count);
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
