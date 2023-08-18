#include "cell_solver.h"

#include <iostream>

#include "../host/cell_solver.h"

using namespace biofvm;
using namespace solvers::device;

void cell_solver::simulate_1d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_ && fuse_)
	{
		compute_fused_1d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
						  ctx_.internalized_substrates, ctx_.diffusion_substrates, reduced_numerators_,
						  reduced_denominators_, reduced_factors_, ctx_.positions, m.mesh.voxel_volume(),
						  m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0]);
	}
	else
	{
		if (compute_internalized_substrates_)
			compute_internalized_1d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
									 ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_,
									 denominators_, factors_, ctx_.positions, m.mesh.voxel_volume(), m.substrates_count,
									 m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0]);

		compute_densities_1d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0], m.mesh.grid_shape[0]);
	}
}

void cell_solver::simulate_2d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_ && fuse_)
	{
		compute_fused_2d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
						  ctx_.internalized_substrates, ctx_.diffusion_substrates, reduced_numerators_,
						  reduced_denominators_, reduced_factors_, ctx_.positions, m.mesh.voxel_volume(),
						  m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1],
						  m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.grid_shape[0], m.mesh.grid_shape[1]);
	}
	else
	{
		if (compute_internalized_substrates_)
			compute_internalized_2d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
									 ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_,
									 denominators_, factors_, ctx_.positions, m.mesh.voxel_volume(), m.substrates_count,
									 m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.voxel_shape[0],
									 m.mesh.voxel_shape[1], m.mesh.grid_shape[0], m.mesh.grid_shape[1]);

		compute_densities_2d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.voxel_shape[0],
							  m.mesh.voxel_shape[1], m.mesh.grid_shape[0], m.mesh.grid_shape[1]);
	}
}

void cell_solver::simulate_3d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_ && fuse_)
	{
		compute_fused_3d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
						  ctx_.internalized_substrates, ctx_.diffusion_substrates, reduced_numerators_,
						  reduced_denominators_, reduced_factors_, ctx_.positions, m.mesh.voxel_volume(),
						  m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1],
						  m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0], m.mesh.voxel_shape[1],
						  m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2]);
	}
	else
	{
		if (compute_internalized_substrates_)
		{
			compute_internalized_3d_(
				cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
				ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_, factors_,
				ctx_.positions, m.mesh.voxel_volume(), m.substrates_count, m.mesh.bounding_box_mins[0],
				m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0], m.mesh.voxel_shape[1],
				m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2]);
		}

		compute_densities_3d_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
							  ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_, reduced_factors_,
							  ctx_.positions, ballots_, m.mesh.voxel_volume(), m.substrates_count,
							  m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2],
							  m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0],
							  m.mesh.grid_shape[1], m.mesh.grid_shape[2]);
	}
}

void cell_solver::simulate_secretion_and_uptake(microenvironment& m, bool recompute)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (recompute)
	{
		resize(m);

		compute_intermediates_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)), numerators_,
							   denominators_, factors_, ctx_.secretion_rates, ctx_.uptake_rates,
							   ctx_.saturation_densities, ctx_.net_export_rates, ctx_.volumes, m.mesh.voxel_volume(),
							   m.diffusion_time_step, m.substrates_count);

		clear_and_ballot_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)), ctx_.positions,
						  ballots_, reduced_numerators_, reduced_denominators_, reduced_factors_, is_conflict_,
						  m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1],
						  m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0], m.mesh.voxel_shape[1],
						  m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2],
						  m.mesh.dims);

		ballot_and_sum_(cl::EnqueueArgs(ctx_.queue, cl::NDRange(agents_count * m.substrates_count)),
						reduced_numerators_, reduced_denominators_, reduced_factors_, numerators_, denominators_,
						factors_, ctx_.positions, ballots_, is_conflict_, m.substrates_count,
						m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2],
						m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0],
						m.mesh.grid_shape[1], m.mesh.grid_shape[2], m.mesh.dims);

		std::array<index_t, 1> is_conflict { 0 };
		cl::copy(ctx_.queue, is_conflict_, is_conflict.begin(), is_conflict.end());
		fuse_ = !is_conflict[0];
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
	}
}

void cell_solver::initialize(microenvironment& m)
{
	capacity_ = 0;
	compute_internalized_substrates_ = m.compute_internalized_substrates;
	fuse_ = false;

	resize(m);

	ballots_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, m.mesh.voxel_count() * m.substrates_count * sizeof(index_t));
	is_conflict_ = cl::Buffer(ctx_.context, CL_MEM_READ_WRITE, sizeof(index_t));
}

cell_solver::cell_solver(device_context& ctx)
	: opencl_solver(ctx, "cell_solver.cl"),
	  clear_and_ballot_(this->program_, "clear_and_ballot"),
	  compute_intermediates_(this->program_, "compute_intermediates"),
	  ballot_and_sum_(this->program_, "ballot_and_sum"),
	  compute_internalized_1d_(this->program_, "compute_internalized_1d"),
	  compute_densities_1d_(this->program_, "compute_densities_1d"),
	  compute_fused_1d_(this->program_, "compute_fused_1d"),
	  compute_internalized_2d_(this->program_, "compute_internalized_2d"),
	  compute_densities_2d_(this->program_, "compute_densities_2d"),
	  compute_fused_2d_(this->program_, "compute_fused_2d"),
	  compute_internalized_3d_(this->program_, "compute_internalized_3d"),
	  compute_densities_3d_(this->program_, "compute_densities_3d"),
	  compute_fused_3d_(this->program_, "compute_fused_3d")
{}
