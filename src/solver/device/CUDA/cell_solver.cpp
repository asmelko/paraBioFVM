#include "cell_solver.h"

#include <iostream>

#include "solver/host/cell_solver.h"

using namespace biofvm;
using namespace solvers::device;

void run_clear_and_ballot(const real_t* cell_positions, index_t* ballots, real_t* reduced_numerators,
						  real_t* reduced_denominators, real_t* reduced_factors, int* conflicts, int* conflicts_wrk,
						  index_t n, index_t substrates_count, index_t x_min, index_t y_min, index_t z_min,
						  index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size,
						  index_t dims, cudaStream_t& stream);

void run_ballot_and_sum(real_t* reduced_numerators, real_t* reduced_denominators, real_t* reduced_factors,
						real_t* numerators, real_t* denominators, real_t* factors, const real_t* secretion_rates,
						const real_t* uptake_rates, const real_t* saturation_densities, const real_t* net_export_rates,
						const real_t* cell_volumes, const real_t* cell_positions, index_t* ballots, index_t* conflicts,
						index_t* conflicts_wrk, index_t n, real_t voxel_volume, real_t time_step,
						index_t substrates_count, index_t x_min, index_t y_min, index_t z_min, index_t x_dt,
						index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size, index_t dims,
						cudaStream_t& stream);

void run_compute_densities_1d(real_t* substrate_densities, const real_t* numerator, const real_t* denominator,
							  const real_t* factor, const real_t* cell_positions, const index_t* ballots, index_t n,
							  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t x_dt,
							  index_t x_size, cudaStream_t& stream);

void run_compute_densities_2d(real_t* substrate_densities, const real_t* numerator, const real_t* denominator,
							  const real_t* factor, const real_t* cell_positions, const index_t* ballots, index_t n,
							  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min, index_t x_dt,
							  index_t y_dt, index_t x_size, index_t y_size, cudaStream_t& stream);

void run_compute_densities_3d(real_t* substrate_densities, const real_t* numerator, const real_t* denominator,
							  const real_t* factor, const real_t* cell_positions, const index_t* ballots, index_t n,
							  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
							  index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size,
							  index_t z_size, cudaStream_t& stream);

void run_compute_fused_1d(real_t* internalized_substrates, real_t* substrate_densities, const real_t* numerator,
						  const real_t* denominator, const real_t* factor, const real_t* reduced_numerator,
						  const real_t* reduced_denominator, const real_t* reduced_factor, const real_t* cell_positions,
						  const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
						  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t x_dt, index_t x_size,
						  cudaStream_t& stream);

void run_compute_fused_2d(real_t* internalized_substrates, real_t* substrate_densities, const real_t* numerator,
						  const real_t* denominator, const real_t* factor, const real_t* reduced_numerator,
						  const real_t* reduced_denominator, const real_t* reduced_factor, const real_t* cell_positions,
						  const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
						  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min, index_t x_dt,
						  index_t y_dt, index_t x_size, index_t y_size, cudaStream_t& stream);

void run_compute_fused_3d(real_t* internalized_substrates, real_t* substrate_densities, const real_t* numerator,
						  const real_t* denominator, const real_t* factor, const real_t* reduced_numerator,
						  const real_t* reduced_denominator, const real_t* reduced_factor, const real_t* cell_positions,
						  const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
						  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min, index_t z_min,
						  index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size,
						  cudaStream_t& stream);


void cell_solver::simulate_1d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		run_compute_fused_1d(ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_,
							 factors_, reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions,
							 ballots_, conflicts_, conflicts_wrk_, agents_count, m.mesh.voxel_volume(),
							 m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0],
							 m.mesh.grid_shape[0], ctx_.substrates_queue);
	}
	else
	{
		run_compute_densities_1d(ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_,
								 reduced_factors_, ctx_.positions, ballots_, agents_count, m.mesh.voxel_volume(),
								 m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.voxel_shape[0],
								 m.mesh.grid_shape[0], ctx_.substrates_queue);
	}
}

void cell_solver::simulate_2d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		run_compute_fused_2d(ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_,
							 factors_, reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions,
							 ballots_, conflicts_, conflicts_wrk_, agents_count, m.mesh.voxel_volume(),
							 m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1],
							 m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.grid_shape[0], m.mesh.grid_shape[1],
							 ctx_.substrates_queue);
	}
	else
	{
		run_compute_densities_2d(ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_,
								 reduced_factors_, ctx_.positions, ballots_, agents_count, m.mesh.voxel_volume(),
								 m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1],
								 m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.grid_shape[0],
								 m.mesh.grid_shape[1], ctx_.substrates_queue);
	}
}

void cell_solver::simulate_3d(microenvironment& m)
{
	index_t agents_count = get_agent_data(m).agents_count;

	if (compute_internalized_substrates_)
	{
		run_compute_fused_3d(ctx_.internalized_substrates, ctx_.diffusion_substrates, numerators_, denominators_,
							 factors_, reduced_numerators_, reduced_denominators_, reduced_factors_, ctx_.positions,
							 ballots_, conflicts_, conflicts_wrk_, agents_count, m.mesh.voxel_volume(),
							 m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1],
							 m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0], m.mesh.voxel_shape[1],
							 m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1], m.mesh.grid_shape[2],
							 ctx_.substrates_queue);
	}
	else
	{
		run_compute_densities_3d(ctx_.diffusion_substrates, reduced_numerators_, reduced_denominators_,
								 reduced_factors_, ctx_.positions, ballots_, agents_count, m.mesh.voxel_volume(),
								 m.substrates_count, m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1],
								 m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0], m.mesh.voxel_shape[1],
								 m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1],
								 m.mesh.grid_shape[2], ctx_.substrates_queue);
	}
}

void cell_solver::simulate_secretion_and_uptake(microenvironment& m, bool recompute)
{
	CUCH(cudaStreamSynchronize(ctx_.cell_data_queue));

	index_t agents_count = get_agent_data(m).agents_count;

	if (recompute)
	{
		resize(m);

		run_clear_and_ballot(ctx_.positions, ballots_, reduced_numerators_, reduced_denominators_, reduced_factors_,
							 conflicts_, conflicts_wrk_, agents_count, m.substrates_count, m.mesh.bounding_box_mins[0],
							 m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2], m.mesh.voxel_shape[0],
							 m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0], m.mesh.grid_shape[1],
							 m.mesh.grid_shape[2], m.mesh.dims, ctx_.substrates_queue);

		run_ballot_and_sum(reduced_numerators_, reduced_denominators_, reduced_factors_, numerators_, denominators_,
						   factors_, ctx_.secretion_rates, ctx_.uptake_rates, ctx_.saturation_densities,
						   ctx_.net_export_rates, ctx_.volumes, ctx_.positions, ballots_, conflicts_, conflicts_wrk_,
						   agents_count, m.mesh.voxel_volume(), m.diffusion_time_step, m.substrates_count,
						   m.mesh.bounding_box_mins[0], m.mesh.bounding_box_mins[1], m.mesh.bounding_box_mins[2],
						   m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.voxel_shape[2], m.mesh.grid_shape[0],
						   m.mesh.grid_shape[1], m.mesh.grid_shape[2], m.mesh.dims, ctx_.substrates_queue);
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
		if (capacity_ != 0)
		{
			CUCH(cudaFree(numerators_));
			CUCH(cudaFree(denominators_));
			CUCH(cudaFree(factors_));

			CUCH(cudaFree(reduced_numerators_));
			CUCH(cudaFree(reduced_denominators_));
			CUCH(cudaFree(reduced_factors_));

			CUCH(cudaFree(conflicts_));
			CUCH(cudaFree(conflicts_wrk_));
		}

		capacity_ = get_agent_data(m).volumes.size();

		CUCH(cudaMalloc(&numerators_, capacity_ * m.substrates_count * sizeof(real_t)));
		CUCH(cudaMalloc(&denominators_, capacity_ * m.substrates_count * sizeof(real_t)));
		CUCH(cudaMalloc(&factors_, capacity_ * m.substrates_count * sizeof(real_t)));

		CUCH(cudaMalloc(&reduced_numerators_, capacity_ * m.substrates_count * sizeof(real_t)));
		CUCH(cudaMalloc(&reduced_denominators_, capacity_ * m.substrates_count * sizeof(real_t)));
		CUCH(cudaMalloc(&reduced_factors_, capacity_ * m.substrates_count * sizeof(real_t)));

		CUCH(cudaMalloc(&conflicts_, capacity_ * sizeof(index_t)));
		CUCH(cudaMalloc(&conflicts_wrk_, capacity_ * m.substrates_count * sizeof(index_t)));
	}
}

void cell_solver::initialize(microenvironment& m)
{
	capacity_ = 0;
	compute_internalized_substrates_ = m.compute_internalized_substrates;

	resize(m);

	CUCH(cudaMalloc(&ballots_, m.mesh.voxel_count() * sizeof(index_t)));
}

cell_solver::cell_solver(device_context& ctx) : cuda_solver(ctx) {}
