#include "cell_solver.h"

#include <atomic>

#include "../../traits.h"
#include "microenvironment.h"
#include "types.h"

using namespace biofvm;

static constexpr index_t no_ballot = -1;

template <index_t dims>
auto fix_dims(const real_t* cell_position, const cartesian_mesh& m)
{
	if constexpr (dims == 1)
	{
		point_t<real_t, 3> pos = { cell_position[0], 0, 0 };
		point_t<index_t, 3> voxel_index = m.voxel_position(pos);
		return noarr::fix<'x'>(voxel_index[0]);
	}
	else if constexpr (dims == 2)
	{
		point_t<real_t, 3> pos = { cell_position[0], cell_position[1], 0 };
		point_t<index_t, 3> voxel_index = m.voxel_position(pos);
		return noarr::fix<'x'>(voxel_index[0]) ^ noarr::fix<'y'>(voxel_index[1]);
	}
	else if constexpr (dims == 3)
	{
		point_t<real_t, 3> pos = { cell_position[0], cell_position[1], cell_position[2] };
		point_t<index_t, 3> voxel_index = m.voxel_position(pos);
		return noarr::fix<'x'>(voxel_index[0]) ^ noarr::fix<'y'>(voxel_index[1]) ^ noarr::fix<'z'>(voxel_index[2]);
	}
}

template <index_t dims>
void clear_ballots(const real_t* __restrict__ cell_positions, std::atomic<index_t>* __restrict__ ballots,
				   std::atomic<real_t>* __restrict__ reduced_numerators,
				   std::atomic<real_t>* __restrict__ reduced_denominators,
				   std::atomic<real_t>* __restrict__ reduced_factors, index_t n, const cartesian_mesh& m,
				   index_t substrate_densities)
{
	const auto ballot_l = noarr::scalar<std::atomic<index_t>>() ^ typename layout_traits<dims>::grid_layout_t()
						  ^ layout_traits<dims>::set_grid_lengths(m.grid_shape);

#pragma omp for
	for (index_t i = 0; i < n; i++)
	{
		auto b_l = ballot_l ^ fix_dims<dims>(cell_positions + dims * i, m);

		auto& b = b_l | noarr::get_at(ballots);

		b.store(no_ballot, std::memory_order_relaxed);

		for (index_t s = 0; s < substrate_densities; s++)
		{
			reduced_numerators[i * substrate_densities + s].store(0, std::memory_order_relaxed);
			reduced_denominators[i * substrate_densities + s].store(0, std::memory_order_relaxed);
			reduced_factors[i * substrate_densities + s].store(0, std::memory_order_relaxed);
		}
	}
}

void compute_intermediates(real_t* __restrict__ numerators, real_t* __restrict__ denominators,
						   real_t* __restrict__ factors, const real_t* __restrict__ secretion_rates,
						   const real_t* __restrict__ uptake_rates, const real_t* __restrict__ saturation_densities,
						   const real_t* __restrict__ net_export_rates, const real_t* __restrict__ cell_volumes,
						   real_t voxel_volume, real_t time_step, index_t n, index_t substrates_count)
{
#pragma omp for
	for (index_t i = 0; i < n; i++)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			numerators[i * substrates_count + s] = secretion_rates[i * substrates_count + s]
												   * saturation_densities[i * substrates_count + s] * time_step
												   * cell_volumes[i] / voxel_volume;

			denominators[i * substrates_count + s] =
				(uptake_rates[i * substrates_count + s] + secretion_rates[i * substrates_count + s]) * time_step
				* cell_volumes[i] / voxel_volume;

			factors[i * substrates_count + s] = net_export_rates[i * substrates_count + s] * time_step / voxel_volume;
		}
	}
}

template <index_t dims>
void ballot_and_sum(std::atomic<real_t>* __restrict__ reduced_numerators,
					std::atomic<real_t>* __restrict__ reduced_denominators,
					std::atomic<real_t>* __restrict__ reduced_factors, const real_t* __restrict__ numerators,
					const real_t* __restrict__ denominators, const real_t* __restrict__ factors,
					const real_t* __restrict__ cell_positions, std::atomic<index_t>* __restrict__ ballots, index_t n,
					index_t substrates_count, const cartesian_mesh& m, std::atomic<bool>* __restrict__ is_conflict)
{
	const auto ballot_l = noarr::scalar<std::atomic<index_t>>() ^ typename layout_traits<dims>::grid_layout_t()
						  ^ layout_traits<dims>::set_grid_lengths(m.grid_shape);

#pragma omp for
	for (index_t i = 0; i < n; i++)
	{
		auto b_l = ballot_l ^ fix_dims<dims>(cell_positions + dims * i, m);

		auto& b = b_l | noarr::get_at(ballots);

		auto expected = no_ballot;
		bool success = b.compare_exchange_strong(expected, i, std::memory_order_acq_rel, std::memory_order_acquire);

		if (success)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				reduced_numerators[i * substrates_count + s].fetch_add(numerators[i * substrates_count + s],
																	   std::memory_order_relaxed);
				reduced_denominators[i * substrates_count + s].fetch_add(denominators[i * substrates_count + s] + 1,
																		 std::memory_order_relaxed);
				reduced_factors[i * substrates_count + s].fetch_add(factors[i * substrates_count + s],
																	std::memory_order_relaxed);
			}
		}
		else
		{
			is_conflict[0].store(true, std::memory_order_relaxed);

			for (index_t s = 0; s < substrates_count; s++)
			{
				reduced_numerators[expected * substrates_count + s].fetch_add(numerators[i * substrates_count + s],
																			  std::memory_order_relaxed);
				reduced_denominators[expected * substrates_count + s].fetch_add(denominators[i * substrates_count + s],
																				std::memory_order_relaxed);
				reduced_factors[expected * substrates_count + s].fetch_add(factors[i * substrates_count + s],
																		   std::memory_order_relaxed);
			}
		}
	}
}

template <typename density_layout_t>
void compute_internalized(real_t* __restrict__ internalized_substrates, const real_t* __restrict__ substrate_densities,
						  const real_t* __restrict__ numerator, const real_t* __restrict__ denominator,
						  const real_t* __restrict__ factor, real_t voxel_volume, density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		internalized_substrates[s] -=
			voxel_volume
			* (((-denominator[s]) * (dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s])
				   / (1 + denominator[s])
			   + factor[s]);
	}
}

template <typename density_layout_t>
void compute_densities(real_t* __restrict__ substrate_densities, const std::atomic<real_t>* __restrict__ numerator,
					   const std::atomic<real_t>* __restrict__ denominator,
					   const std::atomic<real_t>* __restrict__ factor, bool has_ballot, density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	if (has_ballot)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'s'>(substrate_densities, s)) =
				((dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s].load(std::memory_order_relaxed))
					/ denominator[s].load(std::memory_order_relaxed)
				+ factor[s].load(std::memory_order_relaxed);
		}
	}
}

template <typename density_layout_t>
void compute_fused(real_t* __restrict__ substrate_densities, real_t* __restrict__ internalized_substrates,
				   const std::atomic<real_t>* __restrict__ numerator,
				   const std::atomic<real_t>* __restrict__ denominator, const std::atomic<real_t>* __restrict__ factor,
				   index_t voxel_volume, density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		internalized_substrates[s] -= voxel_volume
									  * (((1 - denominator[s].load(std::memory_order_relaxed))
											  * (dens_l | noarr::get_at<'s'>(substrate_densities, s))
										  + numerator[s].load(std::memory_order_relaxed))
											 / (denominator[s].load(std::memory_order_relaxed))
										 + factor[s].load(std::memory_order_relaxed));

		(dens_l | noarr::get_at<'s'>(substrate_densities, s)) =
			((dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s].load(std::memory_order_relaxed))
				/ denominator[s].load(std::memory_order_relaxed)
			+ factor[s].load(std::memory_order_relaxed);
	}
}

template <index_t dims>
void compute_result(agent_data& data, const std::atomic<real_t>* reduced_numerators,
					const std::atomic<real_t>* reduced_denominators, const std::atomic<real_t>* reduced_factors,
					const real_t* numerators, const real_t* denominators, const real_t* factors,
					const std::atomic<index_t>* ballots, bool with_internalized, bool is_conflict)
{
	index_t voxel_volume = data.m.mesh.voxel_volume(); // expecting that voxel volume is the same for all voxels
	auto dens_l = layout_traits<dims>::construct_density_layout(data.m.substrates_count, data.m.mesh.grid_shape);
	auto ballot_l = noarr::scalar<index_t>() ^ typename layout_traits<dims>::grid_layout_t()
					^ layout_traits<dims>::set_grid_lengths(data.m.mesh.grid_shape);

	if (with_internalized && !is_conflict)
	{
#pragma omp for
		for (index_t i = 0; i < data.agents_count; i++)
		{
			auto fixed_dims = fix_dims<dims>(data.positions.data() + i * dims, data.m.mesh);
			compute_fused(
				data.m.substrate_densities.get(), data.internalized_substrates.data() + i * data.m.substrates_count,
				reduced_numerators + i * data.m.substrates_count, reduced_denominators + i * data.m.substrates_count,
				reduced_factors + i * data.m.substrates_count, voxel_volume, dens_l ^ fixed_dims);
		}

		return;
	}

	if (with_internalized)
	{
#pragma omp for
		for (index_t i = 0; i < data.agents_count; i++)
		{
			auto fixed_dims = fix_dims<dims>(data.positions.data() + i * dims, data.m.mesh);
			compute_internalized(data.internalized_substrates.data() + i * data.m.substrates_count,
								 data.m.substrate_densities.get(), numerators + i * data.m.substrates_count,
								 denominators + i * data.m.substrates_count, factors + i * data.m.substrates_count,
								 voxel_volume, dens_l ^ fixed_dims);
		}
	}

#pragma omp for
	for (index_t i = 0; i < data.agents_count; i++)
	{
		auto fixed_dims = fix_dims<dims>(data.positions.data() + i * dims, data.m.mesh);
		auto ballot = (ballot_l ^ fixed_dims) | noarr::get_at(ballots);
		compute_densities(data.m.substrate_densities.get(), reduced_numerators + i * data.m.substrates_count,
						  reduced_denominators + i * data.m.substrates_count,
						  reduced_factors + i * data.m.substrates_count, ballot == i, dens_l ^ fixed_dims);
	}
}

template <index_t dims>
void simulate(agent_data& data, std::atomic<real_t>* reduced_numerators, std::atomic<real_t>* reduced_denominators,
			  std::atomic<real_t>* reduced_factors, real_t* numerators, real_t* denominators, real_t* factors,
			  std::atomic<index_t>* ballots, bool recompute, bool with_internalized,
			  std::atomic<bool>* __restrict__ is_conflict)
{
	if (recompute)
	{
		compute_intermediates(numerators, denominators, factors, data.secretion_rates.data(), data.uptake_rates.data(),
							  data.saturation_densities.data(), data.net_export_rates.data(), data.volumes.data(),
							  data.m.mesh.voxel_volume(), data.m.diffusion_time_step, data.agents_count,
							  data.m.substrates_count);

		clear_ballots<dims>(data.positions.data(), ballots, reduced_numerators, reduced_denominators, reduced_factors,
							data.agents_count, data.m.mesh, data.m.substrates_count);

		ballot_and_sum<dims>(reduced_numerators, reduced_denominators, reduced_factors, numerators, denominators,
							 factors, data.positions.data(), ballots, data.agents_count, data.m.substrates_count,
							 data.m.mesh, is_conflict);
	}

	compute_result<dims>(data, reduced_numerators, reduced_denominators, reduced_factors, numerators, denominators,
						 factors, ballots, with_internalized, is_conflict[0].load(std::memory_order_relaxed));
}

void cell_solver::simulate_secretion_and_uptake(microenvironment& m, bool recompute)
{
#pragma omp single
	if (recompute)
	{
		resize(m);
		is_conflict_.store(false, std::memory_order_relaxed);
	}

	if (m.mesh.dims == 1)
	{
		simulate<1>(m.agents->get_agent_data(), reduced_numerators_.get(), reduced_denominators_.get(),
					reduced_factors_.get(), numerators_.data(), denominators_.data(), factors_.data(), ballots_.get(),
					recompute, compute_internalized_substrates_, &is_conflict_);
	}
	else if (m.mesh.dims == 2)
	{
		simulate<2>(m.agents->get_agent_data(), reduced_numerators_.get(), reduced_denominators_.get(),
					reduced_factors_.get(), numerators_.data(), denominators_.data(), factors_.data(), ballots_.get(),
					recompute, compute_internalized_substrates_, &is_conflict_);
	}
	else if (m.mesh.dims == 3)
	{
		simulate<3>(m.agents->get_agent_data(), reduced_numerators_.get(), reduced_denominators_.get(),
					reduced_factors_.get(), numerators_.data(), denominators_.data(), factors_.data(), ballots_.get(),
					recompute, compute_internalized_substrates_, &is_conflict_);
	}
}

template <typename density_layout_t>
void release_internal(real_t* __restrict__ substrate_densities, real_t* __restrict__ internalized_substrates,
					  const real_t* __restrict__ fraction_released_at_death, real_t voxel_volume,
					  density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		(dens_l | noarr::get_at<'s'>(substrate_densities, s)) +=
			internalized_substrates[s] * fraction_released_at_death[s] / voxel_volume;

		internalized_substrates[s] = 0;
	}
}

template <index_t dims>
void release_dim(agent_data& data, index_t index)
{
	index_t voxel_volume = data.m.mesh.voxel_volume(); // expecting that voxel volume is the same for all voxels
	auto dens_l = layout_traits<dims>::construct_density_layout(data.m.substrates_count, data.m.mesh.grid_shape)
				  ^ fix_dims<dims>(data.positions.data() + index * dims, data.m.mesh);

	release_internal(data.m.substrate_densities.get(),
					 data.internalized_substrates.data() + index * data.m.substrates_count,
					 data.fraction_released_at_death.data() + index * data.m.substrates_count, voxel_volume, dens_l);
}

void cell_solver::release_internalized_substrates(microenvironment& m, index_t index)
{
	if (!compute_internalized_substrates_)
		return;

	if (m.mesh.dims == 1)
		release_dim<1>(m.agents->get_agent_data(), index);
	else if (m.mesh.dims == 2)
		release_dim<2>(m.agents->get_agent_data(), index);
	else if (m.mesh.dims == 3)
		release_dim<3>(m.agents->get_agent_data(), index);
}

void cell_solver::resize(const microenvironment& m)
{
	auto prev_capacity = numerators_.capacity();

	numerators_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);
	denominators_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);
	factors_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);

	auto new_capacity = numerators_.capacity();

	if (new_capacity != prev_capacity)
	{
		reduced_numerators_ = std::make_unique<std::atomic<real_t>[]>(new_capacity);
		reduced_denominators_ = std::make_unique<std::atomic<real_t>[]>(new_capacity);
		reduced_factors_ = std::make_unique<std::atomic<real_t>[]>(new_capacity);
	}
}

void cell_solver::initialize(const microenvironment& m)
{
	compute_internalized_substrates_ = m.compute_internalized_substrates;

	resize(m);

	ballots_ = std::make_unique<std::atomic<index_t>[]>(m.mesh.voxel_count());
}
