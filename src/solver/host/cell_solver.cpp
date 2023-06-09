#include "cell_solver.h"

#include "../../traits.h"
#include "microenvironment.h"

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
void clear_ballots(const real_t* __restrict__ cell_positions, index_t* __restrict__ ballots,
				   real_t* __restrict__ reduced_numerators, real_t* __restrict__ reduced_denominators,
				   real_t* __restrict__ reduced_factors, index_t n, const cartesian_mesh& m,
				   index_t substrate_densities)
{
	auto ballot_l = noarr::scalar<index_t>() ^ typename layout_traits<dims>::grid_layout_t()
					^ layout_traits<dims>::set_grid_lengths(m.grid_shape);

	for (index_t i = 0; i < n; i++)
	{
		auto b_l = ballot_l ^ fix_dims<dims>(cell_positions + dims * i, m);

		index_t& b = b_l | noarr::get_at(ballots);

		b = no_ballot;

		for (index_t s = 0; s < substrate_densities; s++)
		{
			reduced_numerators[i * substrate_densities + s] = 0;
			reduced_denominators[i * substrate_densities + s] = 0;
			reduced_factors[i * substrate_densities + s] = 0;
		}
	}
}

void compute_intermediates(real_t* __restrict__ numerators, real_t* __restrict__ denominators,
						   real_t* __restrict__ factors, const real_t* __restrict__ secretion_rates,
						   const real_t* __restrict__ uptake_rates, const real_t* __restrict__ saturation_densities,
						   const real_t* __restrict__ net_export_rates, const real_t* __restrict__ cell_volumes,
						   real_t voxel_volume, real_t time_step, index_t n, index_t substrates_count)
{
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
void ballot_and_sum(real_t* __restrict__ reduced_numerators, real_t* __restrict__ reduced_denominators,
					real_t* __restrict__ reduced_factors, const real_t* __restrict__ numerators,
					const real_t* __restrict__ denominators, const real_t* __restrict__ factors,
					const real_t* __restrict__ cell_positions, index_t* __restrict__ ballots, index_t n,
					index_t substrates_count, const cartesian_mesh& m, bool* __restrict__ is_conflict)
{
	auto ballot_l = noarr::scalar<index_t>() ^ typename layout_traits<dims>::grid_layout_t()
					^ layout_traits<dims>::set_grid_lengths(m.grid_shape);

	for (index_t i = 0; i < n; i++)
	{
		auto b_l = ballot_l ^ fix_dims<dims>(cell_positions + dims * i, m);

		index_t& b = b_l | noarr::get_at(ballots);

		if (b == no_ballot)
		{
			b = i;

			for (index_t s = 0; s < substrates_count; s++)
			{
				reduced_numerators[i * substrates_count + s] += numerators[i * substrates_count + s];
				reduced_denominators[i * substrates_count + s] += denominators[i * substrates_count + s] + 1;
				reduced_factors[i * substrates_count + s] += factors[i * substrates_count + s];
			}
		}
		else
		{
			is_conflict[0] = true;

			for (index_t s = 0; s < substrates_count; s++)
			{
				reduced_numerators[b * substrates_count + s] += numerators[i * substrates_count + s];
				reduced_denominators[b * substrates_count + s] += denominators[i * substrates_count + s];
				reduced_factors[b * substrates_count + s] += factors[i * substrates_count + s];
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
void compute_densities(real_t* __restrict__ substrate_densities, const real_t* __restrict__ numerator,
					   const real_t* __restrict__ denominator, const real_t* __restrict__ factor, bool has_ballot,
					   density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	if (has_ballot)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'s'>(substrate_densities, s)) =
				((dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s]) / denominator[s] + factor[s];
		}
	}
}

template <typename density_layout_t>
void compute_fused(real_t* __restrict__ substrate_densities, real_t* __restrict__ internalized_substrates,
				   const real_t* __restrict__ numerator, const real_t* __restrict__ denominator,
				   const real_t* __restrict__ factor, index_t voxel_volume, density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		internalized_substrates[s] -=
			voxel_volume
			* (((1 - denominator[s]) * (dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s])
				   / (denominator[s])
			   + factor[s]);

		(dens_l | noarr::get_at<'s'>(substrate_densities, s)) =
			((dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s]) / denominator[s] + factor[s];
	}
}

template <index_t dims>
void compute_result(agent_data& data, const real_t* reduced_numerators, const real_t* reduced_denominators,
					const real_t* reduced_factors, const real_t* numerators, const real_t* denominators,
					const real_t* factors, const index_t* ballots, bool with_internalized, bool is_conflict)
{
	index_t voxel_volume = data.m.mesh.voxel_volume(); // expecting that voxel volume is the same for all voxels
	auto dens_l = layout_traits<dims>::construct_density_layout(data.m.substrates_count, data.m.mesh.grid_shape);
	auto ballot_l = noarr::scalar<index_t>() ^ typename layout_traits<dims>::grid_layout_t()
					^ layout_traits<dims>::set_grid_lengths(data.m.mesh.grid_shape);

	if (with_internalized && !is_conflict)
	{
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
		for (index_t i = 0; i < data.agents_count; i++)
		{
			auto fixed_dims = fix_dims<dims>(data.positions.data() + i * dims, data.m.mesh);
			compute_internalized(data.internalized_substrates.data() + i * data.m.substrates_count,
								 data.m.substrate_densities.get(), numerators + i * data.m.substrates_count,
								 denominators + i * data.m.substrates_count, factors + i * data.m.substrates_count,
								 voxel_volume, dens_l ^ fixed_dims);
		}
	}

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
void simulate(agent_data& data, real_t* reduced_numerators, real_t* reduced_denominators, real_t* reduced_factors,
			  real_t* numerators, real_t* denominators, real_t* factors, index_t* ballots, bool recompute,
			  bool with_internalized, bool* __restrict__ is_conflict)
{
	if (recompute)
	{
		compute_intermediates(numerators, denominators, factors, data.secretion_rates.data(), data.uptake_rates.data(),
							  data.saturation_densities.data(), data.net_export_rates.data(), data.volumes.data(),
							  data.m.mesh.voxel_volume(), data.m.time_step, data.agents_count, data.m.substrates_count);

		clear_ballots<dims>(data.positions.data(), ballots, reduced_numerators, reduced_denominators, reduced_factors,
							data.agents_count, data.m.mesh, data.m.substrates_count);

		ballot_and_sum<dims>(reduced_numerators, reduced_denominators, reduced_factors, numerators, denominators,
							 factors, data.positions.data(), ballots, data.agents_count, data.m.substrates_count,
							 data.m.mesh, is_conflict);
	}

	compute_result<dims>(data, reduced_numerators, reduced_denominators, reduced_factors, numerators, denominators,
						 factors, ballots, with_internalized, *is_conflict);
}

void cell_solver::simulate_secretion_and_uptake(microenvironment& m, bool recompute)
{
	if (recompute)
	{
		resize(m);
		is_conflict_ = false;
	}

	if (m.mesh.dims == 1)
	{
		simulate<1>(m.agents->get_agent_data(), reduced_numerators_.data(), reduced_denominators_.data(),
					reduced_factors_.data(), numerators_.data(), denominators_.data(), factors_.data(), ballots_.get(),
					recompute, compute_internalized_substrates_, &is_conflict_);
	}
	else if (m.mesh.dims == 2)
	{
		simulate<2>(m.agents->get_agent_data(), reduced_numerators_.data(), reduced_denominators_.data(),
					reduced_factors_.data(), numerators_.data(), denominators_.data(), factors_.data(), ballots_.get(),
					recompute, compute_internalized_substrates_, &is_conflict_);
	}
	else if (m.mesh.dims == 3)
	{
		simulate<3>(m.agents->get_agent_data(), reduced_numerators_.data(), reduced_denominators_.data(),
					reduced_factors_.data(), numerators_.data(), denominators_.data(), factors_.data(), ballots_.get(),
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
	numerators_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);
	denominators_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);
	factors_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);

	reduced_numerators_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);
	reduced_denominators_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);
	reduced_factors_.resize(m.substrates_count * m.agents->get_agent_data().agents_count);
}

void cell_solver::initialize(const microenvironment& m)
{
	compute_internalized_substrates_ = m.compute_internalized_substrates;

	resize(m);

	ballots_ = std::make_unique<index_t[]>(m.mesh.voxel_count());
}
