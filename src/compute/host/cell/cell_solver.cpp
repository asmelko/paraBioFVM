#include "cell_solver.h"

#include "../../../microenvironment.h"
#include "../../../traits.h"

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
void clear_ballots(const real_t* __restrict__ cell_positions, index_t* __restrict__ ballots, index_t n,
				   const cartesian_mesh& m)
{
	auto ballot_l = noarr::scalar<index_t>() ^ typename layout_traits<dims>::grid_layout_t()
					^ layout_traits<dims>::set_grid_lengths(m.grid_shape);

	for (index_t i = 0; i < n; i++)
	{
		auto b_l = ballot_l ^ fix_dims<dims>(cell_positions + dims * n, m);

		index_t& b = b_l | noarr::get_at(ballots);

		b = no_ballot;
	}
}

void compute_intermediates(real_t* __restrict__ numerators, real_t* __restrict__ denominators,
						   const real_t* __restrict__ secretion_rates, const real_t* __restrict__ uptake_rates,
						   const real_t* __restrict__ saturation_densities, const real_t* __restrict__ cell_volumes,
						   real_t voxel_volume, real_t time_step, index_t n, index_t substrates_count)
{
	for (index_t i = 0; i < n * substrates_count; i++) // TODO: vectorizable by removing cell_volumes
	{
		numerators[i] = secretion_rates[i] * saturation_densities[i] * time_step * cell_volumes[i / substrates_count]
						/ voxel_volume;

		denominators[i] =
			(uptake_rates[i] + secretion_rates[i]) * time_step * cell_volumes[i / substrates_count] / voxel_volume;
	}
}

template <index_t dims>
void ballot_and_sum(real_t* __restrict__ numerators, real_t* __restrict__ denominators,
					const real_t* __restrict__ cell_positions, index_t* __restrict__ ballots, index_t n,
					index_t substrates_count, const cartesian_mesh& m)
{
	auto ballot_l = noarr::scalar<index_t>() ^ typename layout_traits<dims>::grid_layout_t()
					^ layout_traits<dims>::set_grid_lengths(m.grid_shape);

	for (index_t i = 0; i < n; i++)
	{
		auto b_l = ballot_l ^ fix_dims<dims>(cell_positions + dims * n, m);

		index_t& b = b_l | noarr::get_at(ballots);

		if (b == no_ballot)
		{
			b = i;

			for (index_t s = 0; s < substrates_count; s++)
			{
				denominators[i * substrates_count + s] += 1;
			}
		}
		else
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				numerators[b * substrates_count + s] += numerators[i * substrates_count + s];
				denominators[b * substrates_count + s] += denominators[i * substrates_count + s];
			}
		}
	}
}

template <typename density_layout_t>
void compute_with_internalized(real_t* __restrict__ substrate_densities, real_t* __restrict__ internalized_substrates,
							   const real_t* __restrict__ net_export_rates, const real_t* __restrict__ numerator,
							   const real_t* __restrict__ denominator, real_t voxel_volume, real_t time_step,
							   bool has_ballot, density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	if (has_ballot)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			internalized_substrates[s] -=
				voxel_volume
					* ((1 - denominator[s]) * (dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s])
					/ denominator[s]
				+ time_step * net_export_rates[s];

			const auto net_c = time_step / voxel_volume;

			(dens_l | noarr::get_at<'s'>(substrate_densities, s)) =
				((dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s]) / denominator[s]
				+ net_c * net_export_rates[s];
		}
	}
	else
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			internalized_substrates[s] -=
				voxel_volume
					* ((1 - denominator[s]) * (dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s])
					/ denominator[s]
				+ time_step * net_export_rates[s];
		}
	}
}

template <typename density_layout_t>
void compute_without_internalized(real_t* __restrict__ substrate_densities, const real_t* __restrict__ net_export_rates,
								  const real_t* __restrict__ numerator, const real_t* __restrict__ denominator,
								  real_t voxel_volume, real_t time_step, bool has_ballot, density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	if (has_ballot)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			const auto net_c = time_step / voxel_volume;

			(dens_l | noarr::get_at<'s'>(substrate_densities, s)) =
				((dens_l | noarr::get_at<'s'>(substrate_densities, s)) + numerator[s]) / denominator[s]
				+ net_c * net_export_rates[s];
		}
	}
}

template <index_t dims>
void compute_result(agent_data& data, const real_t* numerators, const real_t* denominators, const index_t* ballots,
					bool with_internalized)
{
	index_t voxel_volume = data.m.mesh.voxel_volume(); // expecting that voxel volume is the same for all voxels
	auto dens_l = layout_traits<dims>::construct_density_layout(data.m.substrates_count, data.m.mesh.grid_shape);
	auto ballot_l = noarr::scalar<index_t>() ^ typename layout_traits<dims>::grid_layout_t()
					^ layout_traits<dims>::set_grid_lengths(data.m.mesh.grid_shape);

	if (with_internalized)
	{
		for (index_t i = 0; i < data.agents_count; i++)
		{
			auto fixed_dims = fix_dims<dims>(data.positions.data() + i * dims, data.m.mesh);
			auto ballot = (ballot_l ^ fixed_dims) | noarr::get_at(ballots);
			compute_with_internalized(
				data.m.substrate_densities.get(), data.internalized_substrates.data() + i * data.m.substrates_count,
				data.net_export_rates.data() + i * data.m.substrates_count, numerators + i * data.m.substrates_count,
				denominators + i * data.m.substrates_count, voxel_volume, data.m.time_step, ballot == i,
				dens_l ^ fixed_dims);
		}
	}
	else
	{
		for (index_t i = 0; i < data.agents_count; i++)
		{
			auto fixed_dims = fix_dims<dims>(data.positions.data() + i * dims, data.m.mesh);
			auto ballot = (ballot_l ^ fixed_dims) | noarr::get_at(ballots);
			compute_without_internalized(
				data.m.substrate_densities.get(), data.net_export_rates.data() + i * data.m.substrates_count,
				numerators + i * data.m.substrates_count, denominators + i * data.m.substrates_count, voxel_volume,
				data.m.time_step, ballot == i, dens_l ^ fixed_dims);
		}
	}
}

template <index_t dims>
void simulate(agent_data& data, real_t* numerators, real_t* denominators, index_t* ballots, bool recompute,
			  bool with_internalized)
{
	if (recompute)
	{
		compute_intermediates(numerators, denominators, data.secretion_rates.data(), data.uptake_rates.data(),
							  data.saturation_densities.data(), data.volumes.data(), data.m.mesh.voxel_volume(),
							  data.m.time_step, data.agents_count, data.m.substrates_count);

		clear_ballots<dims>(data.positions.data(), ballots, data.agents_count, data.m.mesh);

		ballot_and_sum<dims>(numerators, denominators, data.positions.data(), ballots, data.agents_count,
							 data.m.substrates_count, data.m.mesh);
	}

	compute_result<dims>(data, numerators, denominators, ballots, with_internalized);
}

void cell_solver::simulate_secretion_and_uptake(agent_data& data, bool recompute)
{
	if (recompute)
		resize(data);

	if (data.m.mesh.dims == 1)
	{
		simulate<1>(data, numerators_.data(), denominators_.data(), ballots_.get(), recompute,
					compute_internalized_substrates_);
	}
	else if (data.m.mesh.dims == 2)
	{
		simulate<2>(data, numerators_.data(), denominators_.data(), ballots_.get(), recompute,
					compute_internalized_substrates_);
	}
	else if (data.m.mesh.dims == 3)
	{
		simulate<3>(data, numerators_.data(), denominators_.data(), ballots_.get(), recompute,
					compute_internalized_substrates_);
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

void cell_solver::release_internalized_substrates(agent_data& data, index_t index)
{
	if (!compute_internalized_substrates_)
		return;

	if (data.m.mesh.dims == 1)
		release_dim<1>(data, index);
	else if (data.m.mesh.dims == 2)
		release_dim<2>(data, index);
	else if (data.m.mesh.dims == 3)
		release_dim<3>(data, index);
}

void cell_solver::resize(const agent_data& data)
{
	numerators_.resize(data.m.substrates_count * data.agents_count);
	denominators_.resize(data.m.substrates_count * data.agents_count);
}

void cell_solver::initialize(agent_data& data, bool compute_internalized_substrates)
{
	compute_internalized_substrates_ = compute_internalized_substrates;

	resize(data);

	ballots_ = std::make_unique<index_t[]>(data.m.mesh.voxel_count());
}
