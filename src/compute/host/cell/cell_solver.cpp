#include "cell_solver.h"

#include "../../../microenvironment.h"
#include "../../../traits.h"

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

template <typename density_layout_t>
void simulate_internal(real_t* __restrict__ substrate_densities, real_t* __restrict__ internalized_substrates,
					   const real_t* __restrict__ secretion_rates, const real_t* __restrict__ uptake_rates,
					   const real_t* __restrict__ saturation_densities, const real_t* __restrict__ net_export_rates,
					   real_t cell_volume, real_t voxel_volume, real_t time_step, density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		const auto S_times_T = secretion_rates[s] * saturation_densities[s];
		const auto U_plus_S = uptake_rates[s] + secretion_rates[s];

		const auto int_c = cell_volume * time_step;

		internalized_substrates[s] -=
			(-int_c * U_plus_S * substrate_densities[s] + int_c * S_times_T) / (1 + int_c * U_plus_S)
			+ time_step * net_export_rates[s];

		const auto sub_c = int_c / voxel_volume;
		const auto net_c = (1 / voxel_volume) * time_step;

		(dens_l | noarr::get_at<'s'>(substrate_densities, s)) =
			((dens_l | noarr::get_at<'s'>(substrate_densities, s)) + sub_c * S_times_T) / (1 + sub_c * U_plus_S)
			+ net_c * net_export_rates[s];
	}
}

template <index_t dims>
void simulate_dim(agent_data& data)
{
	index_t voxel_volume = data.m.mesh.voxel_volume(); // expecting that voxel volume is the same for all voxels
	auto dens_l = layout_traits<dims>::construct_density_layout(data.m.substrates_count, data.m.mesh.grid_shape);

	for (std::size_t i = 0; i < data.agents.size(); i++)
	{
		simulate_internal(data.m.substrate_densities.get(),
						  data.internalized_substrates.data() + i * data.m.substrates_count,
						  data.secretion_rates.data() + i * data.m.substrates_count,
						  data.uptake_rates.data() + i * data.m.substrates_count,
						  data.saturation_densities.data() + i * data.m.substrates_count,
						  data.net_export_rates.data() + i * data.m.substrates_count, data.volumes[i], voxel_volume,
						  data.m.time_step, dens_l ^ fix_dims<dims>(data.positions.data() + i * dims, data.m.mesh));
	}
}

void cell_solver::simulate_secretion_and_uptake(agent_data& data)
{
	if (data.m.mesh.dims == 1)
		simulate_dim<1>(data);
	else if (data.m.mesh.dims == 2)
		simulate_dim<2>(data);
	else if (data.m.mesh.dims == 3)
		simulate_dim<3>(data);
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
	if (data.m.mesh.dims == 1)
		release_dim<1>(data, index);
	else if (data.m.mesh.dims == 2)
		release_dim<2>(data, index);
	else if (data.m.mesh.dims == 3)
		release_dim<3>(data, index);
}
