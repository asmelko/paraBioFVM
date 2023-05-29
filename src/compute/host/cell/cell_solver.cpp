#include "cell_solver.h"

#include "../../../microenvironment.h"
#include "../../../traits.h"

template <index_t dims>
auto fix_dims(const index_t* voxel_index)
{
	if constexpr (dims == 1)
		return noarr::fix<'x'>(voxel_index[0]);
	else if constexpr (dims == 2)
		return noarr::fix<'x'>(voxel_index[0]) ^ noarr::fix<'y'>(voxel_index[1]);
	else if constexpr (dims == 3)
		return noarr::fix<'x'>(voxel_index[0]) ^ noarr::fix<'y'>(voxel_index[1]) ^ noarr::fix<'z'>(voxel_index[2]);
}

template <typename density_layout_t>
void solve_internal(index_t substrates_count, real_t* __restrict__ substrate_densities,
					real_t* __restrict__ internalized_substrates, real_t* __restrict__ secretion_rates,
					real_t* __restrict__ uptake_rates, real_t* __restrict__ saturation_densities,
					real_t* __restrict__ net_export_rates, real_t cell_volume, real_t voxel_volume, real_t time_step,
					density_layout_t dens_l)
{
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
void solve_dim(agent_data& data)
{
	index_t voxel_volume = data.m.mesh.voxel_volume(); // expecting that voxel volume is the same for all voxels
	auto dens_l = layout_traits<dims>::construct_density_layout(data.m.substrates_count, data.m.mesh.grid_shape);

	for (std::size_t i = 0; i < data.agents.size(); i++)
	{
		solve_internal(data.m.substrates_count, data.m.substrate_densities.get(),
					   data.internalized_substrates.data() + i * data.m.substrates_count,
					   data.secretion_rates.data() + i * data.m.substrates_count,
					   data.uptake_rates.data() + i * data.m.substrates_count,
					   data.saturation_densities.data() + i * data.m.substrates_count,
					   data.net_export_rates.data() + i * data.m.substrates_count, data.volumes[i], voxel_volume,
					   data.m.time_step, dens_l ^ fix_dims<dims>(data.positions.data() + i * dims));
	}
}

void cell_solver::solve(agent_data& data)
{
	if (data.m.mesh.dims == 1)
		solve_dim<1>(data);
	else if (data.m.mesh.dims == 2)
		solve_dim<2>(data);
	else if (data.m.mesh.dims == 3)
		solve_dim<3>(data);
}
