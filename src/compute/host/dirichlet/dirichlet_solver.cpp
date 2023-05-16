#include "dirichlet_solver.h"

#include "../../../traits.h"

template <typename voxel_layout_t>
void apply_condition(real_t* __restrict__ substrate_density, const real_t* __restrict__ dirichlet_values,
					 const bool* __restrict__ dirichlet_conditions, index_t substrate_count,
					 const voxel_layout_t voxel_l)
{
	for (index_t s = 0; s < substrate_count; ++s)
	{
		if (dirichlet_conditions[s])
			(voxel_l | noarr::get_at<'s'>(substrate_density, s)) = dirichlet_values[s];
	}
}

void solve_1D(real_t* __restrict__ substrate_densities, const index_t* __restrict__ dirichlet_voxels,
			  const real_t* __restrict__ dirichlet_values, const bool* __restrict__ dirichlet_conditions,
			  index_t substrates_count, index_t dirichlet_voxels_count, const point_t<index_t, 3>& grid_shape)
{
	auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, grid_shape);

	for (index_t voxel = 0; voxel < dirichlet_voxels_count; ++voxel)
	{
		index_t voxel_index = dirichlet_voxels[voxel];
		auto voxel_l = dens_l ^ noarr::fix<'x'>(voxel_index);

		apply_condition(substrate_densities, dirichlet_values + voxel * substrates_count,
						dirichlet_conditions + voxel * substrates_count, substrates_count, voxel_l);
	}
}

void solve_2D(real_t* __restrict__ substrate_densities, const index_t* __restrict__ dirichlet_voxels,
			  const real_t* __restrict__ dirichlet_values, const bool* __restrict__ dirichlet_conditions,
			  index_t substrates_count, index_t dirichlet_voxels_count, const point_t<index_t, 3>& grid_shape)
{
	auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, grid_shape);

	for (index_t voxel = 0; voxel < dirichlet_voxels_count; ++voxel)
	{
		const index_t* voxel_index = dirichlet_voxels + 2 * voxel;
		auto voxel_l = dens_l ^ noarr::fix<'x'>(voxel_index[0]) ^ noarr::fix<'y'>(voxel_index[1]);

		apply_condition(substrate_densities, dirichlet_values + voxel * substrates_count,
						dirichlet_conditions + voxel * substrates_count, substrates_count, voxel_l);
	}
}

void solve_3D(real_t* __restrict__ substrate_densities, const index_t* __restrict__ dirichlet_voxels,
			  const real_t* __restrict__ dirichlet_values, const bool* __restrict__ dirichlet_conditions,
			  index_t substrates_count, index_t dirichlet_voxels_count, const point_t<index_t, 3>& grid_shape)
{
	auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, grid_shape);

	for (index_t voxel = 0; voxel < dirichlet_voxels_count; ++voxel)
	{
		const index_t* voxel_index = dirichlet_voxels + 3 * voxel;
		auto voxel_l = dens_l ^ noarr::fix<'x'>(voxel_index[0]) ^ noarr::fix<'y'>(voxel_index[1])
					   ^ noarr::fix<'z'>(voxel_index[2]);

		apply_condition(substrate_densities, dirichlet_values + voxel * substrates_count,
						dirichlet_conditions + voxel * substrates_count, substrates_count, voxel_l);
	}
}

void dirichlet_solver::solve(microenvironment& m)
{
	if (m.dirichlet_voxels_count == 0)
		return;

	if (m.mesh.dims == 1)
		solve_1D(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
				 m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count, m.mesh.grid_shape);
	else if (m.mesh.dims == 2)
		solve_2D(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
				 m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count, m.mesh.grid_shape);
	else if (m.mesh.dims == 3)
		solve_3D(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
				 m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count, m.mesh.grid_shape);
}
