#include "dirichlet_solver.h"

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

template <index_t dims>
void solve_internal(real_t* __restrict__ substrate_densities, const index_t* __restrict__ dirichlet_voxels,
					const real_t* __restrict__ dirichlet_values, const bool* __restrict__ dirichlet_conditions,
					index_t substrates_count, index_t dirichlet_voxels_count, const point_t<index_t, 3>& grid_shape)
{
	if (dirichlet_voxels_count == 0)
		return;

	auto dens_l = layout_traits<dims>::construct_density_layout(substrates_count, grid_shape);

	for (index_t voxel = 0; voxel < dirichlet_voxels_count; ++voxel)
	{
		auto voxel_l = dens_l ^ fix_dims<dims>(dirichlet_voxels + dims * voxel);

		for (index_t s = 0; s < substrates_count; ++s)
		{
			if (dirichlet_conditions[voxel * substrates_count + s])
				(voxel_l | noarr::get_at<'s'>(substrate_densities, s)) = dirichlet_values[voxel * substrates_count + s];
		}
	}
}

void dirichlet_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 1)
		solve_internal<1>(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
						  m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count,
						  m.mesh.grid_shape);
	else if (m.mesh.dims == 2)
		solve_internal<2>(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
						  m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count,
						  m.mesh.grid_shape);
	else if (m.mesh.dims == 3)
		solve_internal<3>(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
						  m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count,
						  m.mesh.grid_shape);
}

void dirichlet_solver::solve_1d(microenvironment& m)
{
	solve_internal<1>(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
					  m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count, m.mesh.grid_shape);
}

void dirichlet_solver::solve_2d(microenvironment& m)
{
	solve_internal<2>(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
					  m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count, m.mesh.grid_shape);
}

void dirichlet_solver::solve_3d(microenvironment& m)
{
	solve_internal<3>(m.substrate_densities.get(), m.dirichlet_voxels.get(), m.dirichlet_values.get(),
					  m.dirichlet_conditions.get(), m.substrates_count, m.dirichlet_voxels_count, m.mesh.grid_shape);
}
