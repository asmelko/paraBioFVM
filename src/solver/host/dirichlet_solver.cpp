#include "dirichlet_solver.h"

#include <noarr/structures/extra/traverser.hpp>

#include "../../traits.h"
#include "omp_utils.h"

using namespace biofvm;
using namespace solvers::host;

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
void solve_interior(real_t* __restrict__ substrate_densities, const index_t* __restrict__ dirichlet_voxels,
					const real_t* __restrict__ dirichlet_values, const bool* __restrict__ dirichlet_conditions,
					index_t substrates_count, index_t dirichlet_voxels_count, const point_t<index_t, 3>& grid_shape)
{
	if (dirichlet_voxels_count == 0)
		return;

	auto dens_l = layout_traits<dims>::construct_density_layout(substrates_count, grid_shape);

#pragma omp for
	for (index_t voxel_idx = 0; voxel_idx < dirichlet_voxels_count; ++voxel_idx)
	{
		auto subs_l = dens_l ^ fix_dims<dims>(dirichlet_voxels + dims * voxel_idx);

		for (index_t s = 0; s < substrates_count; ++s)
		{
			if (dirichlet_conditions[voxel_idx * substrates_count + s])
				(subs_l | noarr::get_at<'s'>(substrate_densities, s)) =
					dirichlet_values[voxel_idx * substrates_count + s];
		}
	}
}

template <typename density_layout_t>
void solve_boundary(real_t* __restrict__ substrate_densities, const real_t* __restrict__ dirichlet_values,
					const bool* __restrict__ dirichlet_conditions, const density_layout_t dens_l)
{
	if (dirichlet_values == nullptr)
		return;

	omp_trav_for_each(noarr::traverser(dens_l), [=](auto state) {
		auto s = noarr::get_index<'s'>(state);

		if (dirichlet_conditions[s])
			(dens_l | noarr::get_at(substrate_densities, state)) = dirichlet_values[s];
	});
}

template <index_t dims>
void solve_boundaries(microenvironment& m);

template <>
void solve_boundaries<1>(microenvironment& m)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	solve_boundary(m.substrate_densities.get(), m.dirichlet_min_boundary_values[0].get(),
				   m.dirichlet_min_boundary_conditions[0].get(), dens_l ^ noarr::fix<'x'>(0));
	solve_boundary(m.substrate_densities.get(), m.dirichlet_max_boundary_values[0].get(),
				   m.dirichlet_max_boundary_conditions[0].get(), dens_l ^ noarr::fix<'x'>(m.mesh.grid_shape[0] - 1));
}

template <>
void solve_boundaries<2>(microenvironment& m)
{
	auto dens_l = layout_traits<2>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	solve_boundary(m.substrate_densities.get(), m.dirichlet_min_boundary_values[0].get(),
				   m.dirichlet_min_boundary_conditions[0].get(), dens_l ^ noarr::fix<'x'>(0));
	solve_boundary(m.substrate_densities.get(), m.dirichlet_max_boundary_values[0].get(),
				   m.dirichlet_max_boundary_conditions[0].get(), dens_l ^ noarr::fix<'x'>(m.mesh.grid_shape[0] - 1));

	solve_boundary(m.substrate_densities.get(), m.dirichlet_min_boundary_values[1].get(),
				   m.dirichlet_min_boundary_conditions[1].get(), dens_l ^ noarr::fix<'y'>(0));
	solve_boundary(m.substrate_densities.get(), m.dirichlet_max_boundary_values[1].get(),
				   m.dirichlet_max_boundary_conditions[1].get(), dens_l ^ noarr::fix<'y'>(m.mesh.grid_shape[1] - 1));
}

template <>
void solve_boundaries<3>(microenvironment& m)
{
	auto dens_l = layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	solve_boundary(m.substrate_densities.get(), m.dirichlet_min_boundary_values[0].get(),
				   m.dirichlet_min_boundary_conditions[0].get(), dens_l ^ noarr::fix<'x'>(noarr::lit<0>));
	solve_boundary(m.substrate_densities.get(), m.dirichlet_max_boundary_values[0].get(),
				   m.dirichlet_max_boundary_conditions[0].get(), dens_l ^ noarr::fix<'x'>(m.mesh.grid_shape[0] - 1));

	solve_boundary(m.substrate_densities.get(), m.dirichlet_min_boundary_values[1].get(),
				   m.dirichlet_min_boundary_conditions[1].get(), dens_l ^ noarr::fix<'y'>(noarr::lit<0>));
	solve_boundary(m.substrate_densities.get(), m.dirichlet_max_boundary_values[1].get(),
				   m.dirichlet_max_boundary_conditions[1].get(), dens_l ^ noarr::fix<'y'>(m.mesh.grid_shape[1] - 1));

	solve_boundary(m.substrate_densities.get(), m.dirichlet_min_boundary_values[2].get(),
				   m.dirichlet_min_boundary_conditions[2].get(), dens_l ^ noarr::fix<'z'>(noarr::lit<0>));
	solve_boundary(m.substrate_densities.get(), m.dirichlet_max_boundary_values[2].get(),
				   m.dirichlet_max_boundary_conditions[2].get(), dens_l ^ noarr::fix<'z'>(m.mesh.grid_shape[2] - 1));
}

void dirichlet_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 1)
		solve_1d(m);
	else if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}

void dirichlet_solver::solve_1d(microenvironment& m)
{
	solve_boundaries<1>(m);
	solve_interior<1>(m.substrate_densities.get(), m.dirichlet_interior_voxels.get(), m.dirichlet_interior_values.get(),
					  m.dirichlet_interior_conditions.get(), m.substrates_count, m.dirichlet_interior_voxels_count,
					  m.mesh.grid_shape);
}

void dirichlet_solver::solve_2d(microenvironment& m)
{
	solve_boundaries<2>(m);
	solve_interior<2>(m.substrate_densities.get(), m.dirichlet_interior_voxels.get(), m.dirichlet_interior_values.get(),
					  m.dirichlet_interior_conditions.get(), m.substrates_count, m.dirichlet_interior_voxels_count,
					  m.mesh.grid_shape);
}

void dirichlet_solver::solve_3d(microenvironment& m)
{
	solve_boundaries<3>(m);
	solve_interior<3>(m.substrate_densities.get(), m.dirichlet_interior_voxels.get(), m.dirichlet_interior_values.get(),
					  m.dirichlet_interior_conditions.get(), m.substrates_count, m.dirichlet_interior_voxels_count,
					  m.mesh.grid_shape);
}
