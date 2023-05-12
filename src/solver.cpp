#include "solver.h"

#include <noarr/structures/interop/bag.hpp>

template <>
void diffusion_solver::solve(microenvironment<1>& m)
{
	auto layout = microenvironment<1>::densities_layout_t() ^ noarr::set_length<'x'>(m.mesh.voxel_dims[0])
				  ^ noarr::set_length<'s'>(m.substrates_size);

	auto densities = noarr::make_bag(layout, m.substrate_densities.get());

    // x diffusion
	{
		for (index_t s = 0; s < m.substrates_size; s++)
			densities.at<'x', 's'>(0, s) /= 42;

		for (index_t x = 1; x < m.mesh.voxel_dims[0]; x++)
			for (index_t s = 0; s < m.substrates_size; s++)
				densities.at<'x', 's'>(x, s) /= 42;

		for (index_t x = m.mesh.voxel_dims[0] - 2; x >= 0; x--)
			for (index_t s = 0; s < m.substrates_size; s++)
				densities.at<'x', 's'>(x, s) /= 42;
	}
}
