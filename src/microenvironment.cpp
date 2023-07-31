#include "microenvironment.h"

#include <noarr/structures/extra/traverser.hpp>

#include "agent_container.h"
#include "traits.h"

using namespace biofvm;

template <index_t dims>
void initialize_substrate_densities(real_t* substrate_densities, const real_t* initial_conditions,
									const microenvironment& m)
{
	auto dens_l = layout_traits<dims>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	noarr::traverser(dens_l).for_each([&](auto state) {
		auto s = noarr::get_index<'s'>(state);

		(dens_l | noarr::get_at(substrate_densities, state)) = initial_conditions[s];
	});
}

microenvironment::microenvironment(cartesian_mesh mesh, index_t substrates_count, real_t time_step,
								   const real_t* initial_conditions)
	: mesh(mesh),
	  agents(std::make_unique<agent_container>(*this)),
	  substrates_count(substrates_count),
	  time_step(time_step),
	  substrate_densities(std::make_unique<real_t[]>(substrates_count * mesh.voxel_count())),
	  diffustion_coefficients(nullptr),
	  decay_rates(nullptr),
	  gradients(std::make_unique<real_t[]>(mesh.dims * substrates_count * mesh.voxel_count())),
	  dirichlet_interior_voxels_count(0),
	  dirichlet_interior_voxels(nullptr),
	  dirichlet_interior_values(nullptr),
	  dirichlet_interior_conditions(nullptr),
	  dirichlet_min_boundary_values({ nullptr, nullptr, nullptr }),
	  dirichlet_max_boundary_values({ nullptr, nullptr, nullptr }),
	  dirichlet_min_boundary_conditions({ nullptr, nullptr, nullptr }),
	  dirichlet_max_boundary_conditions({ nullptr, nullptr, nullptr })
{
	if (mesh.dims == 1)
		initialize_substrate_densities<1>(substrate_densities.get(), initial_conditions, *this);
	else if (mesh.dims == 2)
		initialize_substrate_densities<2>(substrate_densities.get(), initial_conditions, *this);
	else if (mesh.dims == 3)
		initialize_substrate_densities<3>(substrate_densities.get(), initial_conditions, *this);
}

index_t microenvironment::find_substrate_index(const std::string& name) const
{
	for (index_t i = 0; i < substrates_count; i++)
		if (substrates_names[i] == name)
			return i;

	return -1;
}
