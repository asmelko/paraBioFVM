#include "microenvironment.h"

#include <noarr/structures/extra/traverser.hpp>

#include "traits.h"

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
								   std::unique_ptr<real_t[]> diffustion_coefficients,
								   std::unique_ptr<real_t[]> decay_rates, std::unique_ptr<real_t[]> initial_conditions)
	: mesh(mesh),
	  substrates_count(substrates_count),
	  time_step(time_step),
	  diffustion_coefficients(std::move(diffustion_coefficients)),
	  decay_rates(std::move(decay_rates)),
	  initial_conditions(std::move(initial_conditions)),
	  substrate_densities(std::make_unique<real_t[]>(substrates_count * mesh.voxel_count())),
	  gradients(std::make_unique<real_t[]>(mesh.dims * substrates_count * mesh.voxel_count())),
	  dirichlet_values(std::make_unique<real_t[]>(substrates_count * mesh.voxel_count())),
	  dirichlet_conditions(std::make_unique<bool[]>(substrates_count * mesh.voxel_count()))
{
	if (mesh.dims == 1)
		initialize_substrate_densities<1>(substrate_densities.get(), this->initial_conditions.get(), *this);
	else if (mesh.dims == 2)
		initialize_substrate_densities<2>(substrate_densities.get(), this->initial_conditions.get(), *this);
	else if (mesh.dims == 3)
		initialize_substrate_densities<3>(substrate_densities.get(), this->initial_conditions.get(), *this);
}
