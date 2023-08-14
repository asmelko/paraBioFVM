#include "microenvironment.h"

#include <cmath>
#include <iostream>

#include <noarr/structures/extra/traverser.hpp>

#include "agent_container.h"
#include "omp_utils.h"
#include "traits.h"

using namespace biofvm;

template <index_t dims>
void initialize_substrate_densities(real_t* substrate_densities, const real_t* initial_conditions,
									const microenvironment& m)
{
	auto dens_l = layout_traits<dims>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	omp_p_trav_for_each(noarr::traverser(dens_l), [=](auto state) {
		auto s = noarr::get_index<'s'>(state);

		(dens_l | noarr::get_at(substrate_densities, state)) = initial_conditions[s];
	});
}

microenvironment::microenvironment(cartesian_mesh mesh, index_t substrates_count, real_t time_step,
								   const real_t* initial_conditions)
	: mesh(mesh),
	  agents(std::make_unique<agent_container>(*this)),
	  substrates_count(substrates_count),
	  diffusion_time_step(time_step),
	  substrate_densities(std::make_unique<real_t[]>(substrates_count * mesh.voxel_count())),
	  diffusion_coefficients(nullptr),
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

real_t microenvironment::substrate_density_value(point_t<index_t, 3> position, index_t substrate_index) const
{
	if (mesh.dims == 1)
	{
		auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);

		return (dens_l | noarr::get_at<'x', 's'>(substrate_densities.get(), position[0], substrate_index));
	}
	else if (mesh.dims == 2)
	{
		auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);

		return (dens_l
				| noarr::get_at<'x', 'y', 's'>(substrate_densities.get(), position[0], position[1], substrate_index));
	}
	else if (mesh.dims == 3)
	{
		auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);

		return (dens_l
				| noarr::get_at<'x', 'y', 'z', 's'>(substrate_densities.get(), position[0], position[1], position[2],
													substrate_index));
	}
	return 0;
}

void microenvironment::display_info()
{
	auto get_initial_condition = [&](index_t s) {
		if (mesh.dims == 1)
		{
			auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);
			return (dens_l | noarr::get_at<'x', 's'>(substrate_densities.get(), 0, s));
		}
		else if (mesh.dims == 2)
		{
			auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);
			return (dens_l | noarr::get_at<'x', 'y', 's'>(substrate_densities.get(), 0, 0, s));
		}
		else if (mesh.dims == 3)
		{
			auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);
			return (dens_l | noarr::get_at<'x', 'y', 'z', 's'>(substrate_densities.get(), 0, 0, 0, s));
		}
		return (real_t)0;
	};

	std::cout << std::endl << "Microenvironment summary: " << name << ": " << std::endl;
	mesh.display_info();
	std::cout << "Densities: (" << substrates_count << " total)" << std::endl;
	for (unsigned int i = 0; i < substrates_names.size(); i++)
	{
		std::cout << "   " << substrates_names[i] << ":" << std::endl
				  << "     units: " << substrates_units[i] << std::endl
				  << "     diffusion coefficient: " << diffusion_coefficients[i] << " " << space_units << "^2 / "
				  << time_units << std::endl
				  << "     decay rate: " << decay_rates[i] << " " << time_units << "^-1" << std::endl
				  << "     diffusion length scale: " << std::sqrt(diffusion_coefficients[i] / (1e-12 + decay_rates[i]))
				  << " " << space_units << std::endl
				  << "     initial condition: " << get_initial_condition(i) << " " << substrates_units[i] << std::endl;
		// 			  << "     boundary condition: " << default_microenvironment_options.Dirichlet_condition_vector[i]
		// 			  << " " << substrates_units[i] << " (enabled: ";
		// 	if (dirichlet_activation_vector[i] == true)
		// 	{
		// 		std::cout << "true";
		// 	}
		// 	else
		// 	{
		// 		std::cout << "false";
		// 	}
		// 	std::cout << ")" << std::endl;
	}
	std::cout << std::endl;

	return;
}
