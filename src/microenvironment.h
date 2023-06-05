#pragma once

#include <memory>
#include <string>
#include <vector>

#include <noarr/structures_extended.hpp>

#include "agent_container.h"
#include "mesh.h"
#include "types.h"

struct microenvironment
{
	microenvironment(cartesian_mesh mesh, index_t substrates_count, real_t time_step, const real_t* initial_conditions);

	std::string name, time_units, space_units;

	std::vector<std::string> substrates_names;
	std::vector<std::string> substrates_units;

	cartesian_mesh mesh;

	agent_container agents;

	index_t substrates_count;
	real_t time_step;

	std::unique_ptr<real_t[]> substrate_densities;

	std::unique_ptr<real_t[]> diffustion_coefficients;
	std::unique_ptr<real_t[]> decay_rates;

	std::unique_ptr<real_t[]> gradients;

	index_t dirichlet_interior_voxels_count;
	std::unique_ptr<index_t[]> dirichlet_interior_voxels;
	std::unique_ptr<real_t[]> dirichlet_interior_values;
	std::unique_ptr<bool[]> dirichlet_interior_conditions;

	point_t<std::unique_ptr<real_t[]>, 3> dirichlet_min_boundary_values;
	point_t<std::unique_ptr<real_t[]>, 3> dirichlet_max_boundary_values;
	point_t<std::unique_ptr<bool[]>, 3> dirichlet_min_boundary_conditions;
	point_t<std::unique_ptr<bool[]>, 3> dirichlet_max_boundary_conditions;
};
