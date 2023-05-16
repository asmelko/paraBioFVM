#pragma once

#include <memory>
#include <string>
#include <vector>

#include <noarr/structures_extended.hpp>

#include "mesh.h"

struct microenvironment
{
	microenvironment(cartesian_mesh mesh, index_t substrates_count, real_t time_step,
					 std::unique_ptr<real_t[]> diffustion_coefficients, std::unique_ptr<real_t[]> decay_rates,
					 std::unique_ptr<real_t[]> initial_conditions);

	cartesian_mesh mesh;

	index_t substrates_count;
	real_t time_step;

	std::unique_ptr<real_t[]> diffustion_coefficients;
	std::unique_ptr<real_t[]> decay_rates;
	std::unique_ptr<real_t[]> initial_conditions;

	std::unique_ptr<real_t[]> substrate_densities;

	std::unique_ptr<real_t[]> gradients;

	std::unique_ptr<real_t[]> dirichlet_values;
	std::unique_ptr<bool[]> dirichlet_conditions;
};
