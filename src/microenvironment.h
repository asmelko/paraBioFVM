#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mesh.h"

struct microenvironment
{
	microenvironment() : mesh(), substrates_count(1) {}

	cartesian_mesh mesh;

	index_t substrates_count;
	real_t time_step;

	std::unique_ptr<real_t[]> diffustion_coefficients;
	std::unique_ptr<real_t[]> decay_rates;

	std::unique_ptr<real_t[]> substrate_densities;

	std::unique_ptr<real_t[]> gradients;

	std::unique_ptr<real_t[]> dirichlet_values;
	std::unique_ptr<bool[]> dirichlet_conditions;
};
