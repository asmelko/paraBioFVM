#pragma once

#include <vector>

#include "microenvironment.h"
#include "types.h"

microenvironment default_microenv(cartesian_mesh mesh);

microenvironment biorobots_microenv(cartesian_mesh mesh);

void add_dirichlet_at(microenvironment& m, index_t substrates_count, const std::vector<point_t<index_t, 3>>& indices,
					  const std::vector<real_t>& values);

void add_boundary_dirichlet(microenvironment& m, index_t substrates_count, index_t dim_idx, bool min, real_t value);

void compute_expected_agent_internalized_1d(microenvironment& m, std::vector<real_t>& expected_internalized);

std::vector<real_t> compute_expected_agent_densities_1d(microenvironment& m);

void set_default_agent_values(agent* a, index_t rates_offset, index_t volume, point_t<real_t, 3> position,
							  index_t dims);
