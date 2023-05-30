#pragma once

#include <vector>

#include "microenvironment.h"
#include "types.h"

microenvironment default_microenv(cartesian_mesh mesh);

microenvironment biorobots_microenv(cartesian_mesh mesh);

void add_dirichlet_at(microenvironment& m, index_t substrates_count, const std::vector<point_t<index_t, 3>>& indices,
					  const std::vector<real_t>& values);

void add_boundary_dirichlet(microenvironment& m, index_t substrates_count, index_t dim_idx, bool min, real_t value);
