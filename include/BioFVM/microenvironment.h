#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "agent_container_base.h"
#include "mesh.h"
#include "types.h"

namespace biofvm {

struct microenvironment;

using bulk_func_t = std::function<void(microenvironment& m, point_t<index_t, 3> voxel_idx, real_t* out)>;

struct microenvironment
{
	microenvironment(cartesian_mesh mesh, index_t substrates_count, real_t time_step, const real_t* initial_conditions);

	std::string name, time_units, space_units;

	std::vector<std::string> substrates_names;
	std::vector<std::string> substrates_units;

	cartesian_mesh mesh;

	agent_container_ptr agents;

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

	bulk_func_t supply_rate_func, uptake_rate_func, supply_target_densities_func;

	bool compute_internalized_substrates;
};

} // namespace biofvm
