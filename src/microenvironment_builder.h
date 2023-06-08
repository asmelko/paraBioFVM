#pragma once

#include <optional>

#include "mesh.h"
#include "microenvironment.h"
#include "types.h"

class microenvironment_builder
{
	std::string name, time_units, space_units;

	real_t time_step;
	std::optional<cartesian_mesh> mesh_;

	std::vector<std::string> substrates_names;
	std::vector<std::string> substrates_units;

	std::vector<real_t> diffusion_coefficients;
	std::vector<real_t> decay_rates;
	std::vector<real_t> initial_conditions;

	std::vector<index_t> dirichlet_voxels;
	std::vector<real_t> dirichlet_values;
	std::vector<bool> dirichlet_conditions;

	std::vector<point_t<real_t, 3>> boundary_dirichlet_mins_values;
	std::vector<point_t<real_t, 3>> boundary_dirichlet_maxs_values;
	std::vector<point_t<bool, 3>> boundary_dirichlet_mins_conditions;
	std::vector<point_t<bool, 3>> boundary_dirichlet_maxs_conditions;

	bulk_func_t supply_rate_func, uptake_rate_func, supply_target_densities_func;

	bool compute_internalized_substrates = false;

	void fill_dirichlet_vectors(microenvironment& m);

public:
	void set_name(const std::string& name);
	void set_time_units(const std::string& units);
	void set_space_units(const std::string& units);

	void set_time_step(real_t time_step);

	// mesh functions
	void resize(index_t dims, point_t<index_t, 3> bounding_box_mins, point_t<index_t, 3> bounding_box_maxs,
				point_t<index_t, 3> voxel_shape);

	// density functions
	void add_density(const std::string& name, const std::string& units, real_t diffusion_coefficient = 0,
					 real_t decay_rate = 0, real_t initial_condition = 0);
	std::size_t get_density_index(const std::string& name) const;

	// dirichlet functions
	void add_dirichlet_node(point_t<index_t, 3> voxel_index, std::vector<real_t> values,
							std::vector<bool> conditions = {});
	void add_boundary_dirichlet_conditions(std::size_t density_index, point_t<real_t, 3> mins_values,
										   point_t<real_t, 3> maxs_values,
										   point_t<bool, 3> mins_conditions = { true, true, true },
										   point_t<bool, 3> maxs_conditions = { true, true, true });

	void set_bulk_functions(bulk_func_t supply_rate_func, bulk_func_t uptake_rate_func,
							bulk_func_t supply_target_densities_func);

	void do_compute_internalized_substrates();

	microenvironment build();
};
