#include "microenvironment_builder.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "microenvironment.h"
#include "types.h"

using namespace biofvm;

void microenvironment_builder::set_name(const std::string& name) { this->name = name; }

void microenvironment_builder::set_time_units(const std::string& time_units) { this->time_units = time_units; }

void microenvironment_builder::set_space_units(const std::string& space_units) { this->space_units = space_units; }

void microenvironment_builder::set_time_step(real_t time_step) { this->time_step = time_step; }

void microenvironment_builder::resize(index_t dims, point_t<index_t, 3> bounding_box_mins,
									  point_t<index_t, 3> bounding_box_maxs, point_t<index_t, 3> voxel_shape)
{
	mesh_ = cartesian_mesh(dims, bounding_box_mins, bounding_box_maxs, voxel_shape);
}

void microenvironment_builder::add_density(const std::string& name, const std::string& units,
										   real_t diffusion_coefficient, real_t decay_rate, real_t initial_condition)
{
	substrates_names.push_back(name);
	substrates_units.push_back(units);
	diffusion_coefficients.push_back(diffusion_coefficient);
	decay_rates.push_back(decay_rate);
	initial_conditions.push_back(initial_condition);

	boundary_dirichlet_mins_values.push_back({ 0, 0, 0 });
	boundary_dirichlet_maxs_values.push_back({ 0, 0, 0 });
	boundary_dirichlet_mins_conditions.push_back({ false, false, false });
	boundary_dirichlet_maxs_conditions.push_back({ false, false, false });
}

std::size_t microenvironment_builder::get_density_index(const std::string& name) const
{
	auto it = std::find(substrates_names.begin(), substrates_names.end(), name);
	if (it == substrates_names.end())
	{
		throw std::runtime_error("Density " + name + " not found");
	}
	return std::distance(substrates_names.begin(), it);
}

void microenvironment_builder::add_dirichlet_node(point_t<index_t, 3> voxel_index, std::vector<real_t> values,
												  std::vector<bool> conditions)
{
	if (values.size() != substrates_names.size())
	{
		throw std::runtime_error("Dirichlet node values size does not match the number of densities");
	}
	if (!conditions.empty() && conditions.size() != substrates_names.size())
	{
		throw std::runtime_error("Dirichlet node conditions size does not match the number of densities");
	}
	if (!mesh_)
	{
		throw std::runtime_error("Dirichlet node cannot be added without a mesh");
	}

	if (mesh_->dims >= 1)
		dirichlet_voxels.push_back(voxel_index[0]);
	if (mesh_->dims >= 2)
		dirichlet_voxels.push_back(voxel_index[1]);
	if (mesh_->dims == 3)
		dirichlet_voxels.push_back(voxel_index[2]);

	dirichlet_values.insert(dirichlet_values.end(), values.begin(), values.end());

	if (conditions.empty())
	{
		conditions.resize(values.size(), true);
	}

	dirichlet_conditions.insert(dirichlet_conditions.end(), conditions.begin(), conditions.end());
}

void microenvironment_builder::add_boundary_dirichlet_conditions(std::size_t density_index,
																 point_t<real_t, 3> mins_values,
																 point_t<real_t, 3> maxs_values,
																 point_t<bool, 3> mins_conditions,
																 point_t<bool, 3> maxs_conditions)
{
	if (density_index >= substrates_names.size())
	{
		throw std::runtime_error("Density index out of bounds");
	}

	boundary_dirichlet_mins_values[density_index] = mins_values;
	boundary_dirichlet_maxs_values[density_index] = maxs_values;
	boundary_dirichlet_mins_conditions[density_index] = mins_conditions;
	boundary_dirichlet_maxs_conditions[density_index] = maxs_conditions;
}

void microenvironment_builder::set_bulk_functions(bulk_func_t supply_rate_func, bulk_func_t uptake_rate_func,
												  bulk_func_t supply_target_densities_func)
{
	this->supply_rate_func = std::move(supply_rate_func);
	this->uptake_rate_func = std::move(uptake_rate_func);
	this->supply_target_densities_func = std::move(supply_target_densities_func);
}

void microenvironment_builder::do_compute_internalized_substrates() { compute_internalized_substrates = true; }

void fill_one(index_t dim_idx, index_t substrates_count, const std::vector<point_t<real_t, 3>>& values,
			  const std::vector<point_t<bool, 3>>& conditions, point_t<std::unique_ptr<real_t[]>, 3>& linearized_values,
			  point_t<std::unique_ptr<bool[]>, 3>& linearized_conditions)
{
	bool any = false;
	for (index_t s = 0; s < substrates_count; s++)
	{
		any |= conditions[s][dim_idx];
	}

	if (any)
	{
		linearized_values[dim_idx] = std::make_unique<real_t[]>(substrates_count);
		linearized_conditions[dim_idx] = std::make_unique<bool[]>(substrates_count);

		for (index_t s = 0; s < substrates_count; s++)
		{
			linearized_values[dim_idx][s] = values[s][dim_idx];
			linearized_conditions[dim_idx][s] = conditions[s][dim_idx];
		}
	}
}

void microenvironment_builder::fill_dirichlet_vectors(microenvironment& m)
{
	for (index_t d = 0; d < m.mesh.dims; d++)
	{
		fill_one(d, m.substrates_count, boundary_dirichlet_mins_values, boundary_dirichlet_mins_conditions,
				 m.dirichlet_min_boundary_values, m.dirichlet_min_boundary_conditions);
		fill_one(d, m.substrates_count, boundary_dirichlet_maxs_values, boundary_dirichlet_maxs_conditions,
				 m.dirichlet_max_boundary_values, m.dirichlet_max_boundary_conditions);
	}
}

void microenvironment_builder::load_initial_conditions_from_file(const std::string& file)
{
	initial_conditions_file = file;
}

void microenvironment_builder::fill_initial_conditions_from_file(microenvironment& m)
{
	if (initial_conditions_file.empty())
	{
		return;
	}

	std::ifstream file(initial_conditions_file);

	if (!file.is_open())
	{
		throw std::runtime_error("Cannot open file " + initial_conditions_file);
	}

	auto split_line = [](const std::string& line) {
		std::vector<std::string> values;
		std::stringstream ss(line);
		std::string value;
		while (std::getline(ss, value, ','))
		{
			values.push_back(value);
		}
		return values;
	};

	std::vector<index_t> substrate_indices;
	// check if csv file has header
	{
		std::string line;
		std::getline(file, line);
		auto values = split_line(line);
		if (values.size() < 4)
		{
			throw std::runtime_error("Invalid initial conditions file format");
		}

		// setup substrate_indices
		if ((values[0] == "X" || values[0] == "x") && (values[1] == "Y" || values[1] == "y")
			&& (values[2] == "Z" || values[2] == "z"))
		{
			for (std::size_t i = 3; i < values.size(); i++)
			{
				auto it = std::find(m.substrates_names.begin(), m.substrates_names.end(), values[i]);
				if (it == m.substrates_names.end())
				{
					throw std::runtime_error("Substrate " + values[i] + " not found during initial conditions loading");
				}
				substrate_indices.push_back(std::distance(m.substrates_names.begin(), it));
			}
		}
		else
		{
			for (std::size_t i = 3; i < values.size(); i++)
				substrate_indices.push_back(i - 3);

			// rewind file
			file.clear();
			file.seekg(0, std::ios::beg);
		}
	}

	while (true)
	{
		std::string line;
		std::getline(file, line);

		if (file.eof())
			break;

		auto values = split_line(line);
		if ((index_t)values.size() != 3 + m.substrates_count)
		{
			throw std::runtime_error("Invalid initial conditions file format");
		}

		point_t<index_t, 3> position;
		position[0] = std::stoi(values[0]);
		position[1] = std::stoi(values[1]);
		position[2] = std::stoi(values[2]);

		for (std::size_t i = 3; i < values.size(); i++)
		{
			m.substrate_density_value(position, substrate_indices[i - 3]) = std::stod(values[i]);
		}
	}
}

microenvironment microenvironment_builder::build()
{
	if (!mesh_)
	{
		throw std::runtime_error("Microenvironment cannot be built without a mesh");
	}

	if (substrates_names.empty())
	{
		throw std::runtime_error("Microenvironment cannot be built with no densities");
	}

	microenvironment m(*mesh_, substrates_names.size(), time_step, initial_conditions.data());

	m.name = std::move(name);
	m.time_units = std::move(time_units);
	m.space_units = std::move(space_units);

	m.substrates_names = std::move(substrates_names);
	m.substrates_units = std::move(substrates_units);

	fill_initial_conditions_from_file(m);

	m.diffusion_coefficients = std::make_unique<real_t[]>(diffusion_coefficients.size());
	std::memcpy(m.diffusion_coefficients.get(), diffusion_coefficients.data(),
				diffusion_coefficients.size() * sizeof(real_t));

	m.decay_rates = std::make_unique<real_t[]>(decay_rates.size());
	std::memcpy(m.decay_rates.get(), decay_rates.data(), decay_rates.size() * sizeof(real_t));

	m.dirichlet_interior_voxels_count = dirichlet_voxels.size() / m.mesh.dims;
	m.dirichlet_interior_voxels = std::make_unique<index_t[]>(dirichlet_voxels.size());
	std::copy(dirichlet_voxels.begin(), dirichlet_voxels.end(), m.dirichlet_interior_voxels.get());

	m.dirichlet_interior_values = std::make_unique<real_t[]>(dirichlet_values.size());
	std::copy(dirichlet_values.begin(), dirichlet_values.end(), m.dirichlet_interior_values.get());

	m.dirichlet_interior_conditions = std::make_unique<bool[]>(dirichlet_conditions.size());
	std::copy(dirichlet_conditions.begin(), dirichlet_conditions.end(), m.dirichlet_interior_conditions.get());

	fill_dirichlet_vectors(m);

	m.supply_rate_func = std::move(supply_rate_func);
	m.uptake_rate_func = std::move(uptake_rate_func);
	m.supply_target_densities_func = std::move(supply_target_densities_func);

	m.compute_internalized_substrates = compute_internalized_substrates;

	return m;
}
