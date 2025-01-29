#include "utils.h"

#include <noarr/structures/interop/bag.hpp>

#include "agent_container.h"
#include "traits.h"

namespace biofvm {

microenvironment default_microenv(cartesian_mesh mesh)
{
	real_t diffusion_diffusion_time_step = 5;
	index_t substrates_count = 2;

	auto diff_coefs = std::make_unique<real_t[]>(2);
	diff_coefs[0] = 4;
	diff_coefs[1] = 2;
	auto decay_rates = std::make_unique<real_t[]>(2);
	decay_rates[0] = 5;
	decay_rates[1] = 3;

	auto initial_conds = std::make_unique<real_t[]>(2);
	initial_conds[0] = 1;
	initial_conds[1] = 1;

	microenvironment m(mesh, substrates_count, diffusion_diffusion_time_step, initial_conds.get());
	m.diffusion_coefficients = std::move(diff_coefs);
	m.decay_rates = std::move(decay_rates);

	return m;
}

microenvironment biorobots_microenv(cartesian_mesh mesh)
{
	real_t diffusion_diffusion_time_step = 0.01;
	index_t substrates_count = 2;

	auto diff_coefs = std::make_unique<real_t[]>(2);
	diff_coefs[0] = 1000;
	diff_coefs[1] = 1000;
	auto decay_rates = std::make_unique<real_t[]>(2);
	decay_rates[0] = 0.1;
	decay_rates[1] = 0.4;

	auto initial_conds = std::make_unique<real_t[]>(2);
	initial_conds[0] = 0;
	initial_conds[1] = 0;

	microenvironment m(mesh, substrates_count, diffusion_diffusion_time_step, initial_conds.get());
	m.diffusion_coefficients = std::move(diff_coefs);
	m.decay_rates = std::move(decay_rates);

	return m;
}

microenvironment interactions_microenv(cartesian_mesh mesh)
{
	real_t diffusion_diffusion_time_step = 0.01;
	index_t substrates_count = 5;

	auto diff_coefs = std::make_unique<real_t[]>(substrates_count);
	diff_coefs[0] = 100000;
	diff_coefs[1] = 100;
	diff_coefs[2] = 1000;
	diff_coefs[3] = 1000;
	diff_coefs[4] = 1000;
	auto decay_rates = std::make_unique<real_t[]>(substrates_count);
	decay_rates[0] = 0.1;
	decay_rates[1] = 0.1;
	decay_rates[2] = 0.1;
	decay_rates[3] = 0.01;
	decay_rates[4] = 0.1;

	auto initial_conds = std::make_unique<real_t[]>(substrates_count);
	initial_conds[0] = 1;
	initial_conds[1] = 0;
	initial_conds[2] = 0;
	initial_conds[3] = 0;
	initial_conds[4] = 0;

	microenvironment m(mesh, substrates_count, diffusion_diffusion_time_step, initial_conds.get());
	m.diffusion_coefficients = std::move(diff_coefs);
	m.decay_rates = std::move(decay_rates);

	return m;
}

void add_dirichlet_at(microenvironment& m, index_t substrates_count, const std::vector<point_t<index_t, 3>>& indices,
					  const std::vector<real_t>& values)
{
	m.dirichlet_interior_voxels_count = indices.size();
	m.dirichlet_interior_voxels = std::make_unique<index_t[]>(m.dirichlet_interior_voxels_count * m.mesh.dims);

	for (int i = 0; i < m.dirichlet_interior_voxels_count; i++)
		for (int d = 0; d < m.mesh.dims; d++)
			m.dirichlet_interior_voxels[i * m.mesh.dims + d] = indices[i][d];

	m.dirichlet_interior_values = std::make_unique<real_t[]>(substrates_count * m.dirichlet_interior_voxels_count);
	m.dirichlet_interior_conditions = std::make_unique<bool[]>(substrates_count * m.dirichlet_interior_voxels_count);

	for (int i = 0; i < m.dirichlet_interior_voxels_count; i++)
	{
		m.dirichlet_interior_values[i * substrates_count] = values[i];
		m.dirichlet_interior_conditions[i * substrates_count] = true; // only the first substrate

		for (int j = 1; j < m.substrates_count; j++)
			m.dirichlet_interior_conditions[i * substrates_count + j] = false;
	}
}

void add_boundary_dirichlet(microenvironment& m, index_t substrates_count, index_t dim_idx, bool min, real_t value)
{
	auto& values = min ? m.dirichlet_min_boundary_values[dim_idx] : m.dirichlet_max_boundary_values[dim_idx];
	auto& conditions =
		min ? m.dirichlet_min_boundary_conditions[dim_idx] : m.dirichlet_max_boundary_conditions[dim_idx];

	if (!values)
	{
		values = std::make_unique<real_t[]>(substrates_count);
		conditions = std::make_unique<bool[]>(substrates_count);

		for (index_t s = 0; s < substrates_count; s++)
		{
			values[s] = 42;
			conditions[s] = false;
		}
	}

	// only the first substrate
	values[0] = value;
	conditions[0] = true;
}

void compute_expected_agent_internalized_1d(microenvironment& m, std::vector<real_t>& expected_internalized)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	auto densities = noarr::make_bag(dens_l, m.substrate_densities.get());

	auto& agent_data = dynamic_cast<agent_container*>(m.agents.get())->data();

	for (index_t i = 0; i < agent_data.agents_count; i++)
	{
		for (index_t s = 0; s < m.substrates_count; s++)
		{
			auto num = agent_data.secretion_rates[i * m.substrates_count + s]
					   * agent_data.saturation_densities[i * m.substrates_count + s] * m.diffusion_time_step
					   * agent_data.volumes[i];

			auto denom = (agent_data.secretion_rates[i * m.substrates_count + s]
						  + agent_data.uptake_rates[i * m.substrates_count + s])
						 * m.diffusion_time_step * agent_data.volumes[i] / m.mesh.voxel_volume();

			auto factor = agent_data.net_export_rates[i * m.substrates_count + s] * m.diffusion_time_step;

			auto mesh_idx = m.mesh.voxel_position<1>(agent_data.positions.data() + i);

			expected_internalized[i * m.substrates_count + s] -=
				(m.mesh.voxel_volume() * -denom * densities.at<'x', 's'>(mesh_idx[0], s) + num) / (1 + denom) + factor;
		}
	}
}

std::vector<real_t> compute_expected_agent_densities_1d(microenvironment& m)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	auto densities = noarr::make_bag(dens_l, m.substrate_densities.get());

	auto& agent_data = dynamic_cast<agent_container*>(m.agents.get())->data();

	std::vector<real_t> expected_densities(m.mesh.voxel_count() * m.substrates_count, 0);

	for (index_t s = 0; s < m.substrates_count; s++)
	{
		for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		{
			real_t num = 0, denom = 0, factor = 0;

			for (index_t i = 0; i < agent_data.agents_count; i++)
			{
				if (agent_data.positions[i] / m.mesh.voxel_shape[0] == x)
				{
					num += agent_data.secretion_rates[i * m.substrates_count + s]
						   * agent_data.saturation_densities[i * m.substrates_count + s] * m.diffusion_time_step
						   * agent_data.volumes[i] / m.mesh.voxel_volume();

					denom += (agent_data.secretion_rates[i * m.substrates_count + s]
							  + agent_data.uptake_rates[i * m.substrates_count + s])
							 * m.diffusion_time_step * agent_data.volumes[i] / m.mesh.voxel_volume();

					factor += agent_data.net_export_rates[i * m.substrates_count + s] * m.diffusion_time_step
							  / m.mesh.voxel_volume();
				}
			}
			expected_densities[x * m.substrates_count + s] =
				((densities.at<'x', 's'>(x, s) + num) / (1 + denom) + factor);
		}
	}

	return expected_densities;
}

void set_default_agent_values(agent* a, index_t rates_offset, index_t volume, point_t<real_t, 3> position, index_t dims)
{
	a->secretion_rates()[0] = rates_offset + 100;
	a->secretion_rates()[1] = 0;

	a->uptake_rates()[0] = rates_offset + 200;
	a->uptake_rates()[1] = 0;

	a->saturation_densities()[0] = rates_offset + 300;
	a->saturation_densities()[1] = 0;

	a->net_export_rates()[0] = rates_offset + 400;
	a->net_export_rates()[1] = 0;

	a->volume() = volume;

	for (index_t i = 0; i < dims; ++i)
		a->position()[i] = position[i];
}

} // namespace biofvm
