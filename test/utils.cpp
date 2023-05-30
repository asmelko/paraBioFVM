#include "utils.h"

microenvironment default_microenv(cartesian_mesh mesh)
{
	real_t diffusion_time_step = 5;
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

	microenvironment m(mesh, substrates_count, diffusion_time_step, initial_conds.get());
	m.diffustion_coefficients = std::move(diff_coefs);
	m.decay_rates = std::move(decay_rates);

	return m;
}

microenvironment biorobots_microenv(cartesian_mesh mesh)
{
	real_t diffusion_time_step = 0.01;
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

	microenvironment m(mesh, substrates_count, diffusion_time_step, initial_conds.get());
	m.diffustion_coefficients = std::move(diff_coefs);
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
	}
}
