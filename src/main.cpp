#include <chrono>
#include <iostream>

#include "solver/host/solver.h"

using namespace biofvm;

void make_agents(microenvironment& m, index_t count, bool conflict)
{
	index_t x = 0, y = 0, z = 0;

	for (index_t i = 0; i < count; i++)
	{
		auto a = m.agents->create_agent();
		a->position()[0] = x;
		a->position()[1] = y;
		a->position()[2] = z;

		x += 20;
		if (x >= m.mesh.bounding_box_maxs[0])
		{
			x -= m.mesh.bounding_box_maxs[0];
			y += 20;
		}
		if (y >= m.mesh.bounding_box_maxs[1])
		{
			y -= m.mesh.bounding_box_maxs[1];
			z += 20;
		}
	}

	if (conflict)
	{
		auto a = m.agents->create_agent();
		a->position()[0] = 0;
		a->position()[1] = 0;
		a->position()[2] = 0;
	}
}

int main()
{
	index_t dx = 16;
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 32 * 10, 32 * 4, 32 }, { dx, dx, dx });

	real_t diffusion_time_step = .1;
	index_t substrates_count = 1;

	auto diff_coefs = std::make_unique<real_t[]>(substrates_count);
	diff_coefs[0] = 10;
	auto decay_rates = std::make_unique<real_t[]>(substrates_count);
	decay_rates[0] = 0;

	auto initial_conds = std::make_unique<real_t[]>(substrates_count);
	initial_conds[0] = 100;

	microenvironment m(mesh, substrates_count, diffusion_time_step, initial_conds.get());

	for (index_t x = 0; x < 4 * (32 / dx); x++)
	{
		m.substrate_densities[x] = 0;
		m.substrate_densities[mesh.grid_shape[0] - 1 - x] = 0;
	}

	// for (index_t x = 0; x < mesh.grid_shape[0]; x++)
	// {
	// 	m.substrate_densities[x + 1 * mesh.grid_shape[0]] = 0;
	// 	m.substrate_densities[x + 2 * mesh.grid_shape[0]] = 0;
	// 	m.substrate_densities[x + 3 * mesh.grid_shape[0]] = 0;
	// }

	m.diffusion_coefficients = std::move(diff_coefs);
	m.decay_rates = std::move(decay_rates);
	m.compute_internalized_substrates = true;

	// make_agents(m, 2'000'000, true);

	biofvm::solvers::host::solver s;

	s.initialize(m);

	// 	for (index_t i = 0; i < 100; ++i)
	// 	{
	// 		std::size_t diffusion_duration, gradient_duration, secretion_duration;
	// #pragma omp parallel private(diffusion_duration, gradient_duration, secretion_duration)
	// 		{
	// 			{
	// 				auto start = std::chrono::high_resolution_clock::now();

	// 				s.diffusion.solve(m);

	// 				auto end = std::chrono::high_resolution_clock::now();

	// 				diffusion_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 			}

	// 			{
	// 				auto start = std::chrono::high_resolution_clock::now();

	// 				s.gradient.solve(m);

	// 				auto end = std::chrono::high_resolution_clock::now();

	// 				gradient_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 			}

	// 			{
	// 				auto start = std::chrono::high_resolution_clock::now();

	// 				s.cell.simulate_secretion_and_uptake(m, i % 10 == 0);

	// 				auto end = std::chrono::high_resolution_clock::now();

	// 				secretion_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 			}

	// #pragma omp master
	// 			std::cout << "Diffusion time: " << diffusion_duration << " ms,\t Gradient time: " << gradient_duration
	// 					  << " ms,\t Secretion time: " << secretion_duration << " ms" << std::endl;
	// 		}
	// 	}
	real_t sum = 0;
	for (index_t j = 0; j < mesh.grid_shape[0]; j++)
	{
		sum += m.substrate_densities[j];
	}
	std::cout << "0|" << sum / mesh.grid_shape[0] << "|";

	for (index_t j = 0; j < mesh.grid_shape[0]; j++)
	{
		std::cout << m.substrate_densities[j] << " ";
	}
	std::cout << std::endl;

	for (real_t i = diffusion_time_step; i <= 1; i += diffusion_time_step)
	{
		s.diffusion.solve(m);

		real_t sum = 0;
		for (index_t j = 0; j < mesh.grid_shape[0]; j++)
		{
			sum += m.substrate_densities[j];
		}
		std::cout << i << "|" << sum / mesh.grid_shape[0] << "|";

		for (index_t j = 0; j < mesh.grid_shape[0]; j++)
		{
			std::cout << m.substrate_densities[j] << " ";
		}
		std::cout << std::endl;
	}
}
