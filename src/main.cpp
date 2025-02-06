#include <chrono>
#include <iostream>

#include "solver/host/solver.h"
#include "vtk_serializer.h"

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
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 5000, 5000, 5000 }, { 20, 20, 20 });

	real_t diffusion_time_step = 1;
	index_t substrates_count = 4;

	auto diff_coefs = std::make_unique<real_t[]>(substrates_count);
	diff_coefs[0] = 10000;
	auto decay_rates = std::make_unique<real_t[]>(substrates_count);
	decay_rates[0] = 0;

	auto initial_conds = std::make_unique<real_t[]>(substrates_count);
	initial_conds[0] = 400;

	microenvironment m(mesh, substrates_count, diffusion_time_step, initial_conds.get());

	m.diffusion_coefficients = std::move(diff_coefs);
	m.decay_rates = std::move(decay_rates);
	m.compute_internalized_substrates = true;

	m.dirichlet_min_boundary_conditions[0] = std::make_unique<bool[]>(substrates_count);
	m.dirichlet_min_boundary_conditions[0][0] = true;
	m.dirichlet_min_boundary_values[0] = std::make_unique<real_t[]>(substrates_count);
	m.dirichlet_min_boundary_values[0][0] = 10000;

	for (index_t i = 0; i < substrates_count; i++)
		m.substrates_names.push_back(std::string("s") + std::to_string(i));

	make_agents(m, 2'000'000, true);

	biofvm::solvers::host::solver s;

	s.initialize(m);

	vtk_serializer serializer("output", m);

	for (index_t i = 0; i < 100; ++i)
	{
		std::size_t diffusion_duration, gradient_duration, secretion_duration;
#pragma omp parallel private(diffusion_duration, gradient_duration, secretion_duration)
		{
			{
				auto start = std::chrono::high_resolution_clock::now();

				s.diffusion.solve(m);

				auto end = std::chrono::high_resolution_clock::now();

				diffusion_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			}

			{
				auto start = std::chrono::high_resolution_clock::now();

				s.gradient.solve(m);

				auto end = std::chrono::high_resolution_clock::now();

				gradient_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			}

			{
				auto start = std::chrono::high_resolution_clock::now();

				s.cell.simulate_secretion_and_uptake(m, i % 10 == 0);

				auto end = std::chrono::high_resolution_clock::now();

				secretion_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			}

#pragma omp barrier
#pragma omp master
			{
				size_t serialize_duration;
				{
					auto start = std::chrono::high_resolution_clock::now();

					serializer.serialize_one_timestep(m);

					auto end = std::chrono::high_resolution_clock::now();

					serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				}

				std::cout << "Diffusion time: " << diffusion_duration << " ms,\t Gradient time: " << gradient_duration
						  << " ms,\t Secretion time: " << secretion_duration
						  << " ms,\t Serialize time: " << serialize_duration << " ms" << std::endl;
			}
#pragma omp barrier
		}
	}

	for (int i = 0; i < 5; i++)
	{
		s.diffusion.solve(m);
		s.gradient.solve(m);
	}
}
