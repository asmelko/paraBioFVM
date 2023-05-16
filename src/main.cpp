#include "diffusion_solvers/host_solver.h"

int main()
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 80, 80, 80 }, { 20, 20, 20 });

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

	microenvironment m(mesh, substrates_count, diffusion_time_step, std::move(diff_coefs), std::move(decay_rates),
					   std::move(initial_conds));

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);
}