#pragma once

#include "dirichlet_solver.h"

namespace biofvm {
namespace solvers {
namespace host {

class lapack_diffusion_solver
{
	std::vector<std::unique_ptr<real_t[]>> ax_, bx_;
	std::vector<std::unique_ptr<real_t[]>> ay_, by_;
	std::vector<std::unique_ptr<real_t[]>> az_, bz_;

	void solve_1d(microenvironment& m);
	void solve_2d(microenvironment& m);
	void solve_3d(microenvironment& m);

public:
	static void precompute_values(std::vector<std::unique_ptr<real_t[]>>& a, std::vector<std::unique_ptr<real_t[]>>& b,
								  index_t shape, index_t dims, index_t n, const microenvironment& m);

	void initialize(microenvironment& m, dirichlet_solver& dirichlet);
	void initialize(microenvironment& m);

	void solve(microenvironment& m);
};

} // namespace host
} // namespace solvers
} // namespace biofvm
