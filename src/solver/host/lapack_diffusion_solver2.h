#pragma once

#include <memory>
#include "dirichlet_solver.h"

namespace biofvm {
namespace solvers {
namespace host {

class lapack_diffusion_solver2
{
	std::vector<std::unique_ptr<real_t[]>> dlx_, dx_, dux_, du2x_;
	std::vector<std::unique_ptr<real_t[]>> dly_, dy_, duy_, du2y_;
	std::vector<std::unique_ptr<real_t[]>> dlz_, dz_, duz_, du2z_;

	std::vector<std::unique_ptr<int[]>> ipivx_, ipivy_, ipivz_;

	void solve_1d(microenvironment& m);
	void solve_2d(microenvironment& m);
	void solve_3d(microenvironment& m);

public:
	static void precompute_values(std::vector<std::unique_ptr<real_t[]>>& dls,
								  std::vector<std::unique_ptr<real_t[]>>& ds,
								  std::vector<std::unique_ptr<real_t[]>>& dus,
								  std::vector<std::unique_ptr<real_t[]>>& du2s,
								  std::vector<std::unique_ptr<int[]>>& ipivs, index_t shape, index_t dims, index_t n,
								  const microenvironment& m);

	void initialize(microenvironment& m, dirichlet_solver& dirichlet);
	void initialize(microenvironment& m);

	void solve(microenvironment& m);
};

} // namespace host
} // namespace solvers
} // namespace biofvm
