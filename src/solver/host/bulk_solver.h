#pragma once

#include "../../../include/BioFVM/microenvironment.h"

/*
Performs bulk supply and uptake of substrates.

For each bulk (voxel), the following is performed:
S = supply_rate_f(m, voxel_idx)
U = uptake_rate_f(m, voxel_idx)
T = supply_target_densities_f(m, voxel_idx)
D = (D + dt*S*T)/(1 + dt*(U+S))
where D is a voxel substrate density vector
*/

namespace biofvm {
namespace solvers {
namespace host {

class bulk_solver
{
	bulk_func_t supply_rate_f_, uptake_rate_f_, supply_target_densities_f_;

	std::unique_ptr<real_t[]> supply_rates_, uptake_rates_, supply_target_densities_;

public:
	void initialize(microenvironment& m);

	void solve(microenvironment& m);

	void solve_1d(microenvironment& m);
	void solve_2d(microenvironment& m);
	void solve_3d(microenvironment& m);
};

} // namespace host
} // namespace solvers
} // namespace biofvm
