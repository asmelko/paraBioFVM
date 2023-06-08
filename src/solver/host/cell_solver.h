#pragma once

#include <memory>

#include "agent_data.h"

/*
Performs secretion and uptake of cells.
Updates substrate denisities of the cell's voxel and conditionally updates the cell's internalized substrates.

D = substrate densities
I = internalized substrates
S = secretion rates
U = uptake rates
T = saturation densities
N = net export rates
c = cell_volume
v = voxel_volume

I -= v((-(c/v)*dt*(U+S)*D + (c/v)*dt*S*T) / (1 + (c/v)*dt*(U+S)) + (1/v)dt*N)

Updating substrate densities is more complex, one has to take care about the case when we have more cells in the same
voxel. The following formula is used:

D = (D + sum_k{(c_k/v)*dt*S_k*T_k}) / (1 + sum_k{(c_k/v)*dt*(U_k+S_k)}) + sum_k{(1/v)*dt*N_k}

where sum is over all cells in the voxel.

Also handles release of internalized substrates:

F = fraction released at death

D = D + I*F/v
*/

namespace biofvm {

class cell_solver
{
	bool compute_internalized_substrates_;

	std::vector<real_t> numerators_;
	std::vector<real_t> denominators_;
	std::vector<real_t> factors_;

	std::unique_ptr<index_t[]> ballots_;

	void resize(const microenvironment& m);

public:
	void initialize(const microenvironment& m);

	void simulate_secretion_and_uptake(microenvironment& m, bool recompute);

	void release_internalized_substrates(microenvironment& m, index_t index);
};

} // namespace biofvm
