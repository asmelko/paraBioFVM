#pragma once

#include "cuda_solver.h"

/*
This solver applies Dirichlet boundary conditions to the microenvironment.
I.e., it sets the values of the substrate concentrations at specific voxels to a constant value.

Implementation works with 3 arrays:
m.dirichlet_voxels - array of dirichlet voxel (1D/2D/3D) indices
m.dirichlet_conditions - array of bools specifying if a substrate of a dirichlet voxel has a dirichled codition
m.dirichlet_values - array of dirichlet values for each substrate with a dirichlet condition

This solver applies only the interior dirichlet conditions. The boundary conditions are handled by the diffusion solver.
*/

namespace biofvm {
namespace solvers {
namespace device {

class dirichlet_solver : cuda_solver
{
public:
	std::array<real_t*, 3> dirichlet_min_boundary_values;
	std::array<real_t*, 3> dirichlet_max_boundary_values;
	std::array<bool*, 3> dirichlet_min_boundary_conditions;
	std::array<bool*, 3> dirichlet_max_boundary_conditions;

	index_t dirichlet_interior_voxels_count;
	index_t* dirichlet_interior_voxels;
	real_t* dirichlet_interior_values;
	bool* dirichlet_interior_conditions;

	dirichlet_solver(device_context& ctx);

	void initialize(microenvironment& m);

	void solve_2d(microenvironment& m);
	void solve_3d(microenvironment& m);

	void solve(microenvironment& m);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
