#pragma once

#include "../../../../include/BioFVM/microenvironment.h"
#include "../../common_solver.h"
#include "opencl_solver.h"

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
namespace solvers {
namespace device {

class cell_solver : opencl_solver, common_solver
{
	bool compute_internalized_substrates_;

	std::size_t capacity_;

	cl::Buffer numerators_, denominators_, factors_;
	cl::Buffer reduced_numerators_, reduced_denominators_, reduced_factors_;
	cl::Buffer ballots_;
	cl::Buffer conflicts_, conflicts_wrk_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, index_t,
					  index_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t>
		clear_and_ballot_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
					  cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real_t,
					  real_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t,
					  index_t>
		ballot_and_sum_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real_t, index_t, index_t,
					  index_t, index_t>
		compute_densities_1d_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
					  cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real_t, index_t, index_t, index_t, index_t>
		compute_fused_1d_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real_t, index_t, index_t,
					  index_t, index_t, index_t, index_t, index_t>
		compute_densities_2d_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
					  cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real_t, index_t, index_t, index_t, index_t,
					  index_t, index_t, index_t>
		compute_fused_2d_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real_t, index_t, index_t,
					  index_t, index_t, index_t, index_t, index_t, index_t, index_t, index_t>
		compute_densities_3d_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
					  cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real_t, index_t, index_t, index_t, index_t,
					  index_t, index_t, index_t, index_t, index_t, index_t>
		compute_fused_3d_;


	void resize(microenvironment& m);

	void simulate_1d(microenvironment& m);
	void simulate_2d(microenvironment& m);
	void simulate_3d(microenvironment& m);

	void prepare_kernel_2d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel);
	void prepare_kernel_3d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel);

	void modify_kernel_2d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel);
	void modify_kernel_3d(microenvironment& m, cl::Kernel fused_kernel, cl::Kernel dens_kernel);

public:
	cell_solver(device_context& ctx);

	void initialize(microenvironment& m);

	void simulate_secretion_and_uptake(microenvironment& m, bool recompute);

	void release_internalized_substrates(microenvironment& m, index_t index);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
