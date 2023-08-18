#pragma once

#include "dirichlet_solver.h"

namespace biofvm {
namespace solvers {
namespace device {

class diffusion_solver : opencl_solver
{
	cl::Buffer bx_, cx_, ex_;
	cl::Buffer by_, cy_, ey_;
	cl::Buffer bz_, cz_, ez_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, index_t, index_t, index_t> solve_slice_2d_x_,
		solve_slice_2d_y_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, index_t, index_t, index_t, index_t>
		solve_slice_3d_x_, solve_slice_3d_y_, solve_slice_3d_z_;

	void precompute_values(cl::Buffer& b, cl::Buffer& c, cl::Buffer& e, index_t shape, index_t dims, index_t n,
						   const microenvironment& m);


public:
	diffusion_solver(device_context& ctx);

	dirichlet_solver dirichlet;

	void initialize(microenvironment& m);

	void solve_2d(microenvironment& m);
	void solve_3d(microenvironment& m);

	void solve(microenvironment& m);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
