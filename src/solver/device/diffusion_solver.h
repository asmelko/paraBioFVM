#pragma once

#include "dirichlet_solver.h"

/*
The diffusion is the problem of solving tridiagonal matrix system with these coeficients:
For dimension x:
a_i  == -dt*diffusion_coefs/dx^2                              1 <= i <= n
b_1  == 1 + dt*decay_rates/dims + dt*diffusion_coefs/dx^2
b_i  == 1 + dt*decay_rates/dims + 2*dt*diffusion_coefs/dx^2   1 <  i <  n
b_n  == 1 + dt*decay_rates/dims + dt*diffusion_coefs/dx^2
c_i  == -dt*diffusion_coefs/dx^2                              1 <= i <= n
d_i  == current diffusion rates
For dimension y/z (if they exist):
substitute dx accordingly to dy/dz

Since the matrix is constant for multiple right hand sides, we precompute its values in the following way:
b_1'  == 1/b_1
b_i'  == 1/(b_i - a_i*c_i*b_(i-1)')                           1 <  i <= n
e_i   == a_i*b_(i-1)'                                         1 <  i <= n

Then, the forward substitution is as follows (n multiplication + n subtractions):
d_i'  == d_i - e_i*d_(i-1)                                    1 <  i <= n
The backpropagation (2n multiplication + n subtractions):
d_n'' == d_n'/b_n'
d_i'' == (d_i' - c_i*d_(i+1)'')*b_i'                          n >  i >= 1

Optimizations:
- The memory accesses of all dimension slices are continuous for the least amount of cache misses. This is done by
reordering the loops for the thomas solver of y and z axes.
*/

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
