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

Then, the forward substitution is as follows (n multiplication + n subtractions):
d_i'  == d_i - a_i*b_i*d_(i-1)                                1 <  i <= n
The backpropagation (2n multiplication + n subtractions):
d_n'' == d_n'/b_n'
d_i'' == (d_i' - c_i*d_(i+1)'')*b_i'                          n >  i >= 1

Optimizations:
- Each dimension swipe handles also dirichlet boundary conditions.
- For small 2D problems, the whole swipes are put into shared memory.
- For large 2D problems, the swipes are divided into blocks, which are computed independently according to the Modified
Thomas Algorithm (all constants precomputed) [PaScaL_TDMA: A library of parallel and scalable solvers for massive
tridiagonal systems].
*/

namespace biofvm {
namespace solvers {
namespace device {

class diffusion_solver : opencl_solver
{
	cl::Buffer bx_, cx_;
	cl::Buffer by_, cy_;
	cl::Buffer bz_, cz_;

	cl::Buffer a_def_x_, r_fwd_x_, c_fwd_x_, a_bck_x_, c_bck_x_, c_rdc_x_, r_rdc_x_;
	cl::Buffer a_def_y_, r_fwd_y_, c_fwd_y_, a_bck_y_, c_bck_y_, c_rdc_y_, r_rdc_y_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, index_t,
					  index_t, index_t>
		solve_slice_2d_x_, solve_slice_2d_y_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
					  cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, index_t, index_t, index_t, index_t>
		solve_slice_2d_x_block_, solve_slice_2d_y_block_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, index_t,
					  index_t, index_t, index_t>
		solve_slice_2d_x_shared_, solve_slice_2d_y_shared_;

	cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, index_t,
					  index_t, index_t, index_t>
		solve_slice_3d_x_, solve_slice_3d_y_, solve_slice_3d_z_;

	bool x_shared_optim_, y_shared_optim_;
	cl::NDRange x_global_, x_local_, y_global_, y_local_, z_global_, z_local_;

	void precompute_values(cl::Buffer& b, cl::Buffer& c, index_t shape, index_t dims, index_t n,
						   const microenvironment& m);

	void precompute_values_modified_thomas(cl::Buffer& a, cl::Buffer& r_fwd, cl::Buffer& c_fwd, cl::Buffer& a_bck,
										   cl::Buffer& c_bck, cl::Buffer& c_rdc, cl::Buffer& r_rdc, index_t shape,
										   index_t dims, index_t n, index_t block_size, const microenvironment& m);

	void prepare_2d_kernels(microenvironment& m);
	void prepare_3d_kernel(microenvironment& m, cl::Buffer& b, cl::Buffer& c, cl::Kernel kernel, cl::NDRange& global,
						   cl::NDRange& local, index_t dim);

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
