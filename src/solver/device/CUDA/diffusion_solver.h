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

class diffusion_solver : cuda_solver
{
	real_t *bx_, *cx_;
	real_t *by_, *cy_;
	real_t *bz_, *cz_;

	real_t *a_def_x_, *r_fwd_x_, *c_fwd_x_, *a_bck_x_, *c_bck_x_, *c_rdc_x_, *r_rdc_x_;
	real_t *a_def_y_, *r_fwd_y_, *c_fwd_y_, *a_bck_y_, *c_bck_y_, *c_rdc_y_, *r_rdc_y_;
	real_t *a_def_z_, *r_fwd_z_, *c_fwd_z_, *a_bck_z_, *c_bck_z_, *c_rdc_z_, *r_rdc_z_;

	void precompute_values(real_t*& b, real_t*& c, index_t shape, index_t dims, index_t n, const microenvironment& m);

	void precompute_values_mod(real_t*& a_def, real_t*& r_fwd, real_t*& c_fwd, real_t*& a_bck, real_t*& c_bck,
							   real_t*& c_rdc, real_t*& r_rdc, index_t shape, index_t dims, index_t n,
							   index_t block_size, const microenvironment& m);

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
