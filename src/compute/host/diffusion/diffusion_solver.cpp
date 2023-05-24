#include "diffusion_solver.h"

#include <iostream>

#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures_extended.hpp>

#include "../../../traits.h"
#include "../dirichlet/dirichlet_solver.h"

void diffusion_solver::initialize(microenvironment& m)
{
	// this is very rough estimate
	// we help vectorize y and z only if the size of substrates in bits fits to less than 3 AVX512 registers
	// if it fits to more registers, we say that vectorization in number of substrates is sufficient enough
	bool help_vectorize = m.substrates_count * sizeof(real_t) * 8 < 3 * 512;

	initialize(m, help_vectorize);
}

void diffusion_solver::initialize(microenvironment& m, bool help_vectorize_yz)
{
	help_vectorize_yz_ = help_vectorize_yz;

	if (m.mesh.dims >= 1)
		precompute_values(bx_, cx_, ex_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);

	if (help_vectorize_yz_)
	{
		if (m.mesh.dims >= 2)
			precompute_values_times_x(by_, cy_, ey_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
		if (m.mesh.dims >= 3)
			precompute_values_times_x(bz_, cz_, ez_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);
	}
	else
	{
		if (m.mesh.dims >= 2)
			precompute_values(by_, cy_, ey_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
		if (m.mesh.dims >= 3)
			precompute_values(bz_, cz_, ez_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);
	}
}

void diffusion_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 1)
		solve_1d(m);

	if (m.mesh.dims == 2)
		solve_2d(m);

	if (m.mesh.dims == 3)
		solve_3d(m);
}

void diffusion_solver::precompute_values(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c,
										 std::unique_ptr<real_t[]>& e, index_t shape, index_t dims, index_t n,
										 const microenvironment& m)
{
	if (n == 1) // special case
	{
		b = std::make_unique<real_t[]>(m.substrates_count);
		for (index_t s = 0; s < m.substrates_count; s++)
			b[s] = 1 / (1 + m.decay_rates[s] * m.time_step / dims);

		return;
	}

	b = std::make_unique<real_t[]>(n * m.substrates_count);
	e = std::make_unique<real_t[]>((n - 1) * m.substrates_count);
	c = std::make_unique<real_t[]>(m.substrates_count);

	auto layout = noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'i'>() ^ noarr::set_length<'i'>(n)
				  ^ noarr::set_length<'s'>(m.substrates_count);

	auto b_diag = noarr::make_bag(layout, b.get());
	auto e_diag = noarr::make_bag(layout, e.get());

	// compute c_i
	for (index_t s = 0; s < m.substrates_count; s++)
	{
		c[s] = -m.time_step * m.diffustion_coefficients[s] / (shape * shape);
	}

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
		{
			for (index_t s = 0; s < m.substrates_count; s++)
				b_diag.at<'i', 's'>(i, s) = 1 + m.decay_rates[s] * m.time_step / dims
											+ m.time_step * m.diffustion_coefficients[s] / (shape * shape);
		}

		for (index_t i = 1; i < n - 1; i++)
		{
			for (index_t s = 0; s < m.substrates_count; s++)
			{
				b_diag.at<'i', 's'>(i, s) = 1 + m.decay_rates[s] * m.time_step / dims
											+ 2 * m.time_step * m.diffustion_coefficients[s] / (shape * shape);
			}
		}
	}

	// compute b_i' and e_i
	{
		for (index_t s = 0; s < m.substrates_count; s++)
			b_diag.at<'i', 's'>(0, s) = 1 / b_diag.at<'i', 's'>(0, s);

		for (index_t i = 1; i < n; i++)
		{
			for (index_t s = 0; s < m.substrates_count; s++)
			{
				b_diag.at<'i', 's'>(i, s) =
					1 / (b_diag.at<'i', 's'>(i, s) - c[s] * c[s] * b_diag.at<'i', 's'>(i - 1, s));

				e_diag.at<'i', 's'>(i - 1, s) = c[s] * b_diag.at<'i', 's'>(i - 1, s);
			}
		}
	}
}

void diffusion_solver::precompute_values_times_x(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c,
												 std::unique_ptr<real_t[]>& e, index_t shape, index_t dims, index_t n,
												 const microenvironment& m)
{
	const index_t r_dim = 8; //(10 * 512) / (m.substrates_count * sizeof(real_t));

	if (n == 1) // special case
	{
		b = std::make_unique<real_t[]>(m.substrates_count * r_dim);

		for (index_t x = 0; x < r_dim; x++)
			for (index_t s = 0; s < m.substrates_count; s++)
				b[x * m.substrates_count + s] = 1 / (1 + m.decay_rates[s] * m.time_step / dims);

		return;
	}

	b = std::make_unique<real_t[]>(n * m.substrates_count * r_dim);
	e = std::make_unique<real_t[]>((n - 1) * m.substrates_count * r_dim);
	c = std::make_unique<real_t[]>(m.substrates_count * r_dim);

	auto layout = noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>() ^ noarr::vector<'i'>()
				  ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'x'>(r_dim)
				  ^ noarr::set_length<'s'>(m.substrates_count);

	auto b_diag = noarr::make_bag(layout, b.get());
	auto e_diag = noarr::make_bag(layout, e.get());

	// compute c_i
	for (index_t x = 0; x < r_dim; x++)
		for (index_t s = 0; s < m.substrates_count; s++)
			c[x * m.substrates_count + s] = -m.time_step * m.diffustion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t x = 0; x < r_dim; x++)
				for (index_t s = 0; s < m.substrates_count; s++)
					b_diag.at<'i', 'x', 's'>(i, x, s) = 1 + m.decay_rates[s] * m.time_step / dims
														+ m.time_step * m.diffustion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t x = 0; x < r_dim; x++)
				for (index_t s = 0; s < m.substrates_count; s++)
					b_diag.at<'i', 'x', 's'>(i, x, s) =
						1 + m.decay_rates[s] * m.time_step / dims
						+ 2 * m.time_step * m.diffustion_coefficients[s] / (shape * shape);
	}

	// compute b_i' and e_i
	{
		for (index_t x = 0; x < r_dim; x++)
			for (index_t s = 0; s < m.substrates_count; s++)
				b_diag.at<'i', 'x', 's'>(0, x, s) = 1 / b_diag.at<'i', 'x', 's'>(0, x, s);

		for (index_t i = 1; i < n; i++)
			for (index_t x = 0; x < r_dim; x++)
				for (index_t s = 0; s < m.substrates_count; s++)
				{
					b_diag.at<'i', 'x', 's'>(i, x, s) =
						1
						/ (b_diag.at<'i', 'x', 's'>(i, x, s)
						   - c[x * m.substrates_count + s] * c[x * m.substrates_count + s]
								 * b_diag.at<'i', 'x', 's'>(i - 1, x, s));

					e_diag.at<'i', 'x', 's'>(i - 1, x, s) =
						c[x * m.substrates_count + s] * b_diag.at<'i', 'x', 's'>(i - 1, x, s);
				}
	}
}

template <char swipe_dim, typename density_layout_t>
void solve_slice(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
				 const real_t* __restrict__ e, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<swipe_dim>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'s'>(substrates_count) ^ noarr::sized_vector<'i'>(n);

	for (index_t i = 1; i < n; i++)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
					  * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i - 1, s));
		}
	}

	for (index_t s = 0; s < substrates_count; s++)
	{
		(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s)) =
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s))
			* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				 - c[s] * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
		}
	}
}

template <typename density_layout_t>
void solve_slice_y(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
				   const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'y'>();
	const index_t x_dim = dens_l_orig | noarr::get_length<'x'>();

	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * x_dim) ^ noarr::sized_vector<'y'>(n);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>();

	for (index_t i = 1; i < n; i++)
	{
		for (index_t x = 0; x < x_dim * substrates_count; x++)
		{
			(dens_l | noarr::get_at<'y', 'X'>(densities, i, x)) =
				(dens_l | noarr::get_at<'y', 'X'>(densities, i, x))
				- (diag_l | noarr::get_at<'y', 'X'>(e, i - 1, x))
					  * (dens_l | noarr::get_at<'y', 'X'>(densities, i - 1, x));
		}
	}
	for (index_t x = 0; x < x_dim * substrates_count; x++)
	{
		(dens_l | noarr::get_at<'y', 'X'>(densities, n - 1, x)) =
			(dens_l | noarr::get_at<'y', 'X'>(densities, n - 1, x)) * (diag_l | noarr::get_at<'y', 'X'>(b, n - 1, x));
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		for (index_t x = 0; x < x_dim * substrates_count; x++)
		{
			(dens_l | noarr::get_at<'y', 'X'>(densities, i, x)) =
				((dens_l | noarr::get_at<'y', 'X'>(densities, i, x))
				 - c[x] * (dens_l | noarr::get_at<'y', 'X'>(densities, i + 1, x)))
				* (diag_l | noarr::get_at<'y', 'X'>(b, i, x));
		}
	}
}

template <typename density_layout_t>
void solve_slice_y2(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'y'>();
	const index_t x_dim = dens_l_orig | noarr::get_length<'x'>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'s'>(substrates_count) ^ noarr::sized_vector<'y'>(n);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>();

	for (index_t i = 1; i < n; i++)
	{
		for (index_t x = 0; x < x_dim * substrates_count; x++)
		{
			(dens_l | noarr::get_at<'y', 'X'>(densities, i, x)) =
				(dens_l | noarr::get_at<'y', 'X'>(densities, i, x))
				- (diag_l | noarr::get_at<'y', 's'>(e, i - 1, x % substrates_count))
					  * (dens_l | noarr::get_at<'y', 'X'>(densities, i - 1, x));
		}
	}
	for (index_t x = 0; x < x_dim * substrates_count; x++)
	{
		(dens_l | noarr::get_at<'y', 'X'>(densities, n - 1, x)) =
			(dens_l | noarr::get_at<'y', 'X'>(densities, n - 1, x))
			* (diag_l | noarr::get_at<'y', 's'>(b, n - 1, x % substrates_count));
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		for (index_t x = 0; x < x_dim * substrates_count; x++)
		{
			(dens_l | noarr::get_at<'y', 'X'>(densities, i, x)) =
				((dens_l | noarr::get_at<'y', 'X'>(densities, i, x))
				 - c[x % substrates_count] * (dens_l | noarr::get_at<'y', 'X'>(densities, i + 1, x)))
				* (diag_l | noarr::get_at<'y', 's'>(b, i, x % substrates_count));
		}
	}
}

template <typename density_layout_t>
void solve_slice_y3(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'y'>();
	const index_t x_dim = dens_l_orig | noarr::get_length<'x'>();

	auto r_dim = 8;

	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * r_dim) ^ noarr::sized_vector<'y'>(n);

	auto dens_l = dens_l_orig ^ noarr::into_blocks<'x', 'x', 'r'>(r_dim) ^ noarr::merge_blocks<'r', 's', 'X'>();

	for (index_t i = 1; i < n; i++)
	{
		for (index_t x = 0; x < x_dim / r_dim; x++)
		{
			for (index_t r = 0; r < r_dim * substrates_count; r++)
				(dens_l | noarr::get_at<'y', 'x', 'X'>(densities, i, x, r)) =
					(dens_l | noarr::get_at<'y', 'x', 'X'>(densities, i, x, r))
					- (diag_l | noarr::get_at<'y', 'X'>(e, i - 1, r))
						  * (dens_l | noarr::get_at<'y', 'x', 'X'>(densities, i - 1, x, r));
		}
	}
	for (index_t x = 0; x < x_dim / r_dim; x++)
	{
		for (index_t r = 0; r < r_dim * substrates_count; r++)
			(dens_l | noarr::get_at<'y', 'x', 'X'>(densities, n - 1, x, r)) =
				(dens_l | noarr::get_at<'y', 'x', 'X'>(densities, n - 1, x, r))
				* (diag_l | noarr::get_at<'y', 'X'>(b, n - 1, r));
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		for (index_t x = 0; x < x_dim / r_dim; x++)
		{
			for (index_t r = 0; r < r_dim * substrates_count; r++)
				(dens_l | noarr::get_at<'y', 'x', 'X'>(densities, i, x, r)) =
					((dens_l | noarr::get_at<'y', 'x', 'X'>(densities, i, x, r))
					 - c[r] * (dens_l | noarr::get_at<'y', 'x', 'X'>(densities, i + 1, x, r)))
					* (diag_l | noarr::get_at<'y', 'X'>(b, i, r));
		}
	}
}

template <typename density_layout_t>
void solve_slice_y4(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'y'>();
	const index_t x_dim = dens_l_orig | noarr::get_length<'x'>();

	auto r_dim = 8;

	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * r_dim) ^ noarr::sized_vector<'y'>(n);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>()
				  ^ noarr::into_blocks_static<'X', 'b', 'x', 'X'>(r_dim * substrates_count);

	for (index_t i = 1; i < n; i++)
	{
		for (index_t x = 0; x < x_dim / r_dim; x++)
		{
			for (index_t r = 0; r < r_dim * substrates_count; r++)
				(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, x, r, noarr::lit<0>)) =
					(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, x, r, noarr::lit<0>))
					- (diag_l | noarr::get_at<'y', 'X'>(e, i - 1, r))
						  * (dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i - 1, x, r, noarr::lit<0>));
		}

		for (index_t r = 0; r < (x_dim % r_dim) * substrates_count; r++)
			(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, 0, r, noarr::lit<1>)) =
				(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, 0, r, noarr::lit<1>))
				- (diag_l | noarr::get_at<'y', 'X'>(e, i - 1, r))
					  * (dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i - 1, 0, r, noarr::lit<1>));
	}

	{
		for (index_t x = 0; x < x_dim / r_dim; x++)
		{
			for (index_t r = 0; r < r_dim * substrates_count; r++)
				(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, n - 1, x, r, noarr::lit<0>)) =
					(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, n - 1, x, r, noarr::lit<0>))
					* (diag_l | noarr::get_at<'y', 'X'>(b, n - 1, r));
		}

		for (index_t r = 0; r < (x_dim % r_dim) * substrates_count; r++)
			(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, n - 1, 0, r, noarr::lit<1>)) =
				(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, n - 1, 0, r, noarr::lit<1>))
				* (diag_l | noarr::get_at<'y', 'X'>(b, n - 1, r));
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		for (index_t x = 0; x < x_dim / r_dim; x++)
		{
			for (index_t r = 0; r < r_dim * substrates_count; r++)
				(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, x, r, noarr::lit<0>)) =
					((dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, x, r, noarr::lit<0>))
					 - c[r] * (dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i + 1, x, r, noarr::lit<0>)))
					* (diag_l | noarr::get_at<'y', 'X'>(b, i, r));
		}

		for (index_t r = 0; r < (x_dim % r_dim) * substrates_count; r++)
			(dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, 0, r, noarr::lit<1>)) =
				((dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i, 0, r, noarr::lit<1>))
				 - c[r] * (dens_l | noarr::get_at<'y', 'x', 'X', 'b'>(densities, i + 1, 0, r, noarr::lit<1>)))
				* (diag_l | noarr::get_at<'y', 'X'>(b, i, r));
	}
}

template <typename density_layout_t>
void solve_slice_y5(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'y'>();

	auto r_dim = 8;

	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * r_dim) ^ noarr::sized_vector<'y'>(n);

	auto c_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * r_dim);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>()
				  ^ noarr::into_blocks_static<'X', 'b', 'x', 'X'>(r_dim * substrates_count);

	noarr::traverser(dens_l).order(noarr::shift<'y'>(noarr::lit<1>)).for_each([&](auto state) {
		auto prev_state = noarr::neighbor<'y'>(state, -1);
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state))
			- (diag_l | noarr::get_at(e, prev_state)) * (dens_l | noarr::get_at(densities, prev_state));
	});

	noarr::traverser(dens_l).order(noarr::fix<'y'>(n - 1)).for_each([&](auto state) {
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state)) * (diag_l | noarr::get_at(b, state));
	});

	for (index_t i = n - 2; i >= 0; i--)
	{
		noarr::traverser(dens_l).order(noarr::fix<'y'>(i)).for_each([&](auto state) {
			auto next_state = noarr::neighbor<'y'>(state, 1);
			(dens_l | noarr::get_at(densities, state)) =
				((dens_l | noarr::get_at(densities, state))
				 - (c_l | noarr::get_at(c, state)) * (dens_l | noarr::get_at(densities, next_state)))
				* (diag_l | noarr::get_at(b, state));
		});
	}
}

template <typename density_layout_t>
void solve_slice_z(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
				   const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'z'>();
	const index_t x_dim = dens_l_orig | noarr::get_length<'x'>();
	const index_t y_dim = dens_l_orig | noarr::get_length<'y'>();

	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * x_dim) ^ noarr::sized_vector<'z'>(n);
	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>();

	for (index_t i = 1; i < n; i++)
		for (index_t y = 0; y < y_dim; y++)
			for (index_t x = 0; x < x_dim * substrates_count; x++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x)) =
					(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x))
					- (diag_l | noarr::get_at<'z', 'X'>(e, i - 1, x))
						  * (dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i - 1, y, x));
			}

	for (index_t y = 0; y < y_dim; y++)
		for (index_t x = 0; x < x_dim * substrates_count; x++)
		{
			(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, n - 1, y, x)) =
				(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, n - 1, y, x))
				* (diag_l | noarr::get_at<'z', 'X'>(b, n - 1, x));
		}

	for (index_t i = n - 2; i >= 0; i--)
		for (index_t y = 0; y < y_dim; y++)
			for (index_t x = 0; x < x_dim * substrates_count; x++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x)) =
					((dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x))
					 - c[x] * (dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i + 1, y, x)))
					* (diag_l | noarr::get_at<'z', 'X'>(b, i, x));
			}
}

template <typename density_layout_t>
void solve_slice_z2(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'z'>();
	const index_t x_dim = dens_l_orig | noarr::get_length<'x'>();
	const index_t y_dim = dens_l_orig | noarr::get_length<'y'>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'s'>(substrates_count) ^ noarr::sized_vector<'z'>(n);
	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>();

	for (index_t i = 1; i < n; i++)
		for (index_t y = 0; y < y_dim; y++)
			for (index_t x = 0; x < x_dim * substrates_count; x++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x)) =
					(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x))
					- (diag_l | noarr::get_at<'z', 's'>(e, i - 1, x % substrates_count))
						  * (dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i - 1, y, x));
			}

	for (index_t y = 0; y < y_dim; y++)
		for (index_t x = 0; x < x_dim * substrates_count; x++)
		{
			(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, n - 1, y, x)) =
				(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, n - 1, y, x))
				* (diag_l | noarr::get_at<'z', 's'>(b, n - 1, x % substrates_count));
		}

	for (index_t i = n - 2; i >= 0; i--)
		for (index_t y = 0; y < y_dim; y++)
			for (index_t x = 0; x < x_dim * substrates_count; x++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x)) =
					((dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i, y, x))
					 - c[x % substrates_count] * (dens_l | noarr::get_at<'z', 'y', 'X'>(densities, i + 1, y, x)))
					* (diag_l | noarr::get_at<'z', 's'>(b, i, x % substrates_count));
			}
}

template <typename density_layout_t>
void solve_slice_z3(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'z'>();
	const index_t x_dim = dens_l_orig | noarr::get_length<'x'>();
	const index_t y_dim = dens_l_orig | noarr::get_length<'y'>();

	auto r_dim = 8;

	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * r_dim) ^ noarr::sized_vector<'z'>(n);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>()
				  ^ noarr::into_blocks_static<'X', 'b', 'x', 'X'>(r_dim * substrates_count);

	for (index_t i = 1; i < n; i++)
		for (index_t y = 0; y < y_dim; y++)
		{
			for (index_t x = 0; x < x_dim / r_dim; x++)
				for (index_t r = 0; r < r_dim * substrates_count; r++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, x, r, noarr::lit<0>)) =
						(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, x, r, noarr::lit<0>))
						- (diag_l | noarr::get_at<'z', 'X'>(e, i - 1, r))
							  * (dens_l
								 | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i - 1, y, x, r, noarr::lit<0>));
				}

			for (index_t r = 0; r < (x_dim % r_dim) * substrates_count; r++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, 0, r, noarr::lit<1>)) =
					(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, 0, r, noarr::lit<1>))
					- (diag_l | noarr::get_at<'z', 'X'>(e, i - 1, r))
						  * (dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i - 1, y, 0, r, noarr::lit<1>));
			}
		}

	for (index_t y = 0; y < y_dim; y++)
	{
		for (index_t x = 0; x < x_dim / r_dim; x++)
			for (index_t r = 0; r < r_dim * substrates_count; r++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, n - 1, y, x, r, noarr::lit<0>)) =
					(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, n - 1, y, x, r, noarr::lit<0>))
					* (diag_l | noarr::get_at<'z', 'X'>(b, n - 1, r));
			}

		for (index_t r = 0; r < (x_dim % r_dim) * substrates_count; r++)
		{
			(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, n - 1, y, 0, r, noarr::lit<1>)) =
				(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, n - 1, y, 0, r, noarr::lit<1>))
				* (diag_l | noarr::get_at<'z', 'X'>(b, n - 1, r));
		}
	}

	for (index_t i = n - 2; i >= 0; i--)
		for (index_t y = 0; y < y_dim; y++)
		{
			for (index_t x = 0; x < x_dim / r_dim; x++)
				for (index_t r = 0; r < r_dim * substrates_count; r++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, x, r, noarr::lit<0>)) =
						((dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, x, r, noarr::lit<0>))
						 - c[r]
							   * (dens_l
								  | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i + 1, y, x, r, noarr::lit<0>)))
						* (diag_l | noarr::get_at<'z', 'X'>(b, i, r));
				}

			for (index_t r = 0; r < (x_dim % r_dim) * substrates_count; r++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, 0, r, noarr::lit<1>)) =
					((dens_l | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i, y, 0, r, noarr::lit<1>))
					 - c[r]
						   * (dens_l
							  | noarr::get_at<'z', 'y', 'x', 'X', 'b'>(densities, i + 1, y, 0, r, noarr::lit<1>)))
					* (diag_l | noarr::get_at<'z', 'X'>(b, i, r));
			}
		}
}

template <typename density_layout_t>
void solve_slice_z5(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<'z'>();

	auto r_dim = 8;

	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * r_dim) ^ noarr::sized_vector<'z'>(n);

	auto c_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * r_dim);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>()
				  ^ noarr::into_blocks_static<'X', 'b', 'x', 'X'>(r_dim * substrates_count);

	noarr::traverser(dens_l).order(noarr::shift<'z'>(noarr::lit<1>)).for_each([&](auto state) {
		auto prev_state = noarr::neighbor<'z'>(state, -1);
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state))
			- (diag_l | noarr::get_at(e, prev_state)) * (dens_l | noarr::get_at(densities, prev_state));
	});

	noarr::traverser(dens_l).order(noarr::fix<'z'>(n - 1)).for_each([&](auto state) {
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state)) * (diag_l | noarr::get_at(b, state));
	});

	for (index_t i = n - 2; i >= 0; i--)
	{
		noarr::traverser(dens_l).order(noarr::fix<'z'>(i)).for_each([&](auto state) {
			auto next_state = noarr::neighbor<'z'>(state, 1);
			(dens_l | noarr::get_at(densities, state)) =
				((dens_l | noarr::get_at(densities, state))
				 - (c_l | noarr::get_at(c, state)) * (dens_l | noarr::get_at(densities, next_state)))
				* (diag_l | noarr::get_at(b, state));
		});
	}
}

void diffusion_solver::solve_1d(microenvironment& m)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	dirichlet_solver::solve_1d(m);

	solve_slice<'x'>(m.substrate_densities.get(), bx_.get(), cx_.get(), ex_.get(), dens_l);

	dirichlet_solver::solve_1d(m);
}

void diffusion_solver::solve_2d(microenvironment& m)
{
	auto dens_l = layout_traits<2>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	dirichlet_solver::solve_2d(m);

	// swipe x
	for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
		solve_slice<'x'>(m.substrate_densities.get(), bx_.get(), cx_.get(), ex_.get(), dens_l ^ noarr::fix<'y'>(y));

	dirichlet_solver::solve_2d(m);

	// swipe y
	if (help_vectorize_yz_)
		solve_slice_y5(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(), dens_l);
	else
	{
		for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
			solve_slice<'y'>(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(), dens_l ^ noarr::fix<'x'>(x));
	}

	dirichlet_solver::solve_2d(m);
}

void diffusion_solver::solve_3d(microenvironment& m)
{
	auto dens_l = layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	dirichlet_solver::solve_3d(m);

	// swipe x
	for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
	{
		for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
		{
			solve_slice<'x'>(m.substrate_densities.get(), bx_.get(), cx_.get(), ex_.get(),
							 dens_l ^ noarr::fix<'y'>(y) ^ noarr::fix<'z'>(z));
		}
	}

	dirichlet_solver::solve_3d(m);

	// swipe y
	if (help_vectorize_yz_)
	{
		for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
		{
			solve_slice_y4(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(), dens_l ^ noarr::fix<'z'>(z));
		}
	}
	else
	{
		for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
		{
			for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
			{
				solve_slice<'y'>(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(),
								 dens_l ^ noarr::fix<'x'>(x) ^ noarr::fix<'z'>(z));
			}
		}
	}

	dirichlet_solver::solve_3d(m);

	// swipe z
	if (help_vectorize_yz_)
		solve_slice_z3(m.substrate_densities.get(), bz_.get(), cz_.get(), ez_.get(), dens_l);
	else
	{
		for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
		{
			for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
			{
				solve_slice<'z'>(m.substrate_densities.get(), bz_.get(), cz_.get(), ez_.get(),
								 dens_l ^ noarr::fix<'x'>(x) ^ noarr::fix<'y'>(y));
			}
		}
	}

	dirichlet_solver::solve_3d(m);
}
