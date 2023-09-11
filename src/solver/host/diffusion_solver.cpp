#include "diffusion_solver.h"

#include <iostream>

#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures_extended.hpp>

#include "../../traits.h"

using namespace biofvm;
using namespace solvers::host;

index_t lcm(index_t a, index_t b)
{
	index_t high = std::max(a, b);
	index_t low = std::min(a, b);

	index_t ret = high;

	while (ret % low != 0)
		ret += high;

	return ret;
}

void diffusion_solver::initialize(microenvironment& m, dirichlet_solver&) { initialize(m); }

void diffusion_solver::initialize(microenvironment& m)
{
	// Here we try to find the least common multiple of substrates size in bits and size of a vector register (we assume
	// 512 bits). Thanks to this factor, we can reorganize loops in diffusion so they are automatically vectorized.
	// Finding least common multiple is the most optimal, as the loop has no remainder wrt vector registers.
	// But we want to limit it - when substrates size is much higher than 512 bits, multiplying it will not benefit much

	index_t register_size = 512;
	index_t substrates_size = m.substrates_count * sizeof(real_t) * 8;
	index_t multiple = lcm(substrates_size, register_size);

	while (multiple > 10 * register_size && multiple > substrates_size)
		multiple -= substrates_size;

	initialize(m, multiple / substrates_size);
}

void diffusion_solver::initialize(microenvironment& m, index_t substrate_factor)
{
	substrate_factor_ = substrate_factor;

	if (m.mesh.dims >= 1)
		precompute_values(bx_, cx_, ex_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m, 1);
	if (m.mesh.dims >= 2)
		precompute_values(by_, cy_, ey_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m,
						  substrate_factor_);
	if (m.mesh.dims >= 3)
		precompute_values(bz_, cz_, ez_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m,
						  substrate_factor_);
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
										 const microenvironment& m, index_t copies)
{
	if (n == 1) // special case
	{
		b = std::make_unique<real_t[]>(m.substrates_count * copies);

		for (index_t x = 0; x < copies; x++)
			for (index_t s = 0; s < m.substrates_count; s++)
				b[x * m.substrates_count + s] = 1 / (1 + m.decay_rates[s] * m.diffusion_time_step / dims);

		return;
	}

	b = std::make_unique<real_t[]>(n * m.substrates_count * copies);
	e = std::make_unique<real_t[]>((n - 1) * m.substrates_count * copies);
	c = std::make_unique<real_t[]>(m.substrates_count * copies);

	auto layout = noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>() ^ noarr::vector<'i'>()
				  ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'x'>(copies)
				  ^ noarr::set_length<'s'>(m.substrates_count);

	auto b_diag = noarr::make_bag(layout, b.get());
	auto e_diag = noarr::make_bag(layout, e.get());

	// compute c_i
	for (index_t x = 0; x < copies; x++)
		for (index_t s = 0; s < m.substrates_count; s++)
			c[x * m.substrates_count + s] = -m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < m.substrates_count; s++)
					b_diag.at<'i', 'x', 's'>(i, x, s) =
						1 + m.decay_rates[s] * m.diffusion_time_step / dims
						+ m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < m.substrates_count; s++)
					b_diag.at<'i', 'x', 's'>(i, x, s) =
						1 + m.decay_rates[s] * m.diffusion_time_step / dims
						+ 2 * m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i' and e_i
	{
		for (index_t x = 0; x < copies; x++)
			for (index_t s = 0; s < m.substrates_count; s++)
				b_diag.at<'i', 'x', 's'>(0, x, s) = 1 / b_diag.at<'i', 'x', 's'>(0, x, s);

		for (index_t i = 1; i < n; i++)
			for (index_t x = 0; x < copies; x++)
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

index_t diffusion_solver::precompute_values_modified_thomas(
	std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& r_fwd, std::unique_ptr<real_t[]>& c_fwd,
	std::unique_ptr<real_t[]>& a_bck, std::unique_ptr<real_t[]>& c_bck, std::unique_ptr<real_t[]>& c_rdc,
	std::unique_ptr<real_t[]>& r_rdc, index_t shape, index_t dims, index_t n, index_t block_size,
	const microenvironment& m)
{
	auto b = std::make_unique<real_t[]>(n * m.substrates_count);
	a = std::make_unique<real_t[]>(m.substrates_count);

	auto layout = noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'i'>() ^ noarr::set_length<'i'>(n)
				  ^ noarr::set_length<'s'>(m.substrates_count);

	auto b_diag = noarr::make_bag(layout, b.get());

	// compute a_i
	for (index_t s = 0; s < m.substrates_count; s++)
		a[s] = -m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t s = 0; s < m.substrates_count; s++)
				b_diag.at<'i', 's'>(i, s) = 1 + m.decay_rates[s] * m.diffusion_time_step / dims
											+ m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t s = 0; s < m.substrates_count; s++)
				b_diag.at<'i', 's'>(i, s) = 1 + m.decay_rates[s] * m.diffusion_time_step / dims
											+ 2 * m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);
	}


	a_bck = std::make_unique<real_t[]>(n * m.substrates_count);
	c_fwd = std::make_unique<real_t[]>(n * m.substrates_count);
	c_bck = std::make_unique<real_t[]>(n * m.substrates_count);
	r_fwd = std::make_unique<real_t[]>(n * m.substrates_count);

	auto a_fwd_diag = noarr::make_bag(layout, a_bck.get());
	auto c_fwd_diag = noarr::make_bag(layout, c_fwd.get());
	auto c_bck_diag = noarr::make_bag(layout, c_bck.get());
	auto r_fwd_diag = noarr::make_bag(layout, r_fwd.get());

	index_t blocks = (n + block_size - 1) / block_size;

	{
		for (index_t block_i = 0; block_i < blocks; block_i++)
		{
			const index_t block_begin = block_i * block_size;
			const index_t block_end = std::min(n, block_i * block_size + block_size);

			for (index_t idx = block_begin; idx < block_end; idx++)
			{
				if (idx - block_begin < 2)
				{
					for (index_t s = 0; s < m.substrates_count; s++)
					{
						a_fwd_diag.at<'i', 's'>(idx, s) = a[s] / b_diag.at<'i', 's'>(idx, s);
						c_fwd_diag.at<'i', 's'>(idx, s) = a[s] / b_diag.at<'i', 's'>(idx, s);
						r_fwd_diag.at<'i', 's'>(idx, s) = 1 / b_diag.at<'i', 's'>(idx, s);
					}
				}
				else
				{
					for (index_t s = 0; s < m.substrates_count; s++)
					{
						r_fwd_diag.at<'i', 's'>(idx, s) =
							1 / (b_diag.at<'i', 's'>(idx, s) - a[s] * c_fwd_diag.at<'i', 's'>(idx - 1, s));
						c_fwd_diag.at<'i', 's'>(idx, s) = a[s] * r_fwd_diag.at<'i', 's'>(idx, s);
						a_fwd_diag.at<'i', 's'>(idx, s) =
							-a[s] * a_fwd_diag.at<'i', 's'>(idx - 1, s) * r_fwd_diag.at<'i', 's'>(idx, s);
					}
				}
			}

			if (block_end - block_begin <= 2)
				continue;

			for (index_t idx = block_end - 1; idx >= block_begin; idx--)
			{
				if (idx >= block_end - 2)
				{
					for (index_t s = 0; s < m.substrates_count; s++)
						c_bck_diag.at<'i', 's'>(idx, s) = c_fwd_diag.at<'i', 's'>(idx, s);
				}
				else if (idx != block_begin)
				{
					for (index_t s = 0; s < m.substrates_count; s++)
					{
						a_fwd_diag.at<'i', 's'>(idx, s) =
							a_fwd_diag.at<'i', 's'>(idx, s)
							- c_fwd_diag.at<'i', 's'>(idx, s) * a_fwd_diag.at<'i', 's'>(idx + 1, s);
						c_bck_diag.at<'i', 's'>(idx, s) =
							-c_fwd_diag.at<'i', 's'>(idx, s) * c_bck_diag.at<'i', 's'>(idx + 1, s);
					}
				}
				else
				{
					for (index_t s = 0; s < m.substrates_count; s++)
					{
						real_t r = 1 / (1 - a_fwd_diag.at<'i', 's'>(idx + 1, s) * c_fwd_diag.at<'i', 's'>(idx, s));
						c_bck_diag.at<'i', 's'>(idx, s) =
							-r * c_fwd_diag.at<'i', 's'>(idx, s) * c_bck_diag.at<'i', 's'>(idx + 1, s);
						a_fwd_diag.at<'i', 's'>(idx, s) *= r;
					}
				}
			}
		}
	}

	r_rdc = std::make_unique<real_t[]>(blocks * 2 * m.substrates_count);
	c_rdc = std::make_unique<real_t[]>(blocks * 2 * m.substrates_count);

	auto r_rdc_diag = noarr::make_bag(layout, r_rdc.get());
	auto c_rdc_diag = noarr::make_bag(layout, c_rdc.get());

	{
		auto a_rdc = std::make_unique<real_t[]>(blocks * 2 * m.substrates_count);
		auto a_rdc_diag = noarr::make_bag(layout, a_rdc.get());

		for (index_t block_i = 0; block_i < blocks; block_i++)
		{
			index_t idx = block_i * block_size;
			for (index_t s = 0; s < m.substrates_count; s++)
			{
				a_rdc_diag.at<'i', 's'>(block_i * 2, s) = a_fwd_diag.at<'i', 's'>(idx, s);
				c_rdc_diag.at<'i', 's'>(block_i * 2, s) = c_bck_diag.at<'i', 's'>(idx, s);

				if (idx + block_size - 1 < n)
				{
					a_rdc_diag.at<'i', 's'>(block_i * 2 + 1, s) = a_fwd_diag.at<'i', 's'>(idx + block_size - 1, s);
					c_rdc_diag.at<'i', 's'>(block_i * 2 + 1, s) = c_bck_diag.at<'i', 's'>(idx + block_size - 1, s);
				}
			}
		}

		int skip = n - ((blocks - 1) * block_size) == 1 ? 1 : 0;

		for (index_t i = 1; i < blocks * 2 - skip; i++)
		{
			for (index_t s = 0; s < m.substrates_count; s++)
			{
				r_rdc_diag.at<'i', 's'>(i, s) =
					1 / (1 - a_rdc_diag.at<'i', 's'>(i, s) * c_rdc_diag.at<'i', 's'>(i - 1, s));
				c_rdc_diag.at<'i', 's'>(i, s) *= r_rdc_diag.at<'i', 's'>(i, s);
			}
		}
	}

	return blocks;
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

template <char swipe_dim, typename density_layout_t>
void solve_slice_omp(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					 const real_t* __restrict__ e, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<swipe_dim>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'s'>(substrates_count) ^ noarr::sized_vector<'i'>(n);

#pragma omp for
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t i = 1; i < n; i++)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
					  * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i - 1, s));
		}
	}

#pragma omp for
	for (index_t s = 0; s < substrates_count; s++)
	{
		(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s)) =
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s))
			* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
	}

#pragma omp for
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t i = n - 2; i >= 0; i--)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				 - c[s] * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
		}
	}
}

template <char swipe_dim, typename density_layout_t>
void solve_slice_yz(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					const real_t* __restrict__ e, const density_layout_t dens_l_orig, const index_t copy_dim)
{
	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<swipe_dim>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * copy_dim)
				  ^ noarr::sized_vector<swipe_dim>(n);

	auto c_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * copy_dim);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>()
				  ^ noarr::into_blocks_static<'X', 'b', 'x', 'X'>(substrates_count * copy_dim);

	noarr::traverser(dens_l).order(noarr::shift<swipe_dim>(noarr::lit<1>)).for_each([=](auto state) {
		auto prev_state = noarr::neighbor<swipe_dim>(state, -1);
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state))
			- (diag_l | noarr::get_at(e, prev_state)) * (dens_l | noarr::get_at(densities, prev_state));
	});

	noarr::traverser(dens_l).order(noarr::fix<swipe_dim>(n - 1)).for_each([=](auto state) {
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state)) * (diag_l | noarr::get_at(b, state));
	});

	for (index_t i = n - 2; i >= 0; i--)
	{
		noarr::traverser(dens_l).order(noarr::fix<swipe_dim>(i)).for_each([=](auto state) {
			auto next_state = noarr::neighbor<swipe_dim>(state, 1);
			(dens_l | noarr::get_at(densities, state)) =
				((dens_l | noarr::get_at(densities, state))
				 - (c_l | noarr::get_at(c, state)) * (dens_l | noarr::get_at(densities, next_state)))
				* (diag_l | noarr::get_at(b, state));
		});
	}
}

template <char swipe_dim, typename density_layout_t>
void solve_slice_yz_omp(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
						const real_t* __restrict__ e, const density_layout_t dens_l_orig, const index_t copy_dim)
{
	constexpr char para_dim = swipe_dim == 'y' ? 'x' : 'y';

	const index_t substrates_count = dens_l_orig | noarr::get_length<'s'>();
	const index_t n = dens_l_orig | noarr::get_length<swipe_dim>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * copy_dim)
				  ^ noarr::sized_vector<swipe_dim>(n);

	auto c_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'X'>(substrates_count * copy_dim);

	auto dens_l = dens_l_orig ^ noarr::merge_blocks<'x', 's', 'X'>()
				  ^ noarr::into_blocks_dynamic<'X', 'x', 'X', 'b'>(substrates_count * copy_dim);

	const index_t p_n = dens_l | noarr::get_length<para_dim>();

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for nowait
		for (index_t j = 0; j < p_n; j++)
		{
			noarr::traverser(dens_l)
				.order(noarr::fix<swipe_dim>(i) ^ noarr::fix<para_dim>(j))
				.for_each([=](auto state) {
					auto prev_state = noarr::neighbor<swipe_dim>(state, -1);
					(dens_l | noarr::get_at(densities, state)) =
						(dens_l | noarr::get_at(densities, state))
						- (diag_l | noarr::get_at(e, prev_state)) * (dens_l | noarr::get_at(densities, prev_state));
				});
		}
	}
#pragma omp barrier

#pragma omp for
	for (index_t j = 0; j < p_n; j++)
	{
		noarr::traverser(dens_l)
			.order(noarr::fix<swipe_dim>(n - 1) ^ noarr::fix<para_dim>(j))
			.for_each([=](auto state) {
				(dens_l | noarr::get_at(densities, state)) =
					(dens_l | noarr::get_at(densities, state)) * (diag_l | noarr::get_at(b, state));
			});
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for nowait
		for (index_t j = 0; j < p_n; j++)
		{
			noarr::traverser(dens_l)
				.order(noarr::fix<swipe_dim>(i) ^ noarr::fix<para_dim>(j))
				.for_each([=](auto state) {
					auto next_state = noarr::neighbor<swipe_dim>(state, 1);
					(dens_l | noarr::get_at(densities, state)) =
						((dens_l | noarr::get_at(densities, state))
						 - (c_l | noarr::get_at(c, state)) * (dens_l | noarr::get_at(densities, next_state)))
						* (diag_l | noarr::get_at(b, state));
				});
		}
	}
#pragma omp barrier
}

void diffusion_solver::solve_1d(microenvironment& m)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	dirichlet_solver::solve_1d(m);

	solve_slice_omp<'x'>(m.substrate_densities.get(), bx_.get(), cx_.get(), ex_.get(), dens_l);

	dirichlet_solver::solve_1d(m);
}

void diffusion_solver::solve_2d(microenvironment& m)
{
	auto dens_l = layout_traits<2>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	dirichlet_solver::solve_2d(m);

	// swipe x
#pragma omp for
	for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
		solve_slice<'x'>(m.substrate_densities.get(), bx_.get(), cx_.get(), ex_.get(), dens_l ^ noarr::fix<'y'>(y));

	dirichlet_solver::solve_2d(m);

	// swipe y
	solve_slice_yz_omp<'y'>(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(), dens_l, substrate_factor_);

	dirichlet_solver::solve_2d(m);
}

void diffusion_solver::solve_3d(microenvironment& m)
{
	auto dens_l = layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	dirichlet_solver::solve_3d(m);

	// swipe x
#pragma omp for
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
#pragma omp for
	for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
	{
		solve_slice_yz<'y'>(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(), dens_l ^ noarr::fix<'z'>(z),
							substrate_factor_);
	}
	dirichlet_solver::solve_3d(m);

	// swipe z
	solve_slice_yz_omp<'z'>(m.substrate_densities.get(), bz_.get(), cz_.get(), ez_.get(), dens_l, substrate_factor_);

	dirichlet_solver::solve_3d(m);
}
