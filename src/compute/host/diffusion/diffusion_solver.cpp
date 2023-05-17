#include "diffusion_solver.h"

#include <iostream>

#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures_extended.hpp>

#include "../../../traits.h"
#include "../dirichlet/dirichlet_solver.h"

void diffusion_solver::initialize(microenvironment& m)
{
	if (m.mesh.dims >= 1)
		precompute_values(bx_, cx_, ex_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);
	if (m.mesh.dims >= 2)
		precompute_values(by_, cy_, ey_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
	if (m.mesh.dims >= 3)
		precompute_values(bz_, cz_, ez_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);
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

template <char swipe_dim, typename density_layout_t>
void solve_slice(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
				 const real_t* __restrict__ e, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<swipe_dim>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::sized_vector<'s'>(substrates_count) ^ noarr::sized_vector<'x'>(n);

	for (index_t i = 1; i < n; i++)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				- (diag_l | noarr::get_at<'x', 's'>(e, i - 1, s))
					  * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i - 1, s));
		}
	}

	for (index_t s = 0; s < substrates_count; s++)
	{
		(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s)) =
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s))
			* (diag_l | noarr::get_at<'x', 's'>(b, n - 1, s));
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				 - c[s] * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'x', 's'>(b, i, s));
		}
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
	for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		solve_slice<'y'>(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(), dens_l ^ noarr::fix<'x'>(x));

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
	for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
	{
		for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		{
			solve_slice<'y'>(m.substrate_densities.get(), by_.get(), cy_.get(), ey_.get(),
							 dens_l ^ noarr::fix<'x'>(x) ^ noarr::fix<'z'>(z));
		}
	}

	dirichlet_solver::solve_3d(m);

	// swipe z
	for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
	{
		for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		{
			solve_slice<'z'>(m.substrate_densities.get(), bz_.get(), cz_.get(), ez_.get(),
							 dens_l ^ noarr::fix<'x'>(x) ^ noarr::fix<'y'>(y));
		}
	}

	dirichlet_solver::solve_3d(m);
}
