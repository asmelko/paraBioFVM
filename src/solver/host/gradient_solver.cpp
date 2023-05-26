#include "gradient_solver.h"

#include <omp.h>

#include <noarr/structures_extended.hpp>

#include "../../traits.h"

using namespace biofvm;

template <typename density_layout_t, typename gradient_layout_t>
void solve_x(real_t* __restrict__ gradients, real_t* __restrict__ densities, index_t dim_shape,
			 const density_layout_t dens_l, const gradient_layout_t grad_l)
{
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t s_dim = dens_l | noarr::get_length<'s'>();

	const real_t border_factor = 1.f / dim_shape;
	const real_t center_factor = 1.f / (dim_shape * 2);

	auto grad_l2 = grad_l ^ noarr::merge_blocks<'x', 's', 'X'>();
	auto dens_l2 = dens_l ^ noarr::merge_blocks<'x', 's', 'X'>();

	for (index_t s = 0; s < s_dim; s++)
	{
		(grad_l | noarr::get_at<'s', 'x'>(gradients, s, 0)) =
			((dens_l | noarr::get_at<'s', 'x'>(densities, s, 1)) - (dens_l | noarr::get_at<'s', 'x'>(densities, s, 0)))
			* border_factor;
	}

	for (index_t x = 1 * s_dim; x < (n - 1) * s_dim; x++)
	{
		(grad_l2 | noarr::get_at<'X'>(gradients, x)) = ((dens_l2 | noarr::get_at<'X'>(densities, x + s_dim))
														- (dens_l2 | noarr::get_at<'X'>(densities, x - s_dim)))
													   * center_factor;
	}

	for (index_t s = 0; s < s_dim; s++)
	{
		(grad_l | noarr::get_at<'s', 'x'>(gradients, s, n - 1)) =
			((dens_l | noarr::get_at<'s', 'x'>(densities, s, n - 1))
			 - (dens_l | noarr::get_at<'s', 'x'>(densities, s, n - 2)))
			* border_factor;
	}
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_y(real_t* __restrict__ gradients, real_t* __restrict__ densities, index_t dim_shape,
			 const density_layout_t dens_l, const gradient_layout_t grad_l)
{
	const index_t y_dim = dens_l | noarr::get_length<'y'>();
	const index_t x_dim = dens_l | noarr::get_length<'x'>();
	const index_t s_dim = dens_l | noarr::get_length<'s'>();

	const real_t border_factor = 1.f / dim_shape;
	const real_t center_factor = 1.f / (dim_shape * 2);

	auto grad_l2 = grad_l ^ noarr::merge_blocks<'x', 's', 'X'>();
	auto dens_l2 = dens_l ^ noarr::merge_blocks<'x', 's', 'X'>();

	for (index_t x = 0; x < x_dim * s_dim; x++)
	{
		(grad_l2 | noarr::get_at<'X', 'y'>(gradients, x, 0)) = ((dens_l2 | noarr::get_at<'X', 'y'>(densities, x, 1))
																- (dens_l2 | noarr::get_at<'X', 'y'>(densities, x, 0)))
															   * border_factor;
	}

	for (index_t y = 1; y < y_dim - 1; y++)
	{
		for (index_t x = 0; x < x_dim * s_dim; x++)
		{
			(grad_l2 | noarr::get_at<'X', 'y'>(gradients, x, y)) =
				((dens_l2 | noarr::get_at<'X', 'y'>(densities, x, y + 1))
				 - (dens_l2 | noarr::get_at<'X', 'y'>(densities, x, y - 1)))
				* center_factor;
		}
	}

	for (index_t x = 0; x < x_dim * s_dim; x++)
	{
		(grad_l2 | noarr::get_at<'X', 'y'>(gradients, x, y_dim - 1)) =
			((dens_l2 | noarr::get_at<'X', 'y'>(densities, x, y_dim - 1))
			 - (dens_l2 | noarr::get_at<'X', 'y'>(densities, x, y_dim - 2)))
			* border_factor;
	}
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_z(real_t* __restrict__ gradients, real_t* __restrict__ densities, index_t dim_shape,
			 const density_layout_t dens_l, const gradient_layout_t grad_l)
{
	const index_t z_dim = dens_l | noarr::get_length<'z'>();
	const index_t y_dim = dens_l | noarr::get_length<'y'>();
	const index_t x_dim = dens_l | noarr::get_length<'x'>();
	const index_t s_dim = dens_l | noarr::get_length<'s'>();

	const real_t border_factor = 1.f / dim_shape;
	const real_t center_factor = 1.f / (dim_shape * 2);

	auto grad_l2 = grad_l ^ noarr::merge_blocks<'x', 's', 'X'>();
	auto dens_l2 = dens_l ^ noarr::merge_blocks<'x', 's', 'X'>();

#pragma omp parallel for
	for (index_t y = 0; y < y_dim; y++)
	{
		for (index_t x = 0; x < x_dim * s_dim; x++)
		{
			(grad_l2 | noarr::get_at<'X', 'y', 'z'>(gradients, x, y, 0)) =
				((dens_l2 | noarr::get_at<'X', 'y', 'z'>(densities, x, y, 1))
				 - (dens_l2 | noarr::get_at<'X', 'y', 'z'>(densities, x, y, 0)))
				* border_factor;
		}
	}

#pragma omp parallel for
	for (index_t z = 1; z < z_dim - 1; z++)
	{
		for (index_t y = 0; y < y_dim; y++)
		{
			for (index_t x = 0; x < x_dim * s_dim; x++)
			{
				(grad_l2 | noarr::get_at<'X', 'y', 'z'>(gradients, x, y, z)) =
					((dens_l2 | noarr::get_at<'X', 'y', 'z'>(densities, x, y, z + 1))
					 - (dens_l2 | noarr::get_at<'X', 'y', 'z'>(densities, x, y, z - 1)))
					* center_factor;
			}
		}
	}

#pragma omp parallel for
	for (index_t y = 0; y < y_dim; y++)
	{
		for (index_t x = 0; x < x_dim * s_dim; x++)
		{
			(grad_l2 | noarr::get_at<'X', 'y', 'z'>(gradients, x, y, z_dim - 1)) =
				((dens_l2 | noarr::get_at<'X', 'y', 'z'>(densities, x, y, z_dim - 1))
				 - (dens_l2 | noarr::get_at<'X', 'y', 'z'>(densities, x, y, z_dim - 2)))
				* border_factor;
		}
	}
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_3d_internal(real_t* __restrict__ gradients, real_t* __restrict__ densities, point_t<index_t, 3> voxel_shape,
					   const density_layout_t dens_l, const gradient_layout_t grad_l)
{
	const index_t z_dim = dens_l | noarr::get_length<'z'>();
	const index_t y_dim = dens_l | noarr::get_length<'y'>();

#pragma omp teams num_teams(3)
	{
		if (omp_get_team_num() == 0)
		{
#pragma omp parallel for
			for (index_t z = 0; z < z_dim; z++)
			{
				for (index_t y = 0; y < y_dim; y++)
				{
					solve_x(gradients, densities, voxel_shape[0], dens_l ^ noarr::fix<'y', 'z'>(y, z),
							grad_l ^ noarr::fix<'y', 'z', 'd'>(y, z, noarr::lit<0>));
				}
			}
		}

		if (omp_get_team_num() == 1)
		{
#pragma omp parallel for
			for (index_t z = 0; z < z_dim; z++)
			{
				solve_y(gradients, densities, voxel_shape[1], dens_l ^ noarr::fix<'z'>(z),
						grad_l ^ noarr::fix<'z', 'd'>(z, noarr::lit<1>));
			}
		}

		if (omp_get_team_num() == 2)
		{
			solve_z(gradients, densities, voxel_shape[2], dens_l, grad_l ^ noarr::fix<'d'>(noarr::lit<2>));
		}
	}
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_2d_internal(real_t* __restrict__ gradients, real_t* __restrict__ densities, point_t<index_t, 3> voxel_shape,
					   const density_layout_t dens_l, const gradient_layout_t grad_l)
{
	const index_t y_dim = dens_l | noarr::get_length<'y'>();

	for (index_t y = 0; y < y_dim; y++)
	{
		solve_x(gradients, densities, voxel_shape[0], dens_l ^ noarr::fix<'y'>(y),
				grad_l ^ noarr::fix<'y', 'd'>(y, noarr::lit<0>));
	}

	solve_y(gradients, densities, voxel_shape[1], dens_l, grad_l ^ noarr::fix<'d'>(noarr::lit<1>));
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_1d_internal(real_t* __restrict__ gradients, real_t* __restrict__ densities, point_t<index_t, 3> voxel_shape,
					   const density_layout_t dens_l, const gradient_layout_t grad_l)
{
	solve_x(gradients, densities, voxel_shape[0], dens_l, grad_l ^ noarr::fix<'d'>(noarr::lit<0>));
}

void gradient_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 1)
		solve_1d_internal(m.gradients.get(), m.substrate_densities.get(), m.mesh.voxel_shape,
						  layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape),
						  layout_traits<1>::construct_gradient_layout(m.substrates_count, m.mesh.grid_shape));
	else if (m.mesh.dims == 2)
		solve_2d_internal(m.gradients.get(), m.substrate_densities.get(), m.mesh.voxel_shape,
						  layout_traits<2>::construct_density_layout(m.substrates_count, m.mesh.grid_shape),
						  layout_traits<2>::construct_gradient_layout(m.substrates_count, m.mesh.grid_shape));
	else if (m.mesh.dims == 3)
		solve_3d_internal(m.gradients.get(), m.substrate_densities.get(), m.mesh.voxel_shape,
						  layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape),
						  layout_traits<3>::construct_gradient_layout(m.substrates_count, m.mesh.grid_shape));
}

void gradient_solver::initialize(microenvironment&) {}
