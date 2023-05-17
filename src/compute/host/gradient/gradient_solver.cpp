#include "gradient_solver.h"

#include <noarr/structures_extended.hpp>

#include "../../../traits.h"

template <char dim, index_t dim_idx, typename density_layout_t, typename gradient_layout_t>
void solve_single(real_t* __restrict__ gradients, real_t* __restrict__ densities, index_t dim_shape,
				  const density_layout_t dens_l, const gradient_layout_t grad_l, index_t index)
{
	const index_t dim_size = dens_l | noarr::get_length<dim>();

	const index_t up = index + (index == dim_size - 1 ? 0 : 1);
	const index_t down = index - (index == 0 ? 0 : 1);
	const real_t divisor = dim_shape + (index == 0 || (index == dim_size - 1) ? 0 : dim_shape);

	(grad_l | noarr::get_at<'d'>(gradients, noarr::lit<dim_idx>)) =
		((dens_l | noarr::get_at<dim>(densities, up)) - (dens_l | noarr::get_at<dim>(densities, down))) / divisor;
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_3d_internal(real_t* __restrict__ gradients, real_t* __restrict__ densities, point_t<index_t, 3> voxel_shape,
					   const density_layout_t dens_l, const gradient_layout_t grad_l)
{
    const index_t z_dim = dens_l | noarr::get_length<'z'>();
    const index_t y_dim = dens_l | noarr::get_length<'y'>();
    const index_t x_dim = dens_l | noarr::get_length<'x'>();
    const index_t s_dim = dens_l | noarr::get_length<'s'>();

	for (index_t z = 0; z < z_dim; z++)
	{
		for (index_t y = 0; y < y_dim; y++)
		{
			for (index_t x = 0; x < x_dim; x++)
			{
				for (index_t s = 0; s < s_dim; s++)
				{
					solve_single<'x', 0>(gradients, densities, voxel_shape[0],
										 dens_l ^ noarr::fix<'s', 'y', 'z'>(s, y, z),
										 grad_l ^ noarr::fix<'s', 'x', 'y', 'z'>(s, x, y, z), x);

					solve_single<'y', 1>(gradients, densities, voxel_shape[1],
										 dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z),
										 grad_l ^ noarr::fix<'s', 'x', 'y', 'z'>(s, x, y, z), y);

					solve_single<'z', 2>(gradients, densities, voxel_shape[2],
										 dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y),
										 grad_l ^ noarr::fix<'s', 'x', 'y', 'z'>(s, x, y, z), z);
				}
			}
		}
	}
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_2d_internal(real_t* __restrict__ gradients, real_t* __restrict__ densities, point_t<index_t, 3> voxel_shape,
					   const density_layout_t dens_l, const gradient_layout_t grad_l)
{
    const index_t y_dim = dens_l | noarr::get_length<'y'>();
    const index_t x_dim = dens_l | noarr::get_length<'x'>();
    const index_t s_dim = dens_l | noarr::get_length<'s'>();

	for (index_t y = 0; y < y_dim; y++)
	{
		for (index_t x = 0; x < x_dim; x++)
		{
			for (index_t s = 0; s < s_dim; s++)
			{
				solve_single<'x', 0>(gradients, densities, voxel_shape[0], dens_l ^ noarr::fix<'s', 'y'>(s, y),
									 grad_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), x);

				solve_single<'y', 1>(gradients, densities, voxel_shape[1], dens_l ^ noarr::fix<'s', 'x'>(s, x),
									 grad_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), y);
			}
		}
	}
}

template <typename density_layout_t, typename gradient_layout_t>
void solve_1d_internal(real_t* __restrict__ gradients, real_t* __restrict__ densities, point_t<index_t, 3> voxel_shape,
					   const density_layout_t dens_l, const gradient_layout_t grad_l)
{
    const index_t x_dim = dens_l | noarr::get_length<'x'>();
    const index_t s_dim = dens_l | noarr::get_length<'s'>();

	for (index_t x = 0; x < x_dim; x++)
	{
		for (index_t s = 0; s < s_dim; s++)
		{
			solve_single<'x', 0>(gradients, densities, voxel_shape[0], dens_l ^ noarr::fix<'s'>(s),
								 grad_l ^ noarr::fix<'s', 'x'>(s, x), x);
		}
	}
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
