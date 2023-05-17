#include "bulk_solver.h"

#include "../../../traits.h"

void bulk_solver::initialize(microenvironment& m, bulk_func_t supply_rate_f, bulk_func_t uptake_rate_f,
							 bulk_func_t supply_target_densities_f)
{
	supply_rate_f_ = supply_rate_f;
	uptake_rate_f_ = uptake_rate_f;
	supply_target_densities_f_ = supply_target_densities_f;

	supply_rates_ = std::make_unique<real_t[]>(m.substrates_count);
	uptake_rates_ = std::make_unique<real_t[]>(m.substrates_count);
	supply_target_densities_ = std::make_unique<real_t[]>(m.substrates_count);
}

template <typename density_layout_t>
void solve_single(real_t* __restrict__ densities, const real_t* __restrict__ supply_rates,
				  const real_t* __restrict__ uptake_rates, const real_t* __restrict__ supply_target_densities,
				  real_t time_step, const density_layout_t dens_l)
{
	const index_t s_dim = dens_l | noarr::get_length<'s'>();

	for (index_t s = 0; s < s_dim; s++)
	{
		(dens_l | noarr::get_at<'s'>(densities, s)) =
			((dens_l | noarr::get_at<'s'>(densities, s)) + time_step * supply_rates[s] * supply_target_densities[s])
			/ (1 + time_step * (uptake_rates[s] + supply_rates[s]));
	}
}

void bulk_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 1)
		solve_1d(m);
	else if (m.mesh.dims == 2)
		solve_2d(m);
	else if (m.mesh.dims == 3)
		solve_3d(m);
}

void bulk_solver::solve_1d(microenvironment& m)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
	{
		supply_rate_f_(m, { x, 0, 0 }, supply_rates_.get());
		uptake_rate_f_(m, { x, 0, 0 }, uptake_rates_.get());
		supply_target_densities_f_(m, { x, 0, 0 }, supply_target_densities_.get());

		solve_single(m.substrate_densities.get(), supply_rates_.get(), uptake_rates_.get(),
					 supply_target_densities_.get(), m.time_step, dens_l ^ noarr::fix<'x'>(x));
	}
}

void bulk_solver::solve_2d(microenvironment& m)
{
	auto dens_l = layout_traits<2>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
	{
		for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		{
			supply_rate_f_(m, { x, y, 0 }, supply_rates_.get());
			uptake_rate_f_(m, { x, y, 0 }, uptake_rates_.get());
			supply_target_densities_f_(m, { x, y, 0 }, supply_target_densities_.get());

			solve_single(m.substrate_densities.get(), supply_rates_.get(), uptake_rates_.get(),
						 supply_target_densities_.get(), m.time_step, dens_l ^ noarr::fix<'x'>(x) ^ noarr::fix<'y'>(y));
		}
	}
}

void bulk_solver::solve_3d(microenvironment& m)
{
	auto dens_l = layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
	{
		for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
		{
			for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
			{
				supply_rate_f_(m, { x, y, z }, supply_rates_.get());
				uptake_rate_f_(m, { x, y, z }, uptake_rates_.get());
				supply_target_densities_f_(m, { x, y, z }, supply_target_densities_.get());

				solve_single(m.substrate_densities.get(), supply_rates_.get(), uptake_rates_.get(),
							 supply_target_densities_.get(), m.time_step,
							 dens_l ^ noarr::fix<'x'>(x) ^ noarr::fix<'y'>(y) ^ noarr::fix<'z'>(z));
			}
		}
	}
}
