#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <noarr/structures/extra/traverser.hpp>

#include "diffusion_solvers/host_solver.h"
#include "traits.h"

microenvironment default_microenv(cartesian_mesh mesh)
{
	real_t diffusion_time_step = 5;
	index_t substrates_count = 2;

	auto diff_coefs = std::make_unique<real_t[]>(2);
	diff_coefs[0] = 4;
	diff_coefs[1] = 2;
	auto decay_rates = std::make_unique<real_t[]>(2);
	decay_rates[0] = 5;
	decay_rates[1] = 3;

	auto initial_conds = std::make_unique<real_t[]>(2);
	initial_conds[0] = 1;
	initial_conds[1] = 1;

	return microenvironment(mesh, substrates_count, diffusion_time_step, std::move(diff_coefs), std::move(decay_rates),
							std::move(initial_conds));
}

TEST(diffusion_host_solver, D1)
{
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 80, 0, 0 }, { 20, 0, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<1>::density_layout_t() ^ noarr::set_length<'s'>(substrates_count)
				  ^ noarr::set_length<'x'>(mesh.grid_shape[0]);

	noarr::traverser(dens_l).for_dims<'x'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.03846154);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0625);
	});
}

TEST(diffusion_host_solver, D2)
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 800, 800, 0 }, { 20, 20, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<2>::density_layout_t() ^ noarr::set_length<'s'>(substrates_count)
				  ^ noarr::set_length<'x'>(mesh.grid_shape[0]) ^ noarr::set_length<'y'>(mesh.grid_shape[1]);

	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0054869675);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.013840831);
	});
}

TEST(diffusion_host_solver, D3)
{
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 1000, 1000, 1000 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<3>::density_layout_t() ^ noarr::set_length<'s'>(substrates_count)
				  ^ noarr::set_length<'x'>(mesh.grid_shape[0]) ^ noarr::set_length<'y'>(mesh.grid_shape[1])
				  ^ noarr::set_length<'z'>(mesh.grid_shape[2]);

	noarr::traverser(dens_l).for_dims<'x', 'y', 'z'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0012299563);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0046296306);
	});
}
