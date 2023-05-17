#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>

#include "compute/host/diffusion/diffusion_solver.h"
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

TEST(host_diffusion_solver, D1)
{
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 80, 0, 0 }, { 20, 0, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.03846154);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0625);
	});
}

TEST(host_diffusion_solver, D2)
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 800, 800, 0 }, { 20, 20, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0054869675);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.013840831);
	});
}

TEST(host_diffusion_solver, D3)
{
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 200, 200, 200 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x', 'y', 'z'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0012299563);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0046296306);
	});
}

void add_dirichlet_at(microenvironment& m, index_t substrates_count, const std::vector<point_t<index_t, 3>>& indices,
					  const std::vector<real_t>& values)
{
	m.dirichlet_voxels_count = indices.size();
	m.dirichlet_voxels = std::make_unique<index_t[]>(m.dirichlet_voxels_count * m.mesh.dims);

	for (int i = 0; i < m.dirichlet_voxels_count; i++)
		for (int d = 0; d < m.mesh.dims; d++)
			m.dirichlet_voxels[i * m.mesh.dims + d] = indices[i][d];

	m.dirichlet_values = std::make_unique<real_t[]>(substrates_count * m.dirichlet_voxels_count);
	m.dirichlet_conditions = std::make_unique<bool[]>(substrates_count * m.dirichlet_voxels_count);

	for (int i = 0; i < m.dirichlet_voxels_count; i++)
	{
		m.dirichlet_values[i * substrates_count] = values[i];
		m.dirichlet_conditions[i * substrates_count] = true; // only the first substrate
	}
}

TEST(host_dirichlet_solver, one_cond_D1)
{
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 100, 0, 0 }, { 20, 0, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	add_dirichlet_at(m, substrates_count, { { 2, 0, 0 } }, { 1 });

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);
		if (noarr::get_index<'x'>(s) == 2)
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 1);
		else
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.03846154);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0625);
	});
}

TEST(host_dirichlet_solver, one_cond_D2)
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 60, 60, 0 }, { 20, 20, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	add_dirichlet_at(m, substrates_count, { { 1, 1, 0 } }, { 10 });

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);

	// second substrate should not change
	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.013840831);
	});

	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);
		// exacly at the boundary
		if (noarr::get_index<'x'>(s) == 1 && noarr::get_index<'y'>(s) == 1)
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 10);
		// diagonally next to the boundary
		else if (std::abs((index_t)noarr::get_index<'x'>(s) - (index_t)1) == 1
				 && std::abs((index_t)noarr::get_index<'y'>(s) - (index_t)1) == 1)
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0054876315);
		// left and right to the boundary
		else if (std::abs((index_t)noarr::get_index<'x'>(s) - (index_t)1) == 1 && noarr::get_index<'y'>(s) == 1)
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0056665326);
		// above and below to the boundary
		else
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0081802057);
	});
}

TEST(host_dirichlet_solver, one_cond_D3)
{
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 60, 60, 60 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	add_dirichlet_at(m, substrates_count, { { 1, 1, 1 } }, { 1000 });

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);

	// second substrate should not change
	noarr::traverser(dens_l).for_dims<'x', 'y', 'z'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0046296306);
	});

	auto densities = noarr::make_bag(dens_l ^ noarr::fix<'s'>(0), m.substrate_densities.get());

	// lower and upper xy slices are the same
	for (auto z : { 0, 2 })
	{
		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(0, 0, z)), 0.0012301364);
		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(1, 0, z)), 0.001549035);
		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(2, 0, z)), 0.0012301364);

		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(0, 1, z)), 0.0012637526);
		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(1, 1, z)), 0.566124);
		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(2, 1, z)), 0.0012637526);

		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(0, 2, z)), 0.0012301364);
		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(1, 2, z)), 0.001549035);
		EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(2, 2, z)), 0.0012301364);
	}

	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(0, 0, 1)), 0.0012637526);
	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(1, 0, 1)), 0.061110392);
	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(2, 0, 1)), 0.0012637526);

	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(0, 1, 1)), 0.0075723953);
	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(1, 1, 1)), 1000);
	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(2, 1, 1)), 0.0075723953);

	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(0, 2, 1)), 0.0012637526);
	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(1, 2, 1)), 0.061110392);
	EXPECT_FLOAT_EQ((densities.at<'x', 'y', 'z'>(2, 2, 1)), 0.0012637526);
}

TEST(host_dirichlet_solver, multiple_cond_D1)
{
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 100, 0, 0 }, { 20, 0, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	add_dirichlet_at(m, substrates_count, { { 0, 0, 0 }, { 4, 0, 0 } }, { 1, 1 });

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);
		if (noarr::get_index<'x'>(s) == 0 || noarr::get_index<'x'>(s) == 4)
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 1);
		else
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.03846154);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0625);
	});
}

TEST(host_dirichlet_solver, multiple_cond_D2)
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 60, 60, 0 }, { 20, 20, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	add_dirichlet_at(m, substrates_count, { { 0, 0, 0 }, { 1, 0, 0 }, { 2, 0, 0 } }, { 10, 10, 10 });

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);

	// second substrate should not change
	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.013840831);
	});

	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);
		// First row
		if (noarr::get_index<'y'>(s) == 0)
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 10);
		// Second row
		else if (noarr::get_index<'y'>(s) == 1)
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0081802057);
		// Thrid row
		else
			EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0054969075);
	});
}

TEST(host_dirichlet_solver, multiple_cond_D3)
{
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 60, 60, 60 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	add_dirichlet_at(m, substrates_count,
					 { { 0, 0, 1 },
					   { 1, 0, 1 },
					   { 2, 0, 1 },
					   { 0, 1, 1 },
					   { 1, 1, 1 },
					   { 2, 1, 1 },
					   { 0, 2, 1 },
					   { 1, 2, 1 },
					   { 2, 2, 1 } },
					 { 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000 });

	diffusion_solver s;

	s.initialize(m);

	s.solve(m);

	auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);

	// second substrate should not change
	noarr::traverser(dens_l).for_dims<'x', 'y', 'z'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0046296306);
	});

	// lower and upper xy slices are the same
	for (auto z : { 0, 2 })
	{
		noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
			auto s = t.state();

			auto l = dens_l ^ noarr::fix(s);

			EXPECT_FLOAT_EQ((l | noarr::get_at<'z', 's'>(m.substrate_densities.get(), z, 0)), 0.566124);
		});
	}

	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s) ^ noarr::fix<'z'>(1);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 1000);
	});
}
