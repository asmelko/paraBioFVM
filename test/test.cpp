#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>

#include "compute/host/diffusion/diffusion_solver.h"
#include "compute/host/gradient/gradient_solver.h"
#include "traits.h"
#include "utils.h"

class host_diffusion_solver : public testing::TestWithParam<index_t>
{};

INSTANTIATE_TEST_SUITE_P(vectorize, host_diffusion_solver, testing::Values(4, 1));

TEST_P(host_diffusion_solver, D1_uniform)
{
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 80, 0, 0 }, { 20, 0, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m, GetParam());

	s.solve(m);

	auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.03846154);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0625);
	});
}

TEST_P(host_diffusion_solver, D2_uniform)
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 800, 800, 0 }, { 20, 20, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m, GetParam());

	s.solve(m);

	auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x', 'y'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0054869675);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.013840831);
	});
}

TEST_P(host_diffusion_solver, D3_uniform)
{
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 200, 200, 200 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	diffusion_solver s;

	s.initialize(m, GetParam());

	s.solve(m);

	auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);

	noarr::traverser(dens_l).for_dims<'x', 'y', 'z'>([&](auto t) {
		auto s = t.state();

		auto l = dens_l ^ noarr::fix(s);

		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 0), 0.0012299563);
		EXPECT_FLOAT_EQ(l | noarr::get_at<'s'>(m.substrate_densities.get(), 1), 0.0046296306);
	});
}

TEST_P(host_diffusion_solver, D1_random)
{
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 60, 0, 0 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = biorobots_microenv(mesh);

	auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);

	// fill with random values
	for (index_t s = 0; s < m.substrates_count; ++s)
		for (index_t x = 0; x < mesh.grid_shape[0]; ++x)
		{
			index_t index = s + x * m.substrates_count;
			(dens_l | noarr::get_at<'x', 's'>(m.substrate_densities.get(), x, s)) = index;
		}

	diffusion_solver s;

	s.initialize(m, GetParam());

	s.solve(m);

	std::vector<float> expected = {
		0.0486842592, 1.0444132121, 1.9980019980, 2.9880478088, 3.9473197368, 4.9316824055
	};

	for (index_t s = 0; s < m.substrates_count; ++s)
		for (index_t x = 0; x < mesh.grid_shape[0]; ++x)
		{
			index_t index = s + x * m.substrates_count;

			EXPECT_FLOAT_EQ((dens_l | noarr::get_at<'x', 's'>(m.substrate_densities.get(), x, s)), expected[index]);
		}
}

TEST_P(host_diffusion_solver, D2_random)
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 60, 60, 0 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = biorobots_microenv(mesh);

	auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);

	// fill with random values
	for (index_t s = 0; s < m.substrates_count; ++s)
		for (index_t x = 0; x < mesh.grid_shape[0]; ++x)
			for (index_t y = 0; y < mesh.grid_shape[1]; ++y)
			{
				index_t index = s + x * m.substrates_count + y * m.substrates_count * mesh.grid_shape[0];
				(dens_l | noarr::get_at<'x', 'y', 's'>(m.substrate_densities.get(), x, y, s)) = index;
			}

	diffusion_solver s;

	s.initialize(m, GetParam());

	s.solve(m);

	std::vector<float> expected = { 0.1948319355,  1.1899772978,  2.1441254507,	 3.1335099015,	4.0934189658,
									5.0770425053,  6.0427124809,  7.0205751090,	 7.9920082,		8.9641077127,
									9.9412995111,  10.9076403164, 11.8905930262, 12.8511729202, 13.8398865413,
									14.7947055239, 15.7891800565, 16.7382381276 };

	for (index_t s = 0; s < m.substrates_count; ++s)
		for (index_t x = 0; x < mesh.grid_shape[0]; ++x)
			for (index_t y = 0; y < mesh.grid_shape[1]; ++y)
			{
				index_t index = s + x * m.substrates_count + y * m.substrates_count * mesh.grid_shape[0];

				EXPECT_FLOAT_EQ((dens_l | noarr::get_at<'x', 'y', 's'>(m.substrate_densities.get(), x, y, s)),
								expected[index]);
			}
}

TEST_P(host_diffusion_solver, D3_random)
{
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 60, 60, 60 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = biorobots_microenv(mesh);

	auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);

	// fill with random values
	for (index_t s = 0; s < m.substrates_count; ++s)
		for (index_t x = 0; x < mesh.grid_shape[0]; ++x)
			for (index_t y = 0; y < mesh.grid_shape[1]; ++y)
				for (index_t z = 0; z < mesh.grid_shape[2]; ++z)
				{
					index_t index = s + x * m.substrates_count + y * m.substrates_count * mesh.grid_shape[0]
									+ z * m.substrates_count * mesh.grid_shape[0] * mesh.grid_shape[1];
					(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(m.substrate_densities.get(), x, y, z, s)) = index;
				}

	diffusion_solver s;

	s.initialize(m, GetParam());

	s.solve(m);

	std::vector<float> expected = {
		0.6333066643,  1.6268066007,  2.5825920996,	 3.5703051208,	4.5318775349,  5.5138036408,  6.4811629703,
		7.4573021609,  8.4304484056,  9.4008006809,	 10.3797338410, 11.3442992010, 12.3290192763, 13.2877977210,
		14.2783047117, 15.2312962410, 16.2275901470, 17.1747947611, 18.1768755823, 19.1182932811, 20.1261610177,
		21.0617918012, 22.0754464530, 23.0052903212, 24.0247318884, 24.9487888412, 25.9740173237, 26.8922873613,
		27.9233027591, 28.8357858813, 29.8725881944, 30.7792844014, 31.8218736297, 32.7227829214, 33.7711590651,
		34.6662814414, 35.7204445004, 36.6097799615, 37.6697299358, 38.5532784815, 39.6190153711, 40.4967770016,
		41.5683008064, 42.4402755216, 43.5175862418, 44.3837740416, 45.4668716771, 46.3272725617, 47.4161571125,
		48.2707710817, 49.3654425478, 50.2142696018, 51.3147279832, 52.1577681218
	};

	for (index_t s = 0; s < m.substrates_count; ++s)
		for (index_t x = 0; x < mesh.grid_shape[0]; ++x)
			for (index_t y = 0; y < mesh.grid_shape[1]; ++y)
				for (index_t z = 0; z < mesh.grid_shape[2]; ++z)
				{
					index_t index = s + x * m.substrates_count + y * m.substrates_count * mesh.grid_shape[0]
									+ z * m.substrates_count * mesh.grid_shape[0] * mesh.grid_shape[1];

					EXPECT_FLOAT_EQ(
						(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(m.substrate_densities.get(), x, y, z, s)),
						expected[index]);
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

TEST(host_gradient_solver, D1)
{
	cartesian_mesh mesh(1, { 0, 0, 0 }, { 100, 0, 0 }, { 20, 0, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	auto dens_l = layout_traits<1>::construct_density_layout(substrates_count, mesh.grid_shape);

	// Set dummy densities
	for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
	{
		(dens_l | noarr::get_at<'s', 'x'>(m.substrate_densities.get(), 0, x)) = x * x;
		(dens_l | noarr::get_at<'s', 'x'>(m.substrate_densities.get(), 1, m.mesh.grid_shape[0] - 1 - x)) = x * x;
	}

	gradient_solver::solve(m);

	auto grad_l = layout_traits<1>::construct_gradient_layout(substrates_count, mesh.grid_shape);

	auto gradients = noarr::make_bag(grad_l, m.gradients.get());

	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 0, 0)), (1 - 0) / 20.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 0, 1)), (4 - 0) / 40.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 0, 2)), (9 - 1) / 40.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 0, 3)), (16 - 4) / 40.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 0, 4)), (16 - 9) / 20.);

	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 1, 4)), -(1 - 0) / 20.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 1, 3)), -(4 - 0) / 40.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 1, 2)), -(9 - 1) / 40.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 1, 1)), -(16 - 4) / 40.);
	EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x'>(0, 1, 0)), -(16 - 9) / 20.);
}

TEST(host_gradient_solver, D2)
{
	cartesian_mesh mesh(2, { 0, 0, 0 }, { 100, 100, 0 }, { 20, 20, 0 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	auto dens_l = layout_traits<2>::construct_density_layout(substrates_count, mesh.grid_shape);

	auto get_dens = [](index_t x, index_t y, index_t s) { return (x + y) * (x + y) + s; };

	// Set dummy densities
	for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
		{
			(dens_l | noarr::get_at<'s', 'x', 'y'>(m.substrate_densities.get(), 0, x, y)) = get_dens(x, y, 0);
			(dens_l | noarr::get_at<'s', 'x', 'y'>(m.substrate_densities.get(), 1, x, y)) = get_dens(x, y, 1);
		}

	gradient_solver::solve(m);

	auto grad_l = layout_traits<2>::construct_gradient_layout(substrates_count, mesh.grid_shape);

	auto gradients = noarr::make_bag(grad_l, m.gradients.get());

	for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
		{
			auto x_up = x + (x == m.mesh.grid_shape[0] - 1 ? 0 : 1);
			auto x_down = x - (x == 0 ? 0 : 1);
			auto x_div = (x == 0 || x == m.mesh.grid_shape[0] - 1) ? 20. : 40.;

			EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y'>(0, 0, x, y)),
							(get_dens(x_up, y, 0) - get_dens(x_down, y, 0)) / x_div);
			EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y'>(0, 1, x, y)),
							(get_dens(x_up, y, 1) - get_dens(x_down, y, 1)) / x_div);


			auto y_up = y + (y == m.mesh.grid_shape[1] - 1 ? 0 : 1);
			auto y_down = y - (y == 0 ? 0 : 1);
			auto y_div = (y == 0 || y == m.mesh.grid_shape[1] - 1) ? 20. : 40.;

			EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y'>(1, 0, x, y)),
							(get_dens(x, y_up, 0) - get_dens(x, y_down, 0)) / y_div);
			EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y'>(1, 1, x, y)),
							(get_dens(x, y_up, 1) - get_dens(x, y_down, 1)) / y_div);
		}
}

TEST(host_gradient_solver, D3)
{
	cartesian_mesh mesh(3, { 0, 0, 0 }, { 100, 100, 100 }, { 20, 20, 20 });

	index_t substrates_count = 2;
	auto m = default_microenv(mesh);

	auto dens_l = layout_traits<3>::construct_density_layout(substrates_count, mesh.grid_shape);

	auto get_dens = [](index_t x, index_t y, index_t z, index_t s) { return (x + y + z) * (x + y + z) + s; };

	// Set dummy densities
	for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
			for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(m.substrate_densities.get(), 0, x, y, z)) =
					get_dens(x, y, z, 0);
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(m.substrate_densities.get(), 1, x, y, z)) =
					get_dens(x, y, z, 1);
			}

	gradient_solver::solve(m);

	auto grad_l = layout_traits<3>::construct_gradient_layout(substrates_count, mesh.grid_shape);

	auto gradients = noarr::make_bag(grad_l, m.gradients.get());

	for (index_t x = 0; x < m.mesh.grid_shape[0]; x++)
		for (index_t y = 0; y < m.mesh.grid_shape[1]; y++)
			for (index_t z = 0; z < m.mesh.grid_shape[2]; z++)
			{
				auto x_up = x + (x == m.mesh.grid_shape[0] - 1 ? 0 : 1);
				auto x_down = x - (x == 0 ? 0 : 1);
				auto x_div = (x == 0 || x == m.mesh.grid_shape[0] - 1) ? 20. : 40.;

				EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y', 'z'>(0, 0, x, y, z)),
								(get_dens(x_up, y, z, 0) - get_dens(x_down, y, z, 0)) / x_div);
				EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y', 'z'>(0, 1, x, y, z)),
								(get_dens(x_up, y, z, 1) - get_dens(x_down, y, z, 1)) / x_div);


				auto y_up = y + (y == m.mesh.grid_shape[1] - 1 ? 0 : 1);
				auto y_down = y - (y == 0 ? 0 : 1);
				auto y_div = (y == 0 || y == m.mesh.grid_shape[1] - 1) ? 20. : 40.;

				EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y', 'z'>(1, 0, x, y, z)),
								(get_dens(x, y_up, z, 0) - get_dens(x, y_down, z, 0)) / y_div);
				EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y', 'z'>(1, 1, x, y, z)),
								(get_dens(x, y_up, z, 1) - get_dens(x, y_down, z, 1)) / y_div);


				auto z_up = z + (z == m.mesh.grid_shape[2] - 1 ? 0 : 1);
				auto z_down = z - (z == 0 ? 0 : 1);
				auto z_div = (z == 0 || z == m.mesh.grid_shape[2] - 1) ? 20. : 40.;

				EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y', 'z'>(2, 0, x, y, z)),
								(get_dens(x, y, z_up, 0) - get_dens(x, y, z_down, 0)) / z_div);
				EXPECT_FLOAT_EQ((gradients.at<'d', 's', 'x', 'y', 'z'>(2, 1, x, y, z)),
								(get_dens(x, y, z_up, 1) - get_dens(x, y, z_down, 1)) / z_div);
			}
}
