#include "lapack_diffusion_solver2.h"

#include "traits.h"

using namespace biofvm;
using namespace biofvm::solvers::host;

extern "C"
{
	extern void sgttrf_(const int* n, float* dl, float* d, float* du, float* du2, int* ipiv, int* info);
	extern void sgttrs_(const char* trans, const int* n, const int* nrhs, const float* dl, const float* d,
						const float* du, const float* du2, const int* ipiv, float* b, const int* ldb, int* info);

	extern void dgttrf_(const int* n, double* dl, double* d, double* du, double* du2, int* ipiv, int* info);
	extern void dgttrs_(const char* trans, const int* n, const int* nrhs, const double* dl, const double* d,
						const double* du, const double* du2, const int* ipiv, double* b, const int* ldb, int* info);
}

#ifdef USE_DOUBLES
	#define gttrf dgttrf_
	#define gttrs dgttrs_
#else
	#define gttrf sgttrf_
	#define gttrs sgttrs_
#endif

void lapack_diffusion_solver2::precompute_values(std::vector<std::unique_ptr<real_t[]>>& dls,
												 std::vector<std::unique_ptr<real_t[]>>& ds,
												 std::vector<std::unique_ptr<real_t[]>>& dus,
												 std::vector<std::unique_ptr<real_t[]>>& du2s,
												 std::vector<std::unique_ptr<int[]>>& ipivs, index_t shape,
												 index_t dims, index_t n, const microenvironment& m)
{
	for (index_t s_idx = 0; s_idx < m.substrates_count; s_idx++)
	{
		auto dl = std::make_unique<real_t[]>(n - 1);
		auto d = std::make_unique<real_t[]>(n);
		auto du = std::make_unique<real_t[]>(n - 1);
		auto du2 = std::make_unique<real_t[]>(n - 2);
		auto ipiv = std::make_unique<int[]>(n);
		for (index_t i = 0; i < n; i++)
		{
			if (i != n - 1)
			{
				dl[i] = -m.diffusion_time_step * m.diffusion_coefficients[s_idx] / (shape * shape);
				du[i] = -m.diffusion_time_step * m.diffusion_coefficients[s_idx] / (shape * shape);
			}

			d[i] = 1 + m.diffusion_time_step * m.decay_rates[s_idx] / dims
				   + m.diffusion_time_step * m.diffusion_coefficients[s_idx] / (shape * shape);

			if (i == 0 || i == n - 1)
				d[i] += m.diffusion_time_step * m.diffusion_coefficients[s_idx] / (shape * shape);
		}

		int info;
		gttrf(&n, dl.get(), d.get(), du.get(), du2.get(), ipiv.get(), &info);

		if (info != 0)
			throw std::runtime_error("LAPACK spttrf failed with error code " + std::to_string(info));

		dls.emplace_back(std::move(dl));
		ds.emplace_back(std::move(d));
		dus.emplace_back(std::move(du));
		du2s.emplace_back(std::move(du2));
		ipivs.emplace_back(std::move(ipiv));
	}
}

void lapack_diffusion_solver2::initialize(microenvironment& m, dirichlet_solver&) { initialize(m); }

void lapack_diffusion_solver2::initialize(microenvironment& m)
{
	if (m.mesh.dims >= 1)
		precompute_values(dlx_, dx_, dux_, du2x_, ipivx_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);
	if (m.mesh.dims >= 2)
		precompute_values(dly_, dy_, duy_, du2y_, ipivy_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
	if (m.mesh.dims >= 3)
		precompute_values(dlz_, dz_, duz_, du2z_, ipivz_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);
}

void lapack_diffusion_solver2::solve(microenvironment& m)
{
	if (m.mesh.dims == 1)
		solve_1d(m);

	if (m.mesh.dims == 2)
		solve_2d(m);

	if (m.mesh.dims == 3)
		solve_3d(m);
}

template <char dim, typename density_layout_t, typename fix_layout_t>
void solve_slice(const std::vector<std::unique_ptr<real_t[]>>& dls, const std::vector<std::unique_ptr<real_t[]>>& ds,
				 const std::vector<std::unique_ptr<real_t[]>>& dus, const std::vector<std::unique_ptr<real_t[]>>& du2s,
				 const std::vector<std::unique_ptr<int[]>>& ipivs, real_t* __restrict__ d, density_layout_t l,
				 fix_layout_t fixed_l)
{
	const index_t n = l | noarr::get_length<dim>();
	const index_t right_hand_sides = (l | noarr::get_size()) / (sizeof(real_t) * n * (l | noarr::get_length<'s'>()));

	// for (index_t s_idx = 0; s_idx < (index_t)(l | noarr::get_length<'s'>()); s_idx++)
	// {
	// 	const index_t begin_offset = (fixed_l | noarr::offset<dim, 's'>(0, s_idx)) / sizeof(real_t);

	// 	int info;
	// 	char c = 'N';
	// 	gttrs(&c, &n, &right_hand_sides, dls[s_idx].get(), ds[s_idx].get(), dus[s_idx].get(), du2s[s_idx].get(),
	// 			ipivs[s_idx].get(), d + begin_offset, &n, &info);

	// 	if (info != 0)
	// 		throw std::runtime_error("LAPACK spttrs failed with error code " + std::to_string(info));
	// }

#pragma omp parallel for
	for (index_t idx = 0; idx < (index_t)(l | noarr::get_length<'s'>()) * right_hand_sides; idx++)
	{
		auto s_idx = idx / right_hand_sides;
		auto dim_idx = idx % right_hand_sides;

		const index_t begin_offset = (fixed_l | noarr::offset<dim, 's'>(0, s_idx)) / sizeof(real_t);

		int info;
		int rhs = 1;
		char c = 'N';
		gttrs(&c, &n, &rhs, dls[s_idx].get(), ds[s_idx].get(), dus[s_idx].get(), du2s[s_idx].get(), ipivs[s_idx].get(),
			  d + begin_offset + n * dim_idx, &n, &info);

		if (info != 0)
			throw std::runtime_error("LAPACK spttrs failed with error code " + std::to_string(info));
	}
}

void lapack_diffusion_solver2::solve_1d(microenvironment& m)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	// dirichlet_solver::solve_1d(m);

	solve_slice<'x'>(dlx_, dx_, dux_, du2x_, ipivx_, m.substrate_densities.get(), dens_l, dens_l);

	// dirichlet_solver::solve_1d(m);
}

void lapack_diffusion_solver2::solve_2d(microenvironment& m)
{
	auto dens_l = layout_traits<2>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	// dirichlet_solver::solve_2d(m);

	solve_slice<'x'>(dlx_, dx_, dux_, du2x_, ipivx_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'y'>(0));

	// dirichlet_solver::solve_2d(m);

	// solve_slice<'y'>(ay_, by_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'x'>(0));

	// dirichlet_solver::solve_2d(m);
}

void lapack_diffusion_solver2::solve_3d(microenvironment&)
{
	// auto dens_l = layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	// dirichlet_solver::solve_3d(m);

	// solve_slice<'x'>(ax_, bx_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'y'>(0) ^
	// noarr::fix<'z'>(0));

	// dirichlet_solver::solve_3d(m);

	// solve_slice<'y'>(ay_, by_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'x'>(0) ^
	// noarr::fix<'z'>(0));

	// dirichlet_solver::solve_3d(m);

	// solve_slice<'z'>(az_, bz_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'x'>(0) ^
	// noarr::fix<'y'>(0));

	// dirichlet_solver::solve_3d(m);
}
