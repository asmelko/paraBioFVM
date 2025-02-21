#include "lapack_diffusion_solver.h"

#include "traits.h"

using namespace biofvm;
using namespace biofvm::solvers::host;

extern "C"
{
	extern void spttrf_(const int* n, float* d, float* e, int* info);
	extern void spttrs_(const int* n, const int* nrhs, const float* d, const float* e, float* b, const int* ldb,
						int* info);

	extern void dpttrf_(const int* n, double* d, double* e, int* info);
	extern void dpttrs_(const int* n, const int* nrhs, const double* d, const double* e, double* b, const int* ldb,
						int* info);
}

#ifdef USE_DOUBLES
	#define pttrf dpttrf_
	#define pttrs dpttrs_
#else
	#define pttrf spttrf_
	#define pttrs spttrs_
#endif

// void lapack_diffusion_solver::initialize_diagonals(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b,
// 												   index_t shape, index_t dims, index_t n, const microenvironment& m)
// {
// 	a = std::make_unique<real_t[]>(m.substrates_count * n);
// 	b = std::make_unique<real_t[]>(m.substrates_count * n);

// 	auto diag_l = noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(m.substrates_count, n);

// 	noarr::traverser(diag_l).for_each([&](auto state) {
// 		index_t s = noarr::get_index<'s'>(state);
// 		index_t i = noarr::get_index<'i'>(state);

// 		(diag_l | noarr::get_at(a.get(), state)) =
// 			-m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);


// 		(diag_l | noarr::get_at(b.get(), state)) =
// 			1 + m.diffusion_time_step * m.decay_rates[s] / dims
// 			+ m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);

// 		if (i == 0 || i == n - 1)
// 			(diag_l | noarr::get_at(b.get(), state)) +=
// 				m.diffusion_time_step * m.diffusion_coefficients[s] / (shape * shape);
// 	});
// }

void lapack_diffusion_solver::precompute_values(std::vector<std::unique_ptr<real_t[]>>& a,
												std::vector<std::unique_ptr<real_t[]>>& b, index_t shape, index_t dims,
												index_t n, const microenvironment& m)
{
	for (index_t s_idx = 0; s_idx < m.substrates_count; s_idx++)
	{
		auto single_substr_a = std::make_unique<real_t[]>(n - 1);
		auto single_substr_b = std::make_unique<real_t[]>(n);
		for (index_t i = 0; i < n; i++)
		{
			if (i != n - 1)
				single_substr_a[i] = -m.diffusion_time_step * m.diffusion_coefficients[s_idx] / (shape * shape);

			single_substr_b[i] = 1 + m.diffusion_time_step * m.decay_rates[s_idx] / dims
								 + m.diffusion_time_step * m.diffusion_coefficients[s_idx] / (shape * shape);

			if (i == 0 || i == n - 1)
				single_substr_b[i] += m.diffusion_time_step * m.diffusion_coefficients[s_idx] / (shape * shape);
		}

		int info;
		pttrf(&n, single_substr_b.get(), single_substr_a.get(), &info);

		if (info != 0)
			throw std::runtime_error("LAPACK spttrf failed with error code " + std::to_string(info));

		a.emplace_back(std::move(single_substr_a));
		b.emplace_back(std::move(single_substr_b));
	}
}

void lapack_diffusion_solver::initialize(microenvironment& m, dirichlet_solver&) { initialize(m); }

void lapack_diffusion_solver::initialize(microenvironment& m)
{
	precompute_values(ax_, bx_, m.mesh.voxel_shape[0], m.mesh.dims, m.mesh.grid_shape[0], m);
	precompute_values(ay_, by_, m.mesh.voxel_shape[1], m.mesh.dims, m.mesh.grid_shape[1], m);
	precompute_values(az_, bz_, m.mesh.voxel_shape[2], m.mesh.dims, m.mesh.grid_shape[2], m);
}

void lapack_diffusion_solver::solve(microenvironment& m)
{
	if (m.mesh.dims == 1)
		solve_1d(m);

	if (m.mesh.dims == 2)
		solve_2d(m);

	if (m.mesh.dims == 3)
		solve_3d(m);
}

template <char dim, typename density_layout_t, typename fix_layout_t>
void solve_slice(const std::vector<std::unique_ptr<real_t[]>>& a, const std::vector<std::unique_ptr<real_t[]>>& b,
				 real_t* __restrict__ d, density_layout_t l, fix_layout_t fixed_l)
{
	const index_t n = l | noarr::get_length<dim>();
	const index_t right_hand_sides = (l | noarr::get_size()) / (sizeof(real_t) * n * (l | noarr::get_length<'s'>()));

	// for (index_t s_idx = 0; s_idx < (index_t)(l | noarr::get_length<'s'>()); s_idx++)
	// {
	// 	const index_t begin_offset = (fixed_l | noarr::offset<dim, 's'>(0, s_idx)) / sizeof(real_t);

	// 	int info;
	// 	pttrs(&n, &right_hand_sides, b[s_idx].get(), a[s_idx].get(), d + begin_offset, &n, &info);

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
		pttrs(&n, &rhs, b[s_idx].get(), a[s_idx].get(), d + begin_offset + n * dim_idx, &n, &info);

		if (info != 0)
			throw std::runtime_error("LAPACK spttrs failed with error code " + std::to_string(info));
	}
}

void lapack_diffusion_solver::solve_1d(microenvironment& m)
{
	auto dens_l = layout_traits<1>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	// dirichlet_solver::solve_1d(m);

	solve_slice<'x'>(ax_, bx_, m.substrate_densities.get(), dens_l, dens_l);

	// dirichlet_solver::solve_1d(m);
}

void lapack_diffusion_solver::solve_2d(microenvironment& m)
{
	auto dens_l = layout_traits<2>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	// dirichlet_solver::solve_2d(m);

	solve_slice<'x'>(ax_, bx_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'y'>(0));

	// dirichlet_solver::solve_2d(m);

	// solve_slice<'y'>(ay_, by_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'x'>(0));

	// dirichlet_solver::solve_2d(m);
}

void lapack_diffusion_solver::solve_3d(microenvironment& m)
{
	auto dens_l = layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);

	dirichlet_solver::solve_3d(m);

	solve_slice<'x'>(ax_, bx_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'y'>(0) ^ noarr::fix<'z'>(0));

	dirichlet_solver::solve_3d(m);

	solve_slice<'y'>(ay_, by_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'x'>(0) ^ noarr::fix<'z'>(0));

	dirichlet_solver::solve_3d(m);

	solve_slice<'z'>(az_, bz_, m.substrate_densities.get(), dens_l, dens_l ^ noarr::fix<'x'>(0) ^ noarr::fix<'y'>(0));

	dirichlet_solver::solve_3d(m);
}
