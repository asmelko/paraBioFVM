#include <memory>
#include <string>
#include <vector>

#include <noarr/structures_extended.hpp>

#include "mesh.h"

template <int dims>
struct layout
{};

template <>
struct layout<1>
{
	using grid_layout_t = decltype(noarr::vector<'x'>());
};

template <>
struct layout<2>
{
	using grid_layout_t = decltype(noarr::vector<'x'>() ^ noarr::vector<'y'>());
};

template <>
struct layout<3>
{
	using grid_layout_t = decltype(noarr::vector<'x'>() ^ noarr::vector<'y'>() ^ noarr::vector<'z'>());
};

template <int dims>
struct microenvironment
{
	microenvironment() : mesh(), substrates_size(1) {}

	using densities_layout_t =
		decltype(noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ typename layout<dims>::grid_layout_t());
	cartesian_mesh<dims> mesh;

	index_t substrates_size;

	std::unique_ptr<real_t[]> substrate_densities;

	std::unique_ptr<real_t[]> gradients;

	std::unique_ptr<real_t[]> dirichlet_values;
	std::unique_ptr<bool[]> dirichlet_conditions;
};
