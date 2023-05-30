#pragma once

#include <noarr/structures_extended.hpp>

#include "types.h"

template <index_t dims>
struct layout_traits
{};

template <>
struct layout_traits<1>
{
	using density_layout_t = decltype(noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>());

	static auto construct_density_layout(index_t substrates_count, const point_t<index_t, 3>& grid_dims)
	{
		return density_layout_t() ^ noarr::set_length<'s'>(substrates_count) ^ noarr::set_length<'x'>(grid_dims[0]);
	}

	using gradient_layout_t =
		decltype(noarr::scalar<real_t>() ^ noarr::array<'d', 1>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>());

	static auto construct_gradient_layout(index_t substrates_count, const point_t<index_t, 3>& grid_dims)
	{
		return gradient_layout_t() ^ noarr::set_length<'s'>(substrates_count) ^ noarr::set_length<'x'>(grid_dims[0]);
	}
};

template <>
struct layout_traits<2>
{
	using density_layout_t =
		decltype(noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>() ^ noarr::vector<'y'>());

	static auto construct_density_layout(index_t substrates_count, const point_t<index_t, 3>& grid_dims)
	{
		return density_layout_t() ^ noarr::set_length<'s'>(substrates_count) ^ noarr::set_length<'x'>(grid_dims[0])
			   ^ noarr::set_length<'y'>(grid_dims[1]);
	}

	using gradient_layout_t = decltype(noarr::scalar<real_t>() ^ noarr::array<'d', 2>() ^ noarr::vector<'s'>()
									   ^ noarr::vector<'x'>() ^ noarr::vector<'y'>());

	static auto construct_gradient_layout(index_t substrates_count, const point_t<index_t, 3>& grid_dims)
	{
		return gradient_layout_t() ^ noarr::set_length<'s'>(substrates_count) ^ noarr::set_length<'x'>(grid_dims[0])
			   ^ noarr::set_length<'y'>(grid_dims[1]);
	}
};

template <>
struct layout_traits<3>
{
	using density_layout_t = decltype(noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>()
									  ^ noarr::vector<'y'>() ^ noarr::vector<'z'>());

	static auto construct_density_layout(index_t substrates_count, const point_t<index_t, 3>& grid_dims)
	{
		return density_layout_t() ^ noarr::set_length<'s'>(substrates_count) ^ noarr::set_length<'x'>(grid_dims[0])
			   ^ noarr::set_length<'y'>(grid_dims[1]) ^ noarr::set_length<'z'>(grid_dims[2]);
	}

	using gradient_layout_t = decltype(noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>()
									   ^ noarr::vector<'y'>() ^ noarr::vector<'z'>() ^ noarr::array<'d', 3>());

	static auto construct_gradient_layout(index_t substrates_count, const point_t<index_t, 3>& grid_dims)
	{
		return gradient_layout_t() ^ noarr::set_length<'s'>(substrates_count) ^ noarr::set_length<'x'>(grid_dims[0])
			   ^ noarr::set_length<'y'>(grid_dims[1]) ^ noarr::set_length<'z'>(grid_dims[2]);
	}
};
