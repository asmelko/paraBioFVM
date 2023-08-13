#pragma once

#include <noarr/structures/interop/traverser_iter.hpp>

// noarr-omp helper for parallel traversal
template <typename T, typename F>
inline void omp_trav_for_each(const T& trav, const F& f)
{
#pragma omp for nowait
	for (auto trav_inner : trav)
		trav_inner.for_each(f);
}

template <typename T, typename F>
inline void omp_p_trav_for_each(const T& trav, const F& f)
{
#pragma omp parallel for
	for (auto trav_inner : trav)
		trav_inner.for_each(f);
}
