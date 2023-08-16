typedef float real_t;
typedef int index_t;

kernel void solve_boundary_2d(global real_t* restrict substrate_densities,
							  global const real_t* restrict dirichlet_values,
							  global const bool* restrict dirichlet_conditions, index_t substrates_count,
							  index_t offset, index_t step)
{
	int id = get_global_id(0);

	for (index_t s = 0; s < substrates_count; ++s)
		if (dirichlet_conditions[s])
			substrate_densities[(id * step + offset) * substrates_count + s] = dirichlet_values[s];
}
