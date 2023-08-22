typedef int index_t;

#ifdef DOUBLE
typedef double real_t;
typedef atomic_double atomic_real_t;
#else
typedef float real_t;
typedef atomic_float atomic_real_t;
#endif

kernel void solve_interior_2d(global real_t* restrict substrate_densities,
							  global const index_t* restrict dirichlet_voxels,
							  global const real_t* restrict dirichlet_values,
							  global const bool* restrict dirichlet_conditions, index_t substrates_count,
							  index_t x_size, index_t y_size, index_t n)
{
	int id = get_global_id(0);

	if (id >= substrates_count * n)
		return;
	
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x = dirichlet_voxels[i * 2];
	index_t y = dirichlet_voxels[i * 2 + 1];

	if (dirichlet_conditions[i * substrates_count + s])
		substrate_densities[(y * x_size + x) * substrates_count + s] = dirichlet_values[i * substrates_count + s];
}

kernel void solve_interior_3d(global real_t* restrict substrate_densities,
							  global const index_t* restrict dirichlet_voxels,
							  global const real_t* restrict dirichlet_values,
							  global const bool* restrict dirichlet_conditions, index_t substrates_count,
							  index_t x_size, index_t y_size, index_t z_size, index_t n)
{
	int id = get_global_id(0);

	if (id >= substrates_count * n)
		return;

	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x = dirichlet_voxels[i * 3];
	index_t y = dirichlet_voxels[i * 3 + 1];
	index_t z = dirichlet_voxels[i * 3 + 2];

	if (dirichlet_conditions[i * substrates_count + s])
		substrate_densities[(z * y_size * x_size + y * x_size + x) * substrates_count + s] =
			dirichlet_values[i * substrates_count + s];
}
