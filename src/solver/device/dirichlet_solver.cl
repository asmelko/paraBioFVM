typedef float real_t;
typedef int index_t;

kernel void solve_interior_2d(global real_t* restrict substrate_densities,
							  global const index_t* restrict dirichlet_voxels,
							  global const real_t* restrict dirichlet_values,
							  global const bool* restrict dirichlet_conditions, index_t substrates_count,
							  index_t x_size, index_t y_size)
{
	int id = get_global_id(0);
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
							  index_t x_size, index_t y_size, index_t z_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x = dirichlet_voxels[i * 3];
	index_t y = dirichlet_voxels[i * 3 + 1];
	index_t z = dirichlet_voxels[i * 3 + 2];

	if (dirichlet_conditions[i * substrates_count + s])
		substrate_densities[(z * y_size * x_size + y * x_size + x) * substrates_count + s] =
			dirichlet_values[i * substrates_count + s];
}

kernel void solve_boundary_2d_x(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values,
								global const bool* restrict dirichlet_conditions, index_t substrates_count,
								index_t x_size, index_t y_size, index_t x_offset)
{
	int id = get_global_id(0);
	index_t y = id / substrates_count;
	index_t s = id % substrates_count;

	if (dirichlet_conditions[s])
		substrate_densities[(y * x_size + x_offset) * substrates_count + s] = dirichlet_values[s];
}

kernel void solve_boundary_2d_y(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values,
								global const bool* restrict dirichlet_conditions, index_t substrates_count,
								index_t x_size, index_t y_size, index_t y_offset)
{
	int id = get_global_id(0);
	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	if (dirichlet_conditions[s])
		substrate_densities[(y_offset * x_size + x) * substrates_count + s] = dirichlet_values[s];
}

kernel void solve_boundary_3d_x(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values,
								global const bool* restrict dirichlet_conditions, index_t substrates_count,
								index_t x_size, index_t y_size, index_t z_size, index_t x_offset)
{
	int id = get_global_id(0);
	index_t z = id / (y_size * substrates_count);
	index_t y = (id / substrates_count) % y_size;
	index_t s = id % substrates_count;

	if (dirichlet_conditions[s])
		substrate_densities[(z * y_size * x_size + y * x_size + x_offset) * substrates_count + s] = dirichlet_values[s];
}

kernel void solve_boundary_3d_y(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values,
								global const bool* restrict dirichlet_conditions, index_t substrates_count,
								index_t x_size, index_t y_size, index_t z_size, index_t y_offset)
{
	int id = get_global_id(0);
	index_t z = id / (y_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	if (dirichlet_conditions[s])
		substrate_densities[(z * y_size * x_size + y_offset * x_size + x) * substrates_count + s] = dirichlet_values[s];
}

kernel void solve_boundary_3d_z(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values,
								global const bool* restrict dirichlet_conditions, index_t substrates_count,
								index_t x_size, index_t y_size, index_t z_size, index_t z_offset)
{
	int id = get_global_id(0);
	index_t y = id / (x_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	if (dirichlet_conditions[s])
		substrate_densities[(z_offset * y_size * x_size + y * x_size + x) * substrates_count + s] = dirichlet_values[s];
}
