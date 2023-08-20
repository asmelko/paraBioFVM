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

void solve_boundary_2d_x(int id, global real_t* restrict substrate_densities,
						 global const real_t* restrict dirichlet_values_min,
						 global const real_t* restrict dirichlet_values_max,
						 global const bool* restrict dirichlet_conditions_min,
						 global const bool* restrict dirichlet_conditions_max, index_t substrates_count, index_t x_size,
						 index_t y_size)
{
	index_t x_offset;
	global const real_t* restrict dirichlet_values;
	global const bool* restrict dirichlet_conditions;

	if (id < x_size * substrates_count)
	{
		x_offset = 0;
		dirichlet_values = dirichlet_values_min;
		dirichlet_conditions = dirichlet_conditions_min;
	}
	else
	{
		x_offset = x_size - 1;
		dirichlet_values = dirichlet_values_max;
		dirichlet_conditions = dirichlet_conditions_max;
		id -= x_size * substrates_count;
	}

	index_t y = id / substrates_count;
	index_t s = id % substrates_count;

	if (dirichlet_conditions && dirichlet_conditions[s])
		substrate_densities[(y * x_size + x_offset) * substrates_count + s] = dirichlet_values[s];
}

void solve_boundary_2d_y(int id, global real_t* restrict substrate_densities,
						 global const real_t* restrict dirichlet_values_min,
						 global const real_t* restrict dirichlet_values_max,
						 global const bool* restrict dirichlet_conditions_min,
						 global const bool* restrict dirichlet_conditions_max, index_t substrates_count, index_t x_size,
						 index_t y_size)
{
	index_t y_offset;
	global const real_t* restrict dirichlet_values;
	global const bool* restrict dirichlet_conditions;

	if (id < y_size * substrates_count)
	{
		y_offset = 0;
		dirichlet_values = dirichlet_values_min;
		dirichlet_conditions = dirichlet_conditions_min;
	}
	else
	{
		y_offset = y_size - 1;
		dirichlet_values = dirichlet_values_max;
		dirichlet_conditions = dirichlet_conditions_max;
		id -= y_size * substrates_count;
	}

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	if (dirichlet_conditions && dirichlet_conditions[s])
		substrate_densities[(y_offset * x_size + x) * substrates_count + s] = dirichlet_values[s];
}

kernel void solve_boundary_2d(
	global real_t* restrict substrate_densities, global const real_t* restrict dirichlet_values_min_x,
	global const real_t* restrict dirichlet_values_max_x, global const bool* restrict dirichlet_conditions_min_x,
	global const bool* restrict dirichlet_conditions_max_x, global const real_t* restrict dirichlet_values_min_y,
	global const real_t* restrict dirichlet_values_max_y, global const bool* restrict dirichlet_conditions_min_y,
	global const bool* restrict dirichlet_conditions_max_y, index_t substrates_count, index_t x_size, index_t y_size)
{
	int id = get_global_id(0);

	if (id < y_size * substrates_count * 2)
	{
		solve_boundary_2d_x(id, substrate_densities, dirichlet_values_min_x, dirichlet_values_max_x,
							dirichlet_conditions_min_x, dirichlet_conditions_max_x, substrates_count, x_size, y_size);
	}
	else
	{
		id -= y_size * substrates_count * 2;
		solve_boundary_2d_y(id, substrate_densities, dirichlet_values_min_y, dirichlet_values_max_y,
							dirichlet_conditions_min_y, dirichlet_conditions_max_y, substrates_count, x_size, y_size);
	}
}

kernel void solve_boundary_3d_x(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values_min,
								global const real_t* restrict dirichlet_values_max,
								global const bool* restrict dirichlet_conditions_min,
								global const bool* restrict dirichlet_conditions_max, index_t substrates_count,
								index_t x_size, index_t y_size, index_t z_size)
{
	int id = get_global_id(0);

	index_t x_offset;
	global const real_t* restrict dirichlet_values;
	global const bool* restrict dirichlet_conditions;

	if (id < y_size * z_size * substrates_count)
	{
		x_offset = 0;
		dirichlet_values = dirichlet_values_min;
		dirichlet_conditions = dirichlet_conditions_min;
	}
	else
	{
		x_offset = x_size - 1;
		dirichlet_values = dirichlet_values_max;
		dirichlet_conditions = dirichlet_conditions_max;
		id -= y_size * z_size * substrates_count;
	}

	index_t z = id / (y_size * substrates_count);
	index_t y = (id / substrates_count) % y_size;
	index_t s = id % substrates_count;

	if (dirichlet_conditions && dirichlet_conditions[s])
		substrate_densities[(z * y_size * x_size + y * x_size + x_offset) * substrates_count + s] = dirichlet_values[s];
}

kernel void solve_boundary_3d_y(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values_min,
								global const real_t* restrict dirichlet_values_max,
								global const bool* restrict dirichlet_conditions_min,
								global const bool* restrict dirichlet_conditions_max, index_t substrates_count,
								index_t x_size, index_t y_size, index_t z_size)
{
	int id = get_global_id(0);

	index_t y_offset;
	global const real_t* restrict dirichlet_values;
	global const bool* restrict dirichlet_conditions;

	if (id < x_size * z_size * substrates_count)
	{
		y_offset = 0;
		dirichlet_values = dirichlet_values_min;
		dirichlet_conditions = dirichlet_conditions_min;
	}
	else
	{
		y_offset = y_size - 1;
		dirichlet_values = dirichlet_values_max;
		dirichlet_conditions = dirichlet_conditions_max;
		id -= x_size * z_size * substrates_count;
	}

	index_t z = id / (y_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	if (dirichlet_conditions && dirichlet_conditions[s])
		substrate_densities[(z * y_size * x_size + y_offset * x_size + x) * substrates_count + s] = dirichlet_values[s];
}

kernel void solve_boundary_3d_z(global real_t* restrict substrate_densities,
								global const real_t* restrict dirichlet_values_min,
								global const real_t* restrict dirichlet_values_max,
								global const bool* restrict dirichlet_conditions_min,
								global const bool* restrict dirichlet_conditions_max, index_t substrates_count,
								index_t x_size, index_t y_size, index_t z_size)
{
	int id = get_global_id(0);

	index_t z_offset;
	global const real_t* restrict dirichlet_values;
	global const bool* restrict dirichlet_conditions;

	if (id < x_size * y_size * substrates_count)
	{
		z_offset = 0;
		dirichlet_values = dirichlet_values_min;
		dirichlet_conditions = dirichlet_conditions_min;
	}
	else
	{
		z_offset = z_size - 1;
		dirichlet_values = dirichlet_values_max;
		dirichlet_conditions = dirichlet_conditions_max;
		id -= x_size * y_size * substrates_count;
	}

	index_t y = id / (x_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	if (dirichlet_conditions && dirichlet_conditions[s])
		substrate_densities[(z_offset * y_size * x_size + y * x_size + x) * substrates_count + s] = dirichlet_values[s];
}
