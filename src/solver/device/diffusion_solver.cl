typedef int index_t;

#ifdef DOUBLE
typedef double real_t;
typedef atomic_double atomic_real_t;
#else
typedef float real_t;
typedef atomic_float atomic_real_t;
#endif

kernel void solve_slice_2d_x(global real_t* restrict densities, global const real_t* b, global const real_t* c,
							 constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size)
{
	__local real_t shared[1000];

	for (index_t i = get_local_id(0); i < x_size * substrates_count; i += get_local_size(0))
	{
		shared[i] = b[i];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	int id = get_global_id(0);

	if (id >= y_size * substrates_count)
		return;

	index_t y = id / substrates_count;
	index_t s = id % substrates_count;

	const real_t a = c[s];

	if (dirichlet_conditions_min[s])
		densities[(y * x_size) * substrates_count + s] = dirichlet_values_min[s];

	real_t tmp = densities[(y * x_size) * substrates_count + s];

	for (index_t x = 1; x < x_size - 1; x++)
	{
		tmp = densities[(y * x_size + x) * substrates_count + s] - a * shared[(x - 1) * substrates_count + s] * tmp;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[(y * x_size + x_size - 1) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[(y * x_size + x_size - 1) * substrates_count + s]
		   - a * shared[(x_size - 2) * substrates_count + s] * tmp)
		  * shared[(x_size - 1) * substrates_count + s];
	densities[(y * x_size + x_size - 1) * substrates_count + s] = tmp;

	for (index_t x = x_size - 2; x >= 0; x--)
	{
		tmp = (densities[(y * x_size + x) * substrates_count + s] - a * tmp) * shared[x * substrates_count + s];
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_3d_x(global real_t* restrict densities, global const real_t* restrict b,
							 global const real_t* restrict c, constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size, index_t z_size)
{
	__local real_t shared[1000];

	for (index_t i = get_local_id(0); i < x_size * substrates_count; i += get_local_size(0))
	{
		shared[i] = b[i];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	int id = get_global_id(0);

	if (id >= y_size * z_size * substrates_count)
		return;

	index_t z = id / (y_size * substrates_count);
	index_t y = (id / substrates_count) % y_size;
	index_t s = id % substrates_count;

	const real_t a = c[s];

	if (dirichlet_conditions_min[s])
		densities[(z * x_size * y_size + y * x_size) * substrates_count + s] = dirichlet_values_min[s];

	real_t tmp = densities[(z * x_size * y_size + y * x_size) * substrates_count + s];

	for (index_t x = 1; x < x_size - 1; x++)
	{
		tmp = densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
			  - a * shared[(x - 1) * substrates_count + s] * tmp;
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s]
		   - a * shared[(x_size - 2) * substrates_count + s] * tmp)
		  * shared[(x_size - 1) * substrates_count + s];
	densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s] = tmp;

	for (index_t x = x_size - 2; x >= 0; x--)
	{
		tmp = (densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] - a * tmp)
			  * shared[x * substrates_count + s];
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_2d_y(global real_t* restrict densities, global const real_t* b, global const real_t* c,
							 constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size)
{
	__local real_t shared[1000];

	for (index_t i = get_local_id(0); i < y_size * substrates_count; i += get_local_size(0))
	{
		shared[i] = b[i];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	int id = get_global_id(0);

	if (id >= x_size * substrates_count)
		return;

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	const real_t a = c[s];

	if (dirichlet_conditions_min[s])
		densities[x * substrates_count + s] = dirichlet_values_min[s];

	real_t tmp = densities[x * substrates_count + s];

	for (index_t y = 1; y < y_size - 1; y++)
	{
		tmp = densities[(y * x_size + x) * substrates_count + s] - a * shared[(y - 1) * substrates_count + s] * tmp;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[((y_size - 1) * x_size + x) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[((y_size - 1) * x_size + x) * substrates_count + s]
		   - a * shared[(y_size - 2) * substrates_count + s] * tmp)
		  * shared[(y_size - 1) * substrates_count + s];
	densities[((y_size - 1) * x_size + x) * substrates_count + s] = tmp;

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		tmp = (densities[(y * x_size + x) * substrates_count + s] - a * tmp) * shared[y * substrates_count + s];
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_3d_y(global real_t* restrict densities, global const real_t* restrict b,
							 global const real_t* restrict c, constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size, index_t z_size)
{
	__local real_t shared[1000];

	for (index_t i = get_local_id(0); i < y_size * substrates_count; i += get_local_size(0))
	{
		shared[i] = b[i];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	int id = get_global_id(0);

	if (id >= x_size * z_size * substrates_count)
		return;

	index_t z = id / (x_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	const real_t a = c[s];

	if (dirichlet_conditions_min[s])
		densities[(z * x_size * y_size + x) * substrates_count + s] = dirichlet_values_min[s];

	real_t tmp = densities[(z * x_size * y_size + x) * substrates_count + s];

	for (index_t y = 1; y < y_size - 1; y++)
	{
		tmp = densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
			  - a * shared[(y - 1) * substrates_count + s] * tmp;
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s]
		   - a * shared[(y_size - 2) * substrates_count + s] * tmp)
		  * shared[(y_size - 1) * substrates_count + s];
	densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s] = tmp;

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		tmp = (densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] - a * tmp)
			  * shared[y * substrates_count + s];
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_3d_z(global real_t* restrict densities, global const real_t* restrict b,
							 global const real_t* restrict c, constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size, index_t z_size)
{
	__local real_t shared[1000];

	for (index_t i = get_local_id(0); i < z_size * substrates_count; i += get_local_size(0))
	{
		shared[i] = b[i];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	int id = get_global_id(0);

	if (id >= x_size * y_size * substrates_count)
		return;

	index_t y = id / (x_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	const real_t a = c[s];

	if (dirichlet_conditions_min[s])
		densities[(y * x_size + x) * substrates_count + s] = dirichlet_values_min[s];

	real_t tmp = densities[(y * x_size + x) * substrates_count + s];

	for (index_t z = 1; z < z_size - 1; z++)
	{
		tmp = densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
			  - a * shared[(z - 1) * substrates_count + s] * tmp;
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s]
		   - a * shared[(z_size - 2) * substrates_count + s] * tmp)
		  * shared[(z_size - 1) * substrates_count + s];
	densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;

	for (index_t z = z_size - 2; z >= 0; z--)
	{
		tmp = (densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] - a * tmp)
			  * shared[z * substrates_count + s];
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}
}
