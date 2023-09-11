typedef int index_t;

#ifdef DOUBLE
typedef double real_t;
typedef atomic_double atomic_real_t;
#else
typedef float real_t;
typedef atomic_float atomic_real_t;
#endif

kernel void solve_slice_2d_x(global real_t* restrict densities, global const real_t* b, constant real_t* c,
							 constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size)
{
	int id = get_global_id(0);
	index_t y = id / substrates_count;
	index_t s = id % substrates_count;

	const real_t a = c[s];

	if (dirichlet_conditions_min[s])
		densities[(y * x_size) * substrates_count + s] = dirichlet_values_min[s];

	real_t tmp = densities[(y * x_size) * substrates_count + s];

	for (index_t x = 1; x < x_size - 1; x++)
	{
		tmp = densities[(y * x_size + x) * substrates_count + s] - a * b[(x - 1) * substrates_count + s] * tmp;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[(y * x_size + x_size - 1) * substrates_count + s] = dirichlet_values_max[s];

	tmp =
		(densities[(y * x_size + x_size - 1) * substrates_count + s] - a * b[(x_size - 2) * substrates_count + s] * tmp)
		* b[(x_size - 1) * substrates_count + s];
	densities[(y * x_size + x_size - 1) * substrates_count + s] = tmp;

	for (index_t x = x_size - 2; x >= 0; x--)
	{
		tmp = (densities[(y * x_size + x) * substrates_count + s] - a * tmp) * b[x * substrates_count + s];
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_3d_x(global real_t* restrict densities, global const real_t* restrict b,
							 constant real_t* restrict c, constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size, index_t z_size)
{
	int id = get_global_id(0);
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
			  - a * b[(x - 1) * substrates_count + s] * tmp;
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s]
		   - a * b[(x_size - 2) * substrates_count + s] * tmp)
		  * b[(x_size - 1) * substrates_count + s];
	densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s] = tmp;

	for (index_t x = x_size - 2; x >= 0; x--)
	{
		tmp = (densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] - a * tmp)
			  * b[x * substrates_count + s];
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_2d_y(global real_t* restrict densities, global const real_t* b, constant real_t* c,
							 constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size)
{
	int id = get_global_id(0);
	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	const real_t a = c[s];

	if (dirichlet_conditions_min[s])
		densities[x * substrates_count + s] = dirichlet_values_min[s];

	real_t tmp = densities[x * substrates_count + s];

	for (index_t y = 1; y < y_size - 1; y++)
	{
		tmp = densities[(y * x_size + x) * substrates_count + s] - a * b[(y - 1) * substrates_count + s] * tmp;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[((y_size - 1) * x_size + x) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[((y_size - 1) * x_size + x) * substrates_count + s]
		   - a * b[(y_size - 2) * substrates_count + s] * tmp)
		  * b[(y_size - 1) * substrates_count + s];
	densities[((y_size - 1) * x_size + x) * substrates_count + s] = tmp;

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		tmp = (densities[(y * x_size + x) * substrates_count + s] - a * tmp) * b[y * substrates_count + s];
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_3d_y(global real_t* restrict densities, global const real_t* restrict b,
							 constant real_t* restrict c, constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size, index_t z_size)
{
	int id = get_global_id(0);
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
			  - a * b[(y - 1) * substrates_count + s] * tmp;
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s]
		   - a * b[(y_size - 2) * substrates_count + s] * tmp)
		  * b[(y_size - 1) * substrates_count + s];
	densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s] = tmp;

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		tmp = (densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] - a * tmp)
			  * b[y * substrates_count + s];
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_3d_z(global real_t* restrict densities, global const real_t* restrict b,
							 constant real_t* restrict c, constant bool* restrict dirichlet_conditions_min,
							 constant real_t* restrict dirichlet_values_min,
							 constant bool* restrict dirichlet_conditions_max,
							 constant real_t* restrict dirichlet_values_max, index_t substrates_count, index_t x_size,
							 index_t y_size, index_t z_size)
{
	int id = get_global_id(0);
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
			  - a * b[(z - 1) * substrates_count + s] * tmp;
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s] = dirichlet_values_max[s];

	tmp = (densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s]
		   - a * b[(z_size - 2) * substrates_count + s] * tmp)
		  * b[(z_size - 1) * substrates_count + s];
	densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;

	for (index_t z = z_size - 2; z >= 0; z--)
	{
		tmp = (densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] - a * tmp)
			  * b[z * substrates_count + s];
		densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] = tmp;
	}
}

kernel void solve_slice_2d_x_block(global real_t* restrict densities, global const real_t* restrict a,
								   global const real_t* restrict r_fwd, global const real_t* restrict c_fwd,
								   global const real_t* restrict a_bck, global const real_t* restrict c_bck,
								   global const real_t* restrict c_rdc, global const real_t* restrict r_rdc,
								   global const bool* restrict dirichlet_conditions_min,
								   global const real_t* restrict dirichlet_values_min,
								   global const bool* restrict dirichlet_conditions_max,
								   global const real_t* restrict dirichlet_values_max, index_t substrates_count,
								   index_t x_size, index_t y_size, index_t block_size)
{
	int lane_id = get_local_id(0) % 32;
	int warp_id = get_local_id(0) / 32;
	int warps = get_local_size(0) / 32;

	int id = lane_id + get_group_id(0) * 32;

	index_t y = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x_blocks = (x_size + block_size - 1) / block_size;

	if (warp_id == 0 && id < y_size * substrates_count)
	{
		if (dirichlet_conditions_min[s])
			densities[(y * x_size) * substrates_count + s] = dirichlet_values_min[s];

		if (dirichlet_conditions_max[s])
			densities[(y * x_size + x_size - 1) * substrates_count + s] = dirichlet_values_max[s];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < y_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < x_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(x_size, block_idx * block_size + block_size);

			if (block_end - block_begin <= 2)
			{
				for (index_t x = block_begin; x < block_end; x++)
					densities[(y * x_size + x) * substrates_count + s] *= r_fwd[x * substrates_count + s];

				continue;
			}

			// fwd
			{
				index_t x = block_begin;

				for (index_t i = 0; i < 2; i++, x++)
					densities[(y * x_size + x) * substrates_count + s] *= r_fwd[x * substrates_count + s];

				for (; x < block_end; x++)
				{
					densities[(y * x_size + x) * substrates_count + s] =
						(densities[(y * x_size + x) * substrates_count + s]
						 - a[s] * densities[(y * x_size + x - 1) * substrates_count + s])
						* r_fwd[x * substrates_count + s];
				}
			}

			// bck
			{
				index_t x;
				for (x = block_end - 3; x > block_begin; x--)
				{
					densities[(y * x_size + x) * substrates_count + s] =
						densities[(y * x_size + x) * substrates_count + s]
						- c_fwd[x * substrates_count + s] * densities[(y * x_size + x + 1) * substrates_count + s];
				}

				densities[(y * x_size + x) * substrates_count + s] =
					(densities[(y * x_size + x) * substrates_count + s]
					 - c_fwd[x * substrates_count + s] * densities[(y * x_size + x + 1) * substrates_count + s])
					/ (1 - a_bck[(x + 1) * substrates_count + s] * c_fwd[x * substrates_count + s]);
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (warp_id == 0 && id < y_size * substrates_count)
	{
		for (index_t i = 1; i < 2 * x_blocks - 1; i++) // we process all pairs but we skip last element of last pair
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_x = block_idx * block_size + offset;

			index_t prev_block_idx = (i - 1) / 2;
			index_t prev_offset = (i - 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_x = prev_block_idx * block_size + prev_offset;

			densities[(y * x_size + my_x) * substrates_count + s] =
				(densities[(y * x_size + my_x) * substrates_count + s]
				 - a_bck[my_x * substrates_count + s] * densities[(y * x_size + prev_my_x) * substrates_count + s])
				* r_rdc[i * substrates_count + s];
		}

		if (x_size - (x_blocks - 1) * block_size != 1) // we process the last element of the last pair only if it exists
		{
			index_t prev_block_idx = x_blocks - 1;
			index_t prev_offset = 0;
			index_t prev_my_x = prev_block_idx * block_size + prev_offset;

			densities[(y * x_size + x_size - 1) * substrates_count + s] =
				(densities[(y * x_size + x_size - 1) * substrates_count + s]
				 - a_bck[(x_size - 1) * substrates_count + s]
					   * densities[(y * x_size + prev_my_x) * substrates_count + s])
				* r_rdc[(2 * x_blocks - 1) * substrates_count + s];

			densities[(y * x_size + prev_my_x) * substrates_count + s] =
				densities[(y * x_size + prev_my_x) * substrates_count + s]
				- c_rdc[(2 * x_blocks - 2) * substrates_count + s]
					  * densities[(y * x_size + x_size - 1) * substrates_count + s];
		}

		for (index_t i = 2 * x_blocks - 3; i >= 0; i--)
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_x = block_idx * block_size + offset;

			index_t prev_block_idx = (i + 1) / 2;
			index_t prev_offset = (i + 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_x = prev_block_idx * block_size + prev_offset;

			densities[(y * x_size + my_x) * substrates_count + s] =
				densities[(y * x_size + my_x) * substrates_count + s]
				- c_rdc[i * substrates_count + s] * densities[(y * x_size + prev_my_x) * substrates_count + s];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < y_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < x_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(x_size, block_idx * block_size + block_size);

			const real_t lower = densities[(y * x_size + block_begin) * substrates_count + s];
			const real_t upper = densities[(y * x_size + (block_end - 1)) * substrates_count + s];

			for (index_t x = block_begin + 1; x < block_end - 1; x++)
			{
				densities[(y * x_size + x) * substrates_count + s] +=
					-a_bck[x * substrates_count + s] * lower - c_bck[x * substrates_count + s] * upper;
			}
		}
	}
}

kernel void solve_slice_3d_x_block(global real_t* restrict densities, global const real_t* restrict a,
								   global const real_t* restrict r_fwd, global const real_t* restrict c_fwd,
								   global const real_t* restrict a_bck, global const real_t* restrict c_bck,
								   global const real_t* restrict c_rdc, global const real_t* restrict r_rdc,
								   global const bool* restrict dirichlet_conditions_min,
								   global const real_t* restrict dirichlet_values_min,
								   global const bool* restrict dirichlet_conditions_max,
								   global const real_t* restrict dirichlet_values_max, index_t substrates_count,
								   index_t x_size, index_t y_size, index_t z_size, index_t block_size)
{
	int lane_id = get_local_id(0) % 32;
	int warp_id = get_local_id(0) / 32;
	int warps = get_local_size(0) / 32;

	int id = lane_id + get_group_id(0) * 32;

	index_t z = id / (y_size * substrates_count);
	index_t y = (id / substrates_count) % y_size;
	index_t s = id % substrates_count;

	index_t x_blocks = (x_size + block_size - 1) / block_size;

	if (warp_id == 0 && id < y_size * z_size * substrates_count)
	{
		if (dirichlet_conditions_min[s])
			densities[(z * x_size * y_size + y * x_size) * substrates_count + s] = dirichlet_values_min[s];

		if (dirichlet_conditions_max[s])
			densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s] = dirichlet_values_max[s];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < y_size * z_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < x_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(x_size, block_idx * block_size + block_size);

			if (block_end - block_begin <= 2)
			{
				for (index_t x = block_begin; x < block_end; x++)
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] *=
						r_fwd[x * substrates_count + s];

				continue;
			}

			// fwd
			{
				index_t x = block_begin;

				for (index_t i = 0; i < 2; i++, x++)
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] *=
						r_fwd[x * substrates_count + s];

				for (; x < block_end; x++)
				{
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
						(densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
						 - a[s] * densities[(z * x_size * y_size + y * x_size + x - 1) * substrates_count + s])
						* r_fwd[x * substrates_count + s];
				}
			}

			// bck
			{
				index_t x;
				for (x = block_end - 3; x > block_begin; x--)
				{
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
						densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
						- c_fwd[x * substrates_count + s]
							  * densities[(z * x_size * y_size + y * x_size + x + 1) * substrates_count + s];
				}

				densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
					(densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
					 - c_fwd[x * substrates_count + s]
						   * densities[(z * x_size * y_size + y * x_size + x + 1) * substrates_count + s])
					/ (1 - a_bck[(x + 1) * substrates_count + s] * c_fwd[x * substrates_count + s]);
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (warp_id == 0 && id < y_size * z_size * substrates_count)
	{
		for (index_t i = 1; i < 2 * x_blocks - 1; i++) // we process all pairs but we skip last element of last pair
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_x = block_idx * block_size + offset;

			index_t prev_block_idx = (i - 1) / 2;
			index_t prev_offset = (i - 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_x = prev_block_idx * block_size + prev_offset;

			densities[(z * x_size * y_size + y * x_size + my_x) * substrates_count + s] =
				(densities[(z * x_size * y_size + y * x_size + my_x) * substrates_count + s]
				 - a_bck[my_x * substrates_count + s]
					   * densities[(z * x_size * y_size + y * x_size + prev_my_x) * substrates_count + s])
				* r_rdc[i * substrates_count + s];
		}

		if (x_size - (x_blocks - 1) * block_size != 1) // we process the last element of the last pair only if it exists
		{
			index_t prev_block_idx = x_blocks - 1;
			index_t prev_offset = 0;
			index_t prev_my_x = prev_block_idx * block_size + prev_offset;

			densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s] =
				(densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s]
				 - a_bck[(x_size - 1) * substrates_count + s]
					   * densities[(z * x_size * y_size + y * x_size + prev_my_x) * substrates_count + s])
				* r_rdc[(2 * x_blocks - 1) * substrates_count + s];

			densities[(z * x_size * y_size + y * x_size + prev_my_x) * substrates_count + s] =
				densities[(z * x_size * y_size + y * x_size + prev_my_x) * substrates_count + s]
				- c_rdc[(2 * x_blocks - 2) * substrates_count + s]
					  * densities[(z * x_size * y_size + y * x_size + x_size - 1) * substrates_count + s];
		}

		for (index_t i = 2 * x_blocks - 3; i >= 0; i--)
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_x = block_idx * block_size + offset;

			index_t prev_block_idx = (i + 1) / 2;
			index_t prev_offset = (i + 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_x = prev_block_idx * block_size + prev_offset;

			densities[(z * x_size * y_size + y * x_size + my_x) * substrates_count + s] =
				densities[(z * x_size * y_size + y * x_size + my_x) * substrates_count + s]
				- c_rdc[i * substrates_count + s]
					  * densities[(z * x_size * y_size + y * x_size + prev_my_x) * substrates_count + s];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < y_size * z_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < x_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(x_size, block_idx * block_size + block_size);

			const real_t lower = densities[(z * x_size * y_size + y * x_size + block_begin) * substrates_count + s];
			const real_t upper = densities[(z * x_size * y_size + y * x_size + (block_end - 1)) * substrates_count + s];

			for (index_t x = block_begin + 1; x < block_end - 1; x++)
			{
				densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] +=
					-a_bck[x * substrates_count + s] * lower - c_bck[x * substrates_count + s] * upper;
			}
		}
	}
}

kernel void solve_slice_2d_y_block(global real_t* restrict densities, global const real_t* restrict a,
								   global const real_t* restrict r_fwd, global const real_t* restrict c_fwd,
								   global const real_t* restrict a_bck, global const real_t* restrict c_bck,
								   global const real_t* restrict c_rdc, global const real_t* restrict r_rdc,
								   global const bool* restrict dirichlet_conditions_min,
								   global const real_t* restrict dirichlet_values_min,
								   global const bool* restrict dirichlet_conditions_max,
								   global const real_t* restrict dirichlet_values_max, index_t substrates_count,
								   index_t x_size, index_t y_size, index_t block_size)
{
	int lane_id = get_local_id(0) % 32;
	int warp_id = get_local_id(0) / 32;
	int warps = get_local_size(0) / 32;

	int id = lane_id + get_group_id(0) * 32;

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	index_t y_blocks = (y_size + block_size - 1) / block_size;

	if (warp_id == 0 && id < x_size * substrates_count)
	{
		if (dirichlet_conditions_min[s])
			densities[x * substrates_count + s] = dirichlet_values_min[s];

		if (dirichlet_conditions_max[s])
			densities[((y_size - 1) * x_size + x) * substrates_count + s] = dirichlet_values_max[s];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < x_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < y_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(y_size, block_idx * block_size + block_size);

			if (block_end - block_begin <= 2)
			{
				for (index_t y = block_begin; y < block_end; y++)
					densities[(y * x_size + x) * substrates_count + s] *= r_fwd[y * substrates_count + s];

				continue;
			}

			// fwd
			{
				index_t y = block_begin;

				for (index_t i = 0; i < 2; i++, y++)
					densities[(y * x_size + x) * substrates_count + s] *= r_fwd[y * substrates_count + s];

				for (; y < block_end; y++)
				{
					densities[(y * x_size + x) * substrates_count + s] =
						(densities[(y * x_size + x) * substrates_count + s]
						 - a[s] * densities[((y - 1) * x_size + x) * substrates_count + s])
						* r_fwd[y * substrates_count + s];
				}
			}

			// bck
			{
				index_t y;
				for (y = block_end - 3; y > block_begin; y--)
				{
					densities[(y * x_size + x) * substrates_count + s] =
						densities[(y * x_size + x) * substrates_count + s]
						- c_fwd[y * substrates_count + s] * densities[((y + 1) * x_size + x) * substrates_count + s];
				}

				densities[(y * x_size + x) * substrates_count + s] =
					(densities[(y * x_size + x) * substrates_count + s]
					 - c_fwd[y * substrates_count + s] * densities[((y + 1) * x_size + x) * substrates_count + s])
					/ (1 - a_bck[(y + 1) * substrates_count + s] * c_fwd[y * substrates_count + s]);
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (warp_id == 0 && id < x_size * substrates_count)
	{
		for (index_t i = 1; i < 2 * y_blocks - 1; i++) // we process all pairs but we skip last element of last pair
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_y = block_idx * block_size + offset;

			index_t prev_block_idx = (i - 1) / 2;
			index_t prev_offset = (i - 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_y = prev_block_idx * block_size + prev_offset;

			densities[(my_y * x_size + x) * substrates_count + s] =
				(densities[(my_y * x_size + x) * substrates_count + s]
				 - a_bck[my_y * substrates_count + s] * densities[(prev_my_y * x_size + x) * substrates_count + s])
				* r_rdc[i * substrates_count + s];
		}

		if (y_size - (y_blocks - 1) * block_size != 1) // we process the last element of the last pair only if it exists
		{
			index_t prev_block_idx = y_blocks - 1;
			index_t prev_offset = 0;
			index_t prev_my_y = prev_block_idx * block_size + prev_offset;

			densities[((y_size - 1) * x_size + x) * substrates_count + s] =
				(densities[((y_size - 1) * x_size + x) * substrates_count + s]
				 - a_bck[(y_size - 1) * substrates_count + s]
					   * densities[(prev_my_y * x_size + x) * substrates_count + s])
				* r_rdc[(2 * y_blocks - 1) * substrates_count + s];

			densities[(prev_my_y * x_size + x) * substrates_count + s] =
				densities[(prev_my_y * x_size + x) * substrates_count + s]
				- c_rdc[(2 * y_blocks - 2) * substrates_count + s]
					  * densities[((y_size - 1) * x_size + x) * substrates_count + s];
		}

		for (index_t i = 2 * y_blocks - 3; i >= 0; i--)
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_y = block_idx * block_size + offset;

			index_t prev_block_idx = (i + 1) / 2;
			index_t prev_offset = (i + 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_y = prev_block_idx * block_size + prev_offset;

			densities[(my_y * x_size + x) * substrates_count + s] =
				densities[(my_y * x_size + x) * substrates_count + s]
				- c_rdc[i * substrates_count + s] * densities[(prev_my_y * x_size + x) * substrates_count + s];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < x_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < y_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(y_size, block_idx * block_size + block_size);

			const real_t lower = densities[(block_begin * x_size + x) * substrates_count + s];
			const real_t upper = densities[((block_end - 1) * x_size + x) * substrates_count + s];

			for (index_t y = block_begin + 1; y < block_end - 1; y++)
			{
				densities[(y * x_size + x) * substrates_count + s] +=
					-a_bck[y * substrates_count + s] * lower - c_bck[y * substrates_count + s] * upper;
			}
		}
	}
}

kernel void solve_slice_3d_y_block(global real_t* restrict densities, global const real_t* restrict a,
								   global const real_t* restrict r_fwd, global const real_t* restrict c_fwd,
								   global const real_t* restrict a_bck, global const real_t* restrict c_bck,
								   global const real_t* restrict c_rdc, global const real_t* restrict r_rdc,
								   global const bool* restrict dirichlet_conditions_min,
								   global const real_t* restrict dirichlet_values_min,
								   global const bool* restrict dirichlet_conditions_max,
								   global const real_t* restrict dirichlet_values_max, index_t substrates_count,
								   index_t x_size, index_t y_size, index_t z_size, index_t block_size)
{
	int lane_id = get_local_id(0) % 32;
	int warp_id = get_local_id(0) / 32;
	int warps = get_local_size(0) / 32;

	int id = lane_id + get_group_id(0) * 32;

	index_t z = id / (x_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	index_t y_blocks = (y_size + block_size - 1) / block_size;

	if (warp_id == 0 && id < x_size * z_size * substrates_count)
	{
		if (dirichlet_conditions_min[s])
			densities[(z * x_size * y_size + x) * substrates_count + s] = dirichlet_values_min[s];

		if (dirichlet_conditions_max[s])
			densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s] =
				dirichlet_values_max[s];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < x_size * z_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < y_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(y_size, block_idx * block_size + block_size);

			if (block_end - block_begin <= 2)
			{
				for (index_t y = block_begin; y < block_end; y++)
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] *=
						r_fwd[y * substrates_count + s];

				continue;
			}

			// fwd
			{
				index_t y = block_begin;

				for (index_t i = 0; i < 2; i++, y++)
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] *=
						r_fwd[y * substrates_count + s];

				for (; y < block_end; y++)
				{
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
						(densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
						 - a[s] * densities[(z * x_size * y_size + (y - 1) * x_size + x) * substrates_count + s])
						* r_fwd[y * substrates_count + s];
				}
			}

			// bck
			{
				index_t y;
				for (y = block_end - 3; y > block_begin; y--)
				{
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
						densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
						- c_fwd[y * substrates_count + s]
							  * densities[(z * x_size * y_size + (y + 1) * x_size + x) * substrates_count + s];
				}

				densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
					(densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
					 - c_fwd[y * substrates_count + s]
						   * densities[(z * x_size * y_size + (y + 1) * x_size + x) * substrates_count + s])
					/ (1 - a_bck[(y + 1) * substrates_count + s] * c_fwd[y * substrates_count + s]);
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (warp_id == 0 && id < x_size * z_size * substrates_count)
	{
		for (index_t i = 1; i < 2 * y_blocks - 1; i++) // we process all pairs but we skip last element of last pair
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_y = block_idx * block_size + offset;

			index_t prev_block_idx = (i - 1) / 2;
			index_t prev_offset = (i - 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_y = prev_block_idx * block_size + prev_offset;

			densities[(z * x_size * y_size + my_y * x_size + x) * substrates_count + s] =
				(densities[(z * x_size * y_size + my_y * x_size + x) * substrates_count + s]
				 - a_bck[my_y * substrates_count + s]
					   * densities[(z * x_size * y_size + prev_my_y * x_size + x) * substrates_count + s])
				* r_rdc[i * substrates_count + s];
		}

		if (y_size - (y_blocks - 1) * block_size != 1) // we process the last element of the last pair only if it exists
		{
			index_t prev_block_idx = y_blocks - 1;
			index_t prev_offset = 0;
			index_t prev_my_y = prev_block_idx * block_size + prev_offset;

			densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s] =
				(densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s]
				 - a_bck[(y_size - 1) * substrates_count + s]
					   * densities[(z * x_size * y_size + prev_my_y * x_size + x) * substrates_count + s])
				* r_rdc[(2 * y_blocks - 1) * substrates_count + s];

			densities[(z * x_size * y_size + prev_my_y * x_size + x) * substrates_count + s] =
				densities[(z * x_size * y_size + prev_my_y * x_size + x) * substrates_count + s]
				- c_rdc[(2 * y_blocks - 2) * substrates_count + s]
					  * densities[(z * x_size * y_size + (y_size - 1) * x_size + x) * substrates_count + s];
		}

		for (index_t i = 2 * y_blocks - 3; i >= 0; i--)
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_y = block_idx * block_size + offset;

			index_t prev_block_idx = (i + 1) / 2;
			index_t prev_offset = (i + 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_y = prev_block_idx * block_size + prev_offset;

			densities[(z * x_size * y_size + my_y * x_size + x) * substrates_count + s] =
				densities[(z * x_size * y_size + my_y * x_size + x) * substrates_count + s]
				- c_rdc[i * substrates_count + s]
					  * densities[(z * x_size * y_size + prev_my_y * x_size + x) * substrates_count + s];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < x_size * z_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < y_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(y_size, block_idx * block_size + block_size);

			const real_t lower = densities[(z * x_size * y_size + block_begin * x_size + x) * substrates_count + s];
			const real_t upper = densities[(z * x_size * y_size + (block_end - 1) * x_size + x) * substrates_count + s];

			for (index_t y = block_begin + 1; y < block_end - 1; y++)
			{
				densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] +=
					-a_bck[y * substrates_count + s] * lower - c_bck[y * substrates_count + s] * upper;
			}
		}
	}
}

kernel void solve_slice_3d_z_block(global real_t* restrict densities, global const real_t* restrict a,
								   global const real_t* restrict r_fwd, global const real_t* restrict c_fwd,
								   global const real_t* restrict a_bck, global const real_t* restrict c_bck,
								   global const real_t* restrict c_rdc, global const real_t* restrict r_rdc,
								   global const bool* restrict dirichlet_conditions_min,
								   global const real_t* restrict dirichlet_values_min,
								   global const bool* restrict dirichlet_conditions_max,
								   global const real_t* restrict dirichlet_values_max, index_t substrates_count,
								   index_t x_size, index_t y_size, index_t z_size, index_t block_size)
{
	int lane_id = get_local_id(0) % 32;
	int warp_id = get_local_id(0) / 32;
	int warps = get_local_size(0) / 32;

	int id = lane_id + get_group_id(0) * 32;

	index_t y = id / (x_size * substrates_count);
	index_t x = (id / substrates_count) % x_size;
	index_t s = id % substrates_count;

	index_t z_blocks = (z_size + block_size - 1) / block_size;

	if (warp_id == 0 && id < x_size * y_size * substrates_count)
	{
		if (dirichlet_conditions_min[s])
			densities[(y * y_size + x) * substrates_count + s] = dirichlet_values_min[s];

		if (dirichlet_conditions_max[s])
			densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s] =
				dirichlet_values_max[s];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < x_size * y_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < z_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(z_size, block_idx * block_size + block_size);

			if (block_end - block_begin <= 2)
			{
				for (index_t z = block_begin; z < block_end; z++)
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] *=
						r_fwd[z * substrates_count + s];

				continue;
			}

			// fwd
			{
				index_t z = block_begin;

				for (index_t i = 0; i < 2; i++, z++)
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] *=
						r_fwd[z * substrates_count + s];

				for (; z < block_end; z++)
				{
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
						(densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
						 - a[s] * densities[((z - 1) * x_size * y_size + y * x_size + x) * substrates_count + s])
						* r_fwd[z * substrates_count + s];
				}
			}

			// bck
			{
				index_t z;
				for (z = block_end - 3; z > block_begin; z--)
				{
					densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
						densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
						- c_fwd[z * substrates_count + s]
							  * densities[((z + 1) * x_size * y_size + y * x_size + x) * substrates_count + s];
				}

				densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] =
					(densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s]
					 - c_fwd[z * substrates_count + s]
						   * densities[((z + 1) * x_size * y_size + y * x_size + x) * substrates_count + s])
					/ (1 - a_bck[(z + 1) * substrates_count + s] * c_fwd[z * substrates_count + s]);
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (warp_id == 0 && id < x_size * y_size * substrates_count)
	{
		for (index_t i = 1; i < 2 * z_blocks - 1; i++) // we process all pairs but we skip last element of last pair
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_z = block_idx * block_size + offset;

			index_t prev_block_idx = (i - 1) / 2;
			index_t prev_offset = (i - 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_z = prev_block_idx * block_size + prev_offset;

			densities[(my_z * x_size * y_size + y * x_size + x) * substrates_count + s] =
				(densities[(my_z * x_size * y_size + y * x_size + x) * substrates_count + s]
				 - a_bck[my_z * substrates_count + s]
					   * densities[(prev_my_z * x_size * y_size + y * x_size + x) * substrates_count + s])
				* r_rdc[i * substrates_count + s];
		}

		if (z_size - (z_blocks - 1) * block_size != 1) // we process the last element of the last pair only if it exists
		{
			index_t prev_block_idx = z_blocks - 1;
			index_t prev_offset = 0;
			index_t prev_my_z = prev_block_idx * block_size + prev_offset;

			densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s] =
				(densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s]
				 - a_bck[(z_size - 1) * substrates_count + s]
					   * densities[(prev_my_z * x_size * y_size + y * x_size + x) * substrates_count + s])
				* r_rdc[(2 * z_blocks - 1) * substrates_count + s];

			densities[(prev_my_z * x_size * y_size + y * x_size + x) * substrates_count + s] =
				densities[(prev_my_z * x_size * y_size + y * x_size + x) * substrates_count + s]
				- c_rdc[(2 * z_blocks - 2) * substrates_count + s]
					  * densities[((z_size - 1) * x_size * y_size + y * x_size + x) * substrates_count + s];
		}

		for (index_t i = 2 * z_blocks - 3; i >= 0; i--)
		{
			index_t block_idx = i / 2;
			index_t offset = i % 2 == 0 ? 0 : block_size - 1;
			index_t my_z = block_idx * block_size + offset;

			index_t prev_block_idx = (i + 1) / 2;
			index_t prev_offset = (i + 1) % 2 == 0 ? 0 : block_size - 1;
			index_t prev_my_z = prev_block_idx * block_size + prev_offset;

			densities[(my_z * x_size * y_size + y * x_size + x) * substrates_count + s] =
				densities[(my_z * x_size * y_size + y * x_size + x) * substrates_count + s]
				- c_rdc[i * substrates_count + s]
					  * densities[(prev_my_z * x_size * y_size + y * x_size + x) * substrates_count + s];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (id < x_size * y_size * substrates_count)
	{
		for (index_t block_idx = warp_id; block_idx < z_blocks; block_idx += warps)
		{
			const index_t block_begin = block_idx * block_size;
			const index_t block_end = min(z_size, block_idx * block_size + block_size);

			const real_t lower = densities[(block_begin * x_size * y_size + y * x_size + x) * substrates_count + s];
			const real_t upper = densities[((block_end - 1) * x_size * y_size + y * x_size + x) * substrates_count + s];

			for (index_t z = block_begin + 1; z < block_end - 1; z++)
			{
				densities[(z * x_size * y_size + y * x_size + x) * substrates_count + s] +=
					-a_bck[z * substrates_count + s] * lower - c_bck[z * substrates_count + s] * upper;
			}
		}
	}
}

kernel void solve_slice_2d_x_shared_full(global real_t* restrict densities, global const real_t* restrict b,
										 global const real_t* restrict c,
										 global const bool* restrict dirichlet_conditions_min,
										 global const real_t* restrict dirichlet_values_min,
										 global const bool* restrict dirichlet_conditions_max,
										 global const real_t* restrict dirichlet_values_max, index_t substrates_count,
										 index_t x_size, index_t y_size, index_t padding, local real_t* restrict shmem)
{
	int threadIdx = get_local_id(0);
	int blockDim = get_local_size(0);
	int blockIdx = get_group_id(0);

	real_t* densities_sh = shmem;
	real_t* b_sh = shmem
				   + ((blockDim + substrates_count - 1) / substrates_count)
						 * (x_size * substrates_count + padding); // padding to remove bank conflicts

	int id = threadIdx + blockIdx * blockDim;

	const index_t y_min = blockIdx * blockDim / substrates_count;
	const index_t y_max = min(((blockIdx + 1) * blockDim - 1) / substrates_count, y_size - 1);

	const index_t offset = x_size * substrates_count;
	const index_t offset2 = x_size * substrates_count + padding;

	for (index_t i = threadIdx; i < x_size * substrates_count; i += blockDim)
	{
		for (index_t y = y_min, y_l = 0; y <= y_max; y++, y_l++)
			densities_sh[y_l * offset2 + i] = densities[y * offset + i];
	}

	for (index_t i = threadIdx; i < x_size * substrates_count; i += blockDim)
	{
		b_sh[i] = b[i];
	}

	index_t y = id / substrates_count - y_min;
	index_t s = id % substrates_count;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (id < y_size * substrates_count)
	{
		const real_t a = -c[s];

		if (dirichlet_conditions_min[s])
			densities_sh[y * offset2 + s] = dirichlet_values_min[s];

		real_t tmp = densities_sh[y * offset2 + s];

		for (index_t x = 1, xp = 0; x < x_size - 1; x++, xp++)
		{
			real_t* density = densities_sh + x * substrates_count + y * offset2 + s;
			const real_t b_ = b_sh[xp * substrates_count + s];
			const real_t d_ = *density;
			tmp = d_ + a * b_ * tmp;
			*density = tmp;
		}

		if (dirichlet_conditions_max[s])
			densities_sh[(x_size - 1) * substrates_count + y * offset2 + s] = dirichlet_values_max[s];

		{
			real_t* density = densities_sh + (x_size - 1) * substrates_count + y * offset2 + s;
			const real_t b_2 = b_sh[(x_size - 2) * substrates_count + s];
			const real_t b_1 = b_sh[(x_size - 1) * substrates_count + s];
			tmp = (*density + a * b_2 * tmp) * b_1;
			*density = tmp;
		}

		for (index_t x = x_size - 2; x >= 0; x--)
		{
			real_t* density = densities_sh + x * substrates_count + y * offset2 + s;
			const real_t b_ = b_sh[x * substrates_count + s];
			tmp = (*density + a * tmp) * b_;
			*density = tmp;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (index_t i = threadIdx; i < x_size * substrates_count; i += blockDim)
	{
		for (index_t y = y_min, y_l = 0; y <= y_max; y++, y_l++)
			densities[y * offset + i] = densities_sh[y_l * offset2 + i];
	}
}

kernel void solve_slice_2d_y_shared_full(global real_t* restrict densities, global const real_t* restrict b,
										 global const real_t* restrict c,
										 global const bool* restrict dirichlet_conditions_min,
										 global const real_t* restrict dirichlet_values_min,
										 global const bool* restrict dirichlet_conditions_max,
										 global const real_t* restrict dirichlet_values_max, index_t substrates_count,
										 index_t x_size, index_t y_size, index_t padding, local real_t* restrict shmem)
{
	int threadIdx = get_local_id(0);
	int blockDim = get_local_size(0);
	int blockIdx = get_group_id(0);

	real_t* densities_sh = shmem;
	real_t* b_sh = shmem + blockDim * y_size;

	int id = threadIdx + blockIdx * blockDim;

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	for (index_t i = threadIdx; i < y_size * substrates_count; i += blockDim)
	{
		b_sh[i] = b[i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (id >= x_size * substrates_count)
		return;

	for (index_t y = 0; y < y_size; y++)
	{
		densities_sh[y * blockDim + threadIdx] = densities[(y * x_size + x) * substrates_count + s];
	}

	const real_t a = -c[s];

	if (dirichlet_conditions_min[s])
		densities_sh[threadIdx] = dirichlet_values_min[s];

	real_t tmp = densities_sh[threadIdx];

	for (index_t y = 1; y < y_size - 1; y++)
	{
		real_t* density = densities_sh + (y * blockDim) + threadIdx;
		const real_t b_ = b_sh[(y - 1) * substrates_count + s];
		const real_t d_ = *density;
		tmp = d_ + a * b_ * tmp;
		*density = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities_sh[((y_size - 1) * blockDim) + threadIdx] = dirichlet_values_max[s];

	{
		const real_t density = densities_sh[(y_size - 1) * blockDim + threadIdx];
		const real_t b_2 = b_sh[(y_size - 2) * substrates_count + s];
		const real_t b_1 = b_sh[(y_size - 1) * substrates_count + s];
		tmp = (density + a * b_2 * tmp) * b_1;
		densities[((y_size - 1) * x_size + x) * substrates_count + s] = tmp;
	}

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		const real_t density = densities_sh[(y * blockDim) + threadIdx];
		const real_t b_ = b_sh[y * substrates_count + s];
		tmp = (density + a * tmp) * b_;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}
