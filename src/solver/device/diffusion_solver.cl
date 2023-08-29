typedef int index_t;

#ifdef DOUBLE
typedef double real_t;
typedef atomic_double atomic_real_t;
#else
typedef float real_t;
typedef atomic_float atomic_real_t;
#endif

kernel void solve_slice_2d_x_m(global real_t* restrict densities, global const real_t* diffusion_coeffs,
							   global const real_t* decay_rates, constant bool* restrict dirichlet_conditions_min,
							   constant real_t* restrict dirichlet_values_min,
							   constant bool* restrict dirichlet_conditions_max,
							   constant real_t* restrict dirichlet_values_max, real_t time_step, real_t shape,
							   index_t substrates_count, index_t x_size, index_t y_size)
{
	__local real_t shared[1000];
	__local real_t* as = shared;
	__local real_t* bs = as + 250;
	__local real_t* cs = bs + 250;
	__local real_t* ds = cs + 250;

	const index_t id = get_global_id(0) / get_local_size(0);
	const index_t y = id / substrates_count;
	const index_t s = id % substrates_count;

	const real_t a = -time_step * diffusion_coeffs[s] / (shape * shape);
	const real_t b0 = 1 + decay_rates[s] * time_step / 2 + time_step * diffusion_coeffs[s] / (shape * shape);
	const real_t b = 1 + decay_rates[s] * time_step / 2 + 2 * time_step * diffusion_coeffs[s] / (shape * shape);
	const real_t c = a;

	for (index_t i = get_local_id(0); i < x_size; i += get_local_size(0))
	{
		if (i == 0)
		{
			as[i] = 0;
			bs[i] = b0;
			cs[i] = c;
			ds[i] =
				dirichlet_conditions_min[s] ? dirichlet_values_min[s] : densities[y * x_size * substrates_count + s];
		}
		else if (i == x_size - 1)
		{
			as[i] = a;
			bs[i] = b0;
			cs[i] = 0;
			ds[i] = dirichlet_conditions_max[s] ? dirichlet_values_max[s]
												: densities[(y * x_size + x_size - 1) * substrates_count + s];
		}
		else
		{
			as[i] = a;
			bs[i] = b;
			cs[i] = c;
			ds[i] = densities[(y * x_size + i) * substrates_count + s];
		}
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	index_t step;
	for (step = 2; step <= x_size; step *= 2)
	{
		for (index_t x = step - 1 + get_local_id(0) * step; x < x_size; x += get_local_size(0) * step)
		{
			const index_t above = x - step / 2;
			const index_t below = x + step / 2;

			const real_t alpha = as[x] / bs[above];
			const real_t gamma = cs[x] / bs[below];


			if (below >= x_size)
			{
				as[x] = -as[above] * alpha;
				bs[x] = bs[x] - cs[above] * alpha;
				cs[x] = 0;
				ds[x] = ds[x] - ds[above] * alpha;
			}
			else
			{
				as[x] = -as[above] * alpha;
				bs[x] = bs[x] - cs[above] * alpha - as[below] * gamma;
				cs[x] = -cs[below] * gamma;
				ds[x] = ds[x] - ds[above] * alpha - ds[below] * gamma;
			}

			//// print local_id, x, above, below
			// printf("F y: %d step: %d id: %d x: %d -:%d +:%d a:%f b:%f c:%f d:%f\n", (int)y, (int)step,
			//	   (int)get_local_id(0), (int)x, (int)above, (int)below, as[x], bs[x], cs[x], ds[x]);
		}

		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}

	step /= 2;
	if (get_local_id(0) == 0)
	{
		ds[step - 1] /= bs[step - 1];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	for (step /= 2; step >= 1; step /= 2)
	{
		for (index_t x = step - 1 + get_local_id(0) * step * 2; x < x_size; x += get_local_size(0) * step * 2)
		{
			const index_t above = x - step;
			const index_t below = x + step;


			if (0 <= above && below < x_size)
			{
				ds[x] = (ds[x] - ds[above] * as[x] - ds[below] * cs[x]) / bs[x];
			}
			else if (above < 0)
			{
				ds[x] = (ds[x] - ds[below] * cs[x]) / bs[x];
			}
			else
			{
				ds[x] = (ds[x] - ds[above] * as[x]) / bs[x];
			}

			// printf("B y: %d step: %d id: %d x: %d -:%d +:%d a:%f b:%f c:%f d:%f\n", (int)y, (int)step,
			//	   (int)get_local_id(0), (int)x, (int)above, (int)below, as[x], bs[x], cs[x], (double)ds[x]);
		}

		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (index_t i = get_local_id(0); i < x_size; i += get_local_size(0))
	{
		densities[(y * x_size + i) * substrates_count + s] = ds[i];
		// printf("y: %d x: %d %f\n", (int)y, (int)i, ds[i]);
	}
}



kernel void solve_slice_2d_y_m(global real_t* restrict densities, global const real_t* diffusion_coeffs,
							   global const real_t* decay_rates, constant bool* restrict dirichlet_conditions_min,
							   constant real_t* restrict dirichlet_values_min,
							   constant bool* restrict dirichlet_conditions_max,
							   constant real_t* restrict dirichlet_values_max, real_t time_step, real_t shape,
							   index_t substrates_count, index_t x_size, index_t y_size)
{
	__local real_t shared[1000];
	__local real_t* as = shared;
	__local real_t* bs = as + 250;
	__local real_t* cs = bs + 250;
	__local real_t* ds = cs + 250;

	const index_t id = get_global_id(0) / get_local_size(0);
	const index_t x = id / substrates_count;
	const index_t s = id % substrates_count;

	const real_t a = -time_step * diffusion_coeffs[s] / (shape * shape);
	const real_t b0 = 1 + decay_rates[s] * time_step / 2 + time_step * diffusion_coeffs[s] / (shape * shape);
	const real_t b = 1 + decay_rates[s] * time_step / 2 + 2 * time_step * diffusion_coeffs[s] / (shape * shape);
	const real_t c = a;

	for (index_t i = get_local_id(0); i < y_size; i += get_local_size(0))
	{
		if (i == 0)
		{
			as[i] = 0;
			bs[i] = b0;
			cs[i] = c;
			ds[i] = dirichlet_conditions_min[s] ? dirichlet_values_min[s] : densities[x * substrates_count + s];
		}
		else if (i == y_size - 1)
		{
			as[i] = a;
			bs[i] = b0;
			cs[i] = 0;
			ds[i] = dirichlet_conditions_max[s] ? dirichlet_values_max[s]
												: densities[((y_size - 1) * y_size + x) * substrates_count + s];
		}
		else
		{
			as[i] = a;
			bs[i] = b;
			cs[i] = c;
			ds[i] = densities[(i * y_size + x) * substrates_count + s];
		}
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	index_t step;
	for (step = 2; step <= y_size; step *= 2)
	{
		for (index_t x = step - 1 + get_local_id(0) * step; x < y_size; x += get_local_size(0) * step)
		{
			const index_t above = x - step / 2;
			const index_t below = x + step / 2;

			const real_t alpha = as[x] / bs[above];
			const real_t gamma = cs[x] / bs[below];


			if (below >= y_size)
			{
				as[x] = -as[above] * alpha;
				bs[x] = bs[x] - cs[above] * alpha;
				cs[x] = 0;
				ds[x] = ds[x] - ds[above] * alpha;
			}
			else
			{
				as[x] = -as[above] * alpha;
				bs[x] = bs[x] - cs[above] * alpha - as[below] * gamma;
				cs[x] = -cs[below] * gamma;
				ds[x] = ds[x] - ds[above] * alpha - ds[below] * gamma;
			}

			//// print local_id, x, above, below
			// printf("F y: %d step: %d id: %d x: %d -:%d +:%d a:%f b:%f c:%f d:%f\n", (int)y, (int)step,
			//	   (int)get_local_id(0), (int)x, (int)above, (int)below, as[x], bs[x], cs[x], ds[x]);
		}

		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}

	step /= 2;
	if (get_local_id(0) == 0)
	{
		ds[step - 1] /= bs[step - 1];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	for (step /= 2; step >= 1; step /= 2)
	{
		for (index_t x = step - 1 + get_local_id(0) * step * 2; x < y_size; x += get_local_size(0) * step * 2)
		{
			const index_t above = x - step;
			const index_t below = x + step;


			if (0 <= above && below < y_size)
			{
				ds[x] = (ds[x] - ds[above] * as[x] - ds[below] * cs[x]) / bs[x];
			}
			else if (above < 0)
			{
				ds[x] = (ds[x] - ds[below] * cs[x]) / bs[x];
			}
			else
			{
				ds[x] = (ds[x] - ds[above] * as[x]) / bs[x];
			}

			// printf("B y: %d step: %d id: %d x: %d -:%d +:%d a:%f b:%f c:%f d:%f\n", (int)y, (int)step,
			//	   (int)get_local_id(0), (int)x, (int)above, (int)below, as[x], bs[x], cs[x], (double)ds[x]);
		}

		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (index_t i = get_local_id(0); i < y_size; i += get_local_size(0))
	{
		densities[(i * y_size + x) * substrates_count + s] = ds[i];
		// printf("y: %d x: %d %f\n", (int)y, (int)i, ds[i]);
	}
}

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
