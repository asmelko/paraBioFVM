typedef int index_t;

#ifdef DOUBLE

	#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
	#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef double real_t;
typedef atomic_double atomic_real_t;
#else
typedef float real_t;
typedef atomic_float atomic_real_t;
#endif

#define no_ballot -1

#ifndef __opencl_c_ext_fp32_global_atomic_add

	#ifdef NVIDIA
float atomic_fetch_add_explicit(__global float* p, float val, int memory_order)
{
	float prev;
	asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(prev) : "l"(p), "f"(val) : "memory");
	return prev;
}
	#else
void atomic_fetch_add_explicit(__global atomic_float* source, float val, int memory_order)
{
	float expected = atomic_load_explicit(source, memory_order);
	while (!atomic_compare_exchange_weak_explicit(source, &expected, expected + val, memory_order, memory_order)) {};
}
	#endif

#endif

#ifndef __opencl_c_ext_fp64_global_atomic_add

	#ifdef NVIDIA
double __attribute__((overloadable)) atomic_fetch_add_explicit(__global double* p, double val, int memory_order)
{
	double prev;
	asm volatile("atom.global.add.f64 %0, [%1], %2;" : "=d"(prev) : "l"(p), "d"(val) : "memory");
	return prev;
}
	#else
void __attribute__((overloadable))
atomic_fetch_add_explicit(__global atomic_double* source, double val, int memory_order)
{
	double expected = atomic_load_explicit(source, memory_order);
	while (!atomic_compare_exchange_weak_explicit(source, &expected, expected + val, memory_order, memory_order)) {};
}
	#endif

#endif

void compute_position_1d(global const real_t* restrict position, index_t x_min, index_t x_dt, index_t* restrict x)
{
	*x = (index_t)((position[0] - x_min) / x_dt);
}

index_t compute_index_1d(global const real_t* restrict position, index_t x_min, index_t x_dt, index_t x_size)
{
	index_t x = (index_t)((position[0] - x_min) / x_dt);

	return x;
}

void compute_position_2d(global const real_t* restrict position, index_t x_min, index_t y_min, index_t x_dt,
						 index_t y_dt, index_t* restrict x, index_t* restrict y)
{
	*x = (index_t)((position[0] - x_min) / x_dt);
	*y = (index_t)((position[1] - y_min) / y_dt);
}

index_t compute_index_2d(global const real_t* restrict position, index_t x_min, index_t y_min, index_t x_dt,
						 index_t y_dt, index_t x_size, index_t y_size)
{
	index_t x = (index_t)((position[0] - x_min) / x_dt);
	index_t y = (index_t)((position[1] - y_min) / y_dt);

	return x + y * x_size;
}

void compute_position_3d(global const real_t* restrict position, index_t x_min, index_t y_min, index_t z_min,
						 index_t x_dt, index_t y_dt, index_t z_dt, index_t* restrict x, index_t* restrict y,
						 index_t* restrict z)
{
	*x = (index_t)((position[0] - x_min) / x_dt);
	*y = (index_t)((position[1] - y_min) / y_dt);
	*z = (index_t)((position[2] - z_min) / z_dt);
}

index_t compute_index_3d(global const real_t* restrict position, index_t x_min, index_t y_min, index_t z_min,
						 index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size)
{
	index_t x = (index_t)((position[0] - x_min) / x_dt);
	index_t y = (index_t)((position[1] - y_min) / y_dt);
	index_t z = (index_t)((position[2] - z_min) / z_dt);

	return x + y * x_size + z * x_size * y_size;
}

index_t compute_index(global const real_t* restrict position, index_t x_min, index_t y_min, index_t z_min, index_t x_dt,
					  index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size, index_t dims)
{
	if (dims == 1)
		return compute_index_1d(position, x_min, x_dt, x_size);
	else if (dims == 2)
		return compute_index_2d(position, x_min, y_min, x_dt, y_dt, x_size, y_size);
	else if (dims == 3)
		return compute_index_3d(position, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size, y_size, z_size);
	return 0;
}

kernel void clear_and_ballot(global const real_t* restrict cell_positions, global atomic_int* restrict ballots,
							 global real_t* restrict reduced_numerators, global real_t* restrict reduced_denominators,
							 global real_t* restrict reduced_factors, global int* restrict is_conflict,
							 index_t substrates_count, index_t x_min, index_t y_min, index_t z_min, index_t x_dt,
							 index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size, index_t dims)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	if (s == 0)
	{
		index_t idx = compute_index(cell_positions + i * dims, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size, y_size,
									z_size, dims);

		atomic_store_explicit(ballots + idx, i, memory_order_relaxed);

		if (i == 0)
			is_conflict[0] = 0;
	}

	reduced_numerators[i * substrates_count + s] = 0;
	reduced_denominators[i * substrates_count + s] = 0;
	reduced_factors[i * substrates_count + s] = 0;
}

kernel void compute_intermediates(global real_t* restrict numerators, global real_t* restrict denominators,
								  global real_t* restrict factors, global const real_t* restrict secretion_rates,
								  global const real_t* restrict uptake_rates,
								  global const real_t* restrict saturation_densities,
								  global const real_t* restrict net_export_rates,
								  global const real_t* restrict cell_volumes, real_t voxel_volume, real_t time_step,
								  index_t substrates_count)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	numerators[i * substrates_count + s] = secretion_rates[i * substrates_count + s]
										   * saturation_densities[i * substrates_count + s] * time_step
										   * cell_volumes[i] / voxel_volume;

	denominators[i * substrates_count + s] =
		(uptake_rates[i * substrates_count + s] + secretion_rates[i * substrates_count + s]) * time_step
		* cell_volumes[i] / voxel_volume;

	factors[i * substrates_count + s] = net_export_rates[i * substrates_count + s] * time_step / voxel_volume;
}

kernel void ballot_and_sum(global atomic_real_t* restrict reduced_numerators,
						   global atomic_real_t* restrict reduced_denominators,
						   global atomic_real_t* restrict reduced_factors, global const real_t* restrict numerators,
						   global const real_t* restrict denominators, global const real_t* restrict factors,
						   global const real_t* restrict cell_positions, global index_t* restrict ballots,
						   global atomic_int* restrict is_conflict, index_t substrates_count, index_t x_min,
						   index_t y_min, index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size,
						   index_t y_size, index_t z_size, index_t dims)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t idx = ballots[compute_index(cell_positions + i * dims, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size,
										y_size, z_size, dims)];

	index_t add_one = idx == i ? 1 : 0;

	if (idx != i)
	{
		atomic_store_explicit(is_conflict, 1, memory_order_relaxed);
	}

	atomic_fetch_add_explicit(reduced_numerators + idx * substrates_count + s, numerators[i * substrates_count + s],
							  memory_order_relaxed);
	atomic_fetch_add_explicit(reduced_denominators + idx * substrates_count + s,
							  denominators[i * substrates_count + s] + add_one, memory_order_relaxed);
	atomic_fetch_add_explicit(reduced_factors + idx * substrates_count + s, factors[i * substrates_count + s],
							  memory_order_relaxed);
}

kernel void compute_internalized_1d(global real_t* restrict internalized_substrates,
									global const real_t* restrict substrate_densities,
									global const real_t* restrict numerator, global const real_t* restrict denominator,
									global const real_t* restrict factor, global const real_t* restrict cell_positions,
									real_t voxel_volume, index_t substrates_count, index_t x_min, index_t x_dt,
									index_t x_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x;

	compute_position_1d(cell_positions + i, x_min, x_dt, &x);

	internalized_substrates[i * substrates_count + s] -=
		voxel_volume
		* ((-denominator[i * substrates_count + s] * substrate_densities[x * substrates_count + s]
			+ numerator[i * substrates_count + s])
			   / (1 + denominator[i * substrates_count + s])
		   + factor[i * substrates_count + s]);
}

kernel void compute_internalized_2d(global real_t* restrict internalized_substrates,
									global const real_t* restrict substrate_densities,
									global const real_t* restrict numerator, global const real_t* restrict denominator,
									global const real_t* restrict factor, global const real_t* restrict cell_positions,
									real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
									index_t x_dt, index_t y_dt, index_t x_size, index_t y_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x, y;

	compute_position_2d(cell_positions + i * 2, x_min, y_min, x_dt, y_dt, &x, &y);

	internalized_substrates[i * substrates_count + s] -=
		voxel_volume
		* ((-denominator[i * substrates_count + s] * substrate_densities[(y * x_size + x) * substrates_count + s]
			+ numerator[i * substrates_count + s])
			   / (1 + denominator[i * substrates_count + s])
		   + factor[i * substrates_count + s]);
}

kernel void compute_internalized_3d(global real_t* restrict internalized_substrates,
									global const real_t* restrict substrate_densities,
									global const real_t* restrict numerator, global const real_t* restrict denominator,
									global const real_t* restrict factor, global const real_t* restrict cell_positions,
									real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
									index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size,
									index_t y_size, index_t z_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x, y, z;

	compute_position_3d(cell_positions + i * 3, x_min, y_min, z_min, x_dt, y_dt, z_dt, &x, &y, &z);

	internalized_substrates[i * substrates_count + s] -=
		voxel_volume
		* ((-denominator[i * substrates_count + s]
				* substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s]
			+ numerator[i * substrates_count + s])
			   / (1 + denominator[i * substrates_count + s])
		   + factor[i * substrates_count + s]);
}

kernel void compute_densities_1d(global real_t* restrict substrate_densities, global const real_t* restrict numerator,
								 global const real_t* restrict denominator, global const real_t* restrict factor,
								 global const real_t* restrict cell_positions, global const index_t* restrict ballots,
								 real_t voxel_volume, index_t substrates_count, index_t x_min, index_t x_dt,
								 index_t x_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x;

	compute_position_1d(cell_positions + i, x_min, x_dt, &x);

	index_t idx = ballots[x];

	if (idx != i)
		return;

	substrate_densities[x * substrates_count + s] =
		(substrate_densities[x * substrates_count + s] + numerator[i * substrates_count + s])
			/ denominator[i * substrates_count + s]
		+ factor[i * substrates_count + s];
}

kernel void compute_densities_2d(global real_t* restrict substrate_densities, global const real_t* restrict numerator,
								 global const real_t* restrict denominator, global const real_t* restrict factor,
								 global const real_t* restrict cell_positions, global const index_t* restrict ballots,
								 real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
								 index_t x_dt, index_t y_dt, index_t x_size, index_t y_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x, y;

	compute_position_2d(cell_positions + i * 2, x_min, y_min, x_dt, y_dt, &x, &y);

	index_t idx = ballots[y * x_size + x];

	if (idx != i)
		return;

	substrate_densities[(y * x_size + x) * substrates_count + s] =
		(substrate_densities[(y * x_size + x) * substrates_count + s] + numerator[i * substrates_count + s])
			/ denominator[i * substrates_count + s]
		+ factor[i * substrates_count + s];
}

kernel void compute_densities_3d(global real_t* restrict substrate_densities, global const real_t* restrict numerator,
								 global const real_t* restrict denominator, global const real_t* restrict factor,
								 global const real_t* restrict cell_positions, global const index_t* restrict ballots,
								 real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
								 index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size,
								 index_t y_size, index_t z_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x, y, z;

	compute_position_3d(cell_positions + i * 3, x_min, y_min, z_min, x_dt, y_dt, z_dt, &x, &y, &z);

	index_t idx = ballots[(z * y_size + y) * x_size + x];

	if (idx != i)
		return;

	substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s] =
		(substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s]
		 + numerator[i * substrates_count + s])
			/ denominator[i * substrates_count + s]
		+ factor[i * substrates_count + s];
}

kernel void compute_fused_1d(global real_t* restrict internalized_substrates,
							 global real_t* restrict substrate_densities, global const real_t* restrict numerator,
							 global const real_t* restrict denominator, global const real_t* restrict factor,
							 global const real_t* restrict cell_positions, real_t voxel_volume,
							 index_t substrates_count, index_t x_min, index_t x_dt, index_t x_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x;

	compute_position_1d(cell_positions + i, x_min, x_dt, &x);

	internalized_substrates[i * substrates_count + s] -=
		voxel_volume
		* (((1 - denominator[i * substrates_count + s]) * substrate_densities[x * substrates_count + s]
			+ numerator[i * substrates_count + s])
			   / denominator[i * substrates_count + s]
		   + factor[i * substrates_count + s]);

	substrate_densities[x * substrates_count + s] =
		(substrate_densities[x * substrates_count + s] + numerator[i * substrates_count + s])
			/ denominator[i * substrates_count + s]
		+ factor[i * substrates_count + s];
}

kernel void compute_fused_2d(global real_t* restrict internalized_substrates,
							 global real_t* restrict substrate_densities, global const real_t* restrict numerator,
							 global const real_t* restrict denominator, global const real_t* restrict factor,
							 global const real_t* restrict cell_positions, real_t voxel_volume,
							 index_t substrates_count, index_t x_min, index_t y_min, index_t x_dt, index_t y_dt,
							 index_t x_size, index_t y_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x, y;

	compute_position_2d(cell_positions + i * 2, x_min, y_min, x_dt, y_dt, &x, &y);

	internalized_substrates[i * substrates_count + s] -=
		voxel_volume
		* (((1 - denominator[i * substrates_count + s]) * substrate_densities[(y * x_size + x) * substrates_count + s]
			+ numerator[i * substrates_count + s])
			   / denominator[i * substrates_count + s]
		   + factor[i * substrates_count + s]);

	substrate_densities[(y * x_size + x) * substrates_count + s] =
		(substrate_densities[(y * x_size + x) * substrates_count + s] + numerator[i * substrates_count + s])
			/ denominator[i * substrates_count + s]
		+ factor[i * substrates_count + s];
}

kernel void compute_fused_3d(global real_t* restrict internalized_substrates,
							 global real_t* restrict substrate_densities, global const real_t* restrict numerator,
							 global const real_t* restrict denominator, global const real_t* restrict factor,
							 global const real_t* restrict cell_positions, real_t voxel_volume,
							 index_t substrates_count, index_t x_min, index_t y_min, index_t z_min, index_t x_dt,
							 index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size)
{
	int id = get_global_id(0);
	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x, y, z;

	compute_position_3d(cell_positions + i * 3, x_min, y_min, z_min, x_dt, y_dt, z_dt, &x, &y, &z);

	internalized_substrates[i * substrates_count + s] -=
		voxel_volume
		* (((1 - denominator[i * substrates_count + s])
				* substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s]
			+ numerator[i * substrates_count + s])
			   / denominator[i * substrates_count + s]
		   + factor[i * substrates_count + s]);

	substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s] =
		(substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s]
		 + numerator[i * substrates_count + s])
			/ denominator[i * substrates_count + s]
		+ factor[i * substrates_count + s];
}
