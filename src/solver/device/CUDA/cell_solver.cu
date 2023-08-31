#include <cuda/atomic>

#include "types.h"

using namespace biofvm;

__device__ void compute_position_1d(const real_t* __restrict__ position, index_t x_min, index_t x_dt,
									index_t* __restrict__ x)
{
	*x = (index_t)((position[0] - x_min) / x_dt);
}

__device__ index_t compute_index_1d(const real_t* __restrict__ position, index_t x_min, index_t x_dt, index_t x_size)
{
	index_t x = (index_t)((position[0] - x_min) / x_dt);

	return x;
}

__device__ void compute_position_2d(const real_t* __restrict__ position, index_t x_min, index_t y_min, index_t x_dt,
									index_t y_dt, index_t* __restrict__ x, index_t* __restrict__ y)
{
	*x = (index_t)((position[0] - x_min) / x_dt);
	*y = (index_t)((position[1] - y_min) / y_dt);
}

__device__ index_t compute_index_2d(const real_t* __restrict__ position, index_t x_min, index_t y_min, index_t x_dt,
									index_t y_dt, index_t x_size, index_t y_size)
{
	index_t x = (index_t)((position[0] - x_min) / x_dt);
	index_t y = (index_t)((position[1] - y_min) / y_dt);

	return x + y * x_size;
}

__device__ void compute_position_3d(const real_t* __restrict__ position, index_t x_min, index_t y_min, index_t z_min,
									index_t x_dt, index_t y_dt, index_t z_dt, index_t* __restrict__ x,
									index_t* __restrict__ y, index_t* __restrict__ z)
{
	*x = (index_t)((position[0] - x_min) / x_dt);
	*y = (index_t)((position[1] - y_min) / y_dt);
	*z = (index_t)((position[2] - z_min) / z_dt);
}

__device__ index_t compute_index_3d(const real_t* __restrict__ position, index_t x_min, index_t y_min, index_t z_min,
									index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size,
									index_t z_size)
{
	index_t x = (index_t)((position[0] - x_min) / x_dt);
	index_t y = (index_t)((position[1] - y_min) / y_dt);
	index_t z = (index_t)((position[2] - z_min) / z_dt);

	return x + y * x_size + z * x_size * y_size;
}

__device__ index_t compute_index(const real_t* __restrict__ position, index_t x_min, index_t y_min, index_t z_min,
								 index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size,
								 index_t z_size, index_t dims)
{
	if (dims == 1)
		return compute_index_1d(position, x_min, x_dt, x_size);
	else if (dims == 2)
		return compute_index_2d(position, x_min, y_min, x_dt, y_dt, x_size, y_size);
	else if (dims == 3)
		return compute_index_3d(position, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size, y_size, z_size);
	return 0;
}

__global__ void clear_and_ballot(const real_t* __restrict__ cell_positions, index_t* __restrict__ ballots,
								 real_t* __restrict__ reduced_numerators, real_t* __restrict__ reduced_denominators,
								 real_t* __restrict__ reduced_factors, int* __restrict__ conflicts,
								 int* __restrict__ conflicts_wrk, index_t n, index_t substrates_count, index_t x_min,
								 index_t y_min, index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size,
								 index_t y_size, index_t z_size, index_t dims)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t idx =
		compute_index(cell_positions + i * dims, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size, y_size, z_size, dims);

	if (s == 0)
	{
		cuda::atomic_ref<index_t, cuda::thread_scope_device>(ballots[idx]).store(i, cuda::memory_order_relaxed);
		conflicts[i] = 0;
	}

	conflicts_wrk[i * substrates_count + s] = 0;
	reduced_numerators[i * substrates_count + s] = 0;
	reduced_denominators[i * substrates_count + s] = 0;
	reduced_factors[i * substrates_count + s] = 0;
}

__global__ void ballot_and_sum(real_t* __restrict__ reduced_numerators, real_t* __restrict__ reduced_denominators,
							   real_t* __restrict__ reduced_factors, real_t* __restrict__ numerators,
							   real_t* __restrict__ denominators, real_t* __restrict__ factors,
							   const real_t* __restrict__ secretion_rates, const real_t* __restrict__ uptake_rates,
							   const real_t* __restrict__ saturation_densities,
							   const real_t* __restrict__ net_export_rates, const real_t* __restrict__ cell_volumes,
							   const real_t* __restrict__ cell_positions, index_t* __restrict__ ballots,
							   index_t* __restrict__ conflicts, index_t* __restrict__ conflicts_wrk, index_t n,
							   real_t voxel_volume, real_t time_step, index_t substrates_count, index_t x_min,
							   index_t y_min, index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size,
							   index_t y_size, index_t z_size, index_t dims)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t idx = ballots[compute_index(cell_positions + i * dims, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size,
										y_size, z_size, dims)];

	index_t add_one = idx == i ? 1 : 0;

	if (s == 0)
		cuda::atomic_ref<index_t, cuda::thread_scope_device>(conflicts[idx]).fetch_add(1, cuda::memory_order_relaxed);

	cuda::atomic_ref<index_t, cuda::thread_scope_device>(conflicts_wrk[idx * substrates_count + s])
		.fetch_add(1, cuda::memory_order_relaxed);

	float num = secretion_rates[i * substrates_count + s] * saturation_densities[i * substrates_count + s] * time_step
				* cell_volumes[i] / voxel_volume;

	float denom = (uptake_rates[i * substrates_count + s] + secretion_rates[i * substrates_count + s]) * time_step
				  * cell_volumes[i] / voxel_volume;

	float factor = net_export_rates[i * substrates_count + s] * time_step / voxel_volume;

	numerators[i * substrates_count + s] = num;
	denominators[i * substrates_count + s] = denom;
	factors[i * substrates_count + s] = factor;

	cuda::atomic_ref<real_t, cuda::thread_scope_device>(reduced_numerators[idx * substrates_count + s])
		.fetch_add(num, cuda::memory_order_relaxed);
	cuda::atomic_ref<real_t, cuda::thread_scope_device>(reduced_denominators[idx * substrates_count + s])
		.fetch_add(denom + add_one, cuda::memory_order_relaxed);
	cuda::atomic_ref<real_t, cuda::thread_scope_device>(reduced_factors[idx * substrates_count + s])
		.fetch_add(factor, cuda::memory_order_relaxed);
}

__global__ void compute_densities_1d(real_t* __restrict__ substrate_densities, const real_t* __restrict__ numerator,
									 const real_t* __restrict__ denominator, const real_t* __restrict__ factor,
									 const real_t* __restrict__ cell_positions, const index_t* __restrict__ ballots,
									 index_t n, real_t voxel_volume, index_t substrates_count, index_t x_min,
									 index_t x_dt, index_t x_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

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

__global__ void compute_densities_2d(real_t* __restrict__ substrate_densities, const real_t* __restrict__ numerator,
									 const real_t* __restrict__ denominator, const real_t* __restrict__ factor,
									 const real_t* __restrict__ cell_positions, const index_t* __restrict__ ballots,
									 index_t n, real_t voxel_volume, index_t substrates_count, index_t x_min,
									 index_t y_min, index_t x_dt, index_t y_dt, index_t x_size, index_t y_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

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

__global__ void compute_densities_3d(real_t* __restrict__ substrate_densities, const real_t* __restrict__ numerator,
									 const real_t* __restrict__ denominator, const real_t* __restrict__ factor,
									 const real_t* __restrict__ cell_positions, const index_t* __restrict__ ballots,
									 index_t n, real_t voxel_volume, index_t substrates_count, index_t x_min,
									 index_t y_min, index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt,
									 index_t x_size, index_t y_size, index_t z_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

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

__global__ void compute_fused_1d(real_t* __restrict__ internalized_substrates, real_t* __restrict__ substrate_densities,
								 const real_t* __restrict__ numerator, const real_t* __restrict__ denominator,
								 const real_t* __restrict__ factor, const real_t* __restrict__ reduced_numerator,
								 const real_t* __restrict__ reduced_denominator,
								 const real_t* __restrict__ reduced_factor, const real_t* __restrict__ cell_positions,
								 const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
								 real_t voxel_volume, index_t substrates_count, index_t x_min, index_t x_dt,
								 index_t x_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

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

	index_t idx = ballots[x];

	int val = cuda::atomic_ref<index_t, cuda::thread_scope_device>(conflicts_wrk[idx * substrates_count + s])
				  .fetch_sub(1, cuda::memory_order_acq_rel);

	if (val != 1)
		return;

	substrate_densities[x * substrates_count + s] =
		(substrate_densities[x * substrates_count + s] + reduced_numerator[idx * substrates_count + s])
			/ reduced_denominator[idx * substrates_count + s]
		+ reduced_factor[idx * substrates_count + s];

	conflicts_wrk[idx * substrates_count + s] = conflicts[idx];
}

__global__ void compute_fused_2d(real_t* __restrict__ internalized_substrates, real_t* __restrict__ substrate_densities,
								 const real_t* __restrict__ numerator, const real_t* __restrict__ denominator,
								 const real_t* __restrict__ factor, const real_t* __restrict__ reduced_numerator,
								 const real_t* __restrict__ reduced_denominator,
								 const real_t* __restrict__ reduced_factor, const real_t* __restrict__ cell_positions,
								 const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
								 real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
								 index_t x_dt, index_t y_dt, index_t x_size, index_t y_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

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

	index_t idx = ballots[y * x_size + x];

	int val = cuda::atomic_ref<index_t, cuda::thread_scope_device>(conflicts_wrk[idx * substrates_count + s])
				  .fetch_sub(1, cuda::memory_order_acq_rel);

	if (val != 1)
		return;

	substrate_densities[(y * x_size + x) * substrates_count + s] =
		(substrate_densities[(y * x_size + x) * substrates_count + s] + reduced_numerator[idx * substrates_count + s])
			/ reduced_denominator[idx * substrates_count + s]
		+ reduced_factor[idx * substrates_count + s];

	conflicts_wrk[idx * substrates_count + s] = conflicts[idx];
}

__global__ void compute_fused_3d(real_t* __restrict__ internalized_substrates, real_t* __restrict__ substrate_densities,
								 const real_t* __restrict__ numerator, const real_t* __restrict__ denominator,
								 const real_t* __restrict__ factor, const real_t* __restrict__ reduced_numerator,
								 const real_t* __restrict__ reduced_denominator,
								 const real_t* __restrict__ reduced_factor, const real_t* __restrict__ cell_positions,
								 const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
								 real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
								 index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size,
								 index_t y_size, index_t z_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

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

	index_t idx = ballots[(z * y_size + y) * x_size + x];

	int val = cuda::atomic_ref<index_t, cuda::thread_scope_device>(conflicts_wrk[idx * substrates_count + s])
				  .fetch_sub(1, cuda::memory_order_acq_rel);

	if (val != 1)
		return;

	substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s] =
		(substrate_densities[((z * y_size + y) * x_size + x) * substrates_count + s]
		 + reduced_numerator[idx * substrates_count + s])
			/ reduced_denominator[idx * substrates_count + s]
		+ reduced_factor[idx * substrates_count + s];

	conflicts_wrk[idx * substrates_count + s] = conflicts[idx];
}

void run_clear_and_ballot(const real_t* cell_positions, index_t* ballots, real_t* reduced_numerators,
						  real_t* reduced_denominators, real_t* reduced_factors, int* conflicts, int* conflicts_wrk,
						  index_t n, index_t substrates_count, index_t x_min, index_t y_min, index_t z_min,
						  index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size,
						  index_t dims, cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n + block_size - 1) / block_size;

	clear_and_ballot<<<blocks, block_size, 0, stream>>>(
		cell_positions, ballots, reduced_numerators, reduced_denominators, reduced_factors, conflicts, conflicts_wrk, n,
		substrates_count, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size, y_size, z_size, dims);
}

void run_ballot_and_sum(real_t* reduced_numerators, real_t* reduced_denominators, real_t* reduced_factors,
						real_t* numerators, real_t* denominators, real_t* factors, const real_t* secretion_rates,
						const real_t* uptake_rates, const real_t* saturation_densities, const real_t* net_export_rates,
						const real_t* cell_volumes, const real_t* cell_positions, index_t* ballots, index_t* conflicts,
						index_t* conflicts_wrk, index_t n, real_t voxel_volume, real_t time_step,
						index_t substrates_count, index_t x_min, index_t y_min, index_t z_min, index_t x_dt,
						index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size, index_t dims,
						cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n * substrates_count + block_size - 1) / block_size;

	ballot_and_sum<<<blocks, block_size, 0, stream>>>(
		reduced_numerators, reduced_denominators, reduced_factors, numerators, denominators, factors, secretion_rates,
		uptake_rates, saturation_densities, net_export_rates, cell_volumes, cell_positions, ballots, conflicts,
		conflicts_wrk, n, voxel_volume, time_step, substrates_count, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size,
		y_size, z_size, dims);
}

void run_compute_densities_1d(real_t* substrate_densities, const real_t* numerator, const real_t* denominator,
							  const real_t* factor, const real_t* cell_positions, const index_t* ballots, index_t n,
							  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t x_dt,
							  index_t x_size, cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n * substrates_count + block_size - 1) / block_size;

	compute_densities_1d<<<blocks, block_size, 0, stream>>>(substrate_densities, numerator, denominator, factor,
															cell_positions, ballots, n, voxel_volume, substrates_count,
															x_min, x_dt, x_size);
}

void run_compute_densities_2d(real_t* substrate_densities, const real_t* numerator, const real_t* denominator,
							  const real_t* factor, const real_t* cell_positions, const index_t* ballots, index_t n,
							  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min, index_t x_dt,
							  index_t y_dt, index_t x_size, index_t y_size, cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n * substrates_count + block_size - 1) / block_size;

	compute_densities_2d<<<blocks, block_size, 0, stream>>>(substrate_densities, numerator, denominator, factor,
															cell_positions, ballots, n, voxel_volume, substrates_count,
															x_min, y_min, x_dt, y_dt, x_size, y_size);
}

void run_compute_densities_3d(real_t* substrate_densities, const real_t* numerator, const real_t* denominator,
							  const real_t* factor, const real_t* cell_positions, const index_t* ballots, index_t n,
							  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min,
							  index_t z_min, index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size,
							  index_t z_size, cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n * substrates_count + block_size - 1) / block_size;

	compute_densities_3d<<<blocks, block_size, 0, stream>>>(
		substrate_densities, numerator, denominator, factor, cell_positions, ballots, n, voxel_volume, substrates_count,
		x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size, y_size, z_size);
}

void run_compute_fused_1d(real_t* internalized_substrates, real_t* substrate_densities, const real_t* numerator,
						  const real_t* denominator, const real_t* factor, const real_t* reduced_numerator,
						  const real_t* reduced_denominator, const real_t* reduced_factor, const real_t* cell_positions,
						  const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
						  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t x_dt, index_t x_size,
						  cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n * substrates_count + block_size - 1) / block_size;

	compute_fused_1d<<<blocks, block_size, 0, stream>>>(
		internalized_substrates, substrate_densities, numerator, denominator, factor, reduced_numerator,
		reduced_denominator, reduced_factor, cell_positions, ballots, conflicts, conflicts_wrk, n, voxel_volume,
		substrates_count, x_min, x_dt, x_size);
}

void run_compute_fused_2d(real_t* internalized_substrates, real_t* substrate_densities, const real_t* numerator,
						  const real_t* denominator, const real_t* factor, const real_t* reduced_numerator,
						  const real_t* reduced_denominator, const real_t* reduced_factor, const real_t* cell_positions,
						  const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
						  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min, index_t x_dt,
						  index_t y_dt, index_t x_size, index_t y_size, cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n * substrates_count + block_size - 1) / block_size;

	compute_fused_2d<<<blocks, block_size, 0, stream>>>(
		internalized_substrates, substrate_densities, numerator, denominator, factor, reduced_numerator,
		reduced_denominator, reduced_factor, cell_positions, ballots, conflicts, conflicts_wrk, n, voxel_volume,
		substrates_count, x_min, y_min, x_dt, y_dt, x_size, y_size);
}

void run_compute_fused_3d(real_t* internalized_substrates, real_t* substrate_densities, const real_t* numerator,
						  const real_t* denominator, const real_t* factor, const real_t* reduced_numerator,
						  const real_t* reduced_denominator, const real_t* reduced_factor, const real_t* cell_positions,
						  const index_t* ballots, const int* conflicts, index_t* conflicts_wrk, index_t n,
						  real_t voxel_volume, index_t substrates_count, index_t x_min, index_t y_min, index_t z_min,
						  index_t x_dt, index_t y_dt, index_t z_dt, index_t x_size, index_t y_size, index_t z_size,
						  cudaStream_t& stream)
{
	int block_size = 256;
	int blocks = (n * substrates_count + block_size - 1) / block_size;

	compute_fused_3d<<<blocks, block_size, 0, stream>>>(
		internalized_substrates, substrate_densities, numerator, denominator, factor, reduced_numerator,
		reduced_denominator, reduced_factor, cell_positions, ballots, conflicts, conflicts_wrk, n, voxel_volume,
		substrates_count, x_min, y_min, z_min, x_dt, y_dt, z_dt, x_size, y_size, z_size);
}
