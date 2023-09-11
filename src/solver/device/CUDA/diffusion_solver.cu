#include <cooperative_groups.h>
#include <cstdio>
#include <iostream>

#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/pipeline>

#include "types.h"

using namespace biofvm;

__global__ void solve_slice_2d_x(real_t* __restrict__ densities, const real_t* __restrict__ b,
								 const real_t* __restrict__ c, const bool* __restrict__ dirichlet_conditions_min,
								 const real_t* __restrict__ dirichlet_values_min,
								 const bool* __restrict__ dirichlet_conditions_max,
								 const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
								 index_t x_size, index_t y_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void solve_slice_2d_x_shared_full(real_t* __restrict__ densities, const real_t* __restrict__ b,
											 const real_t* __restrict__ c,
											 const bool* __restrict__ dirichlet_conditions_min,
											 const real_t* __restrict__ dirichlet_values_min,
											 const bool* __restrict__ dirichlet_conditions_max,
											 const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
											 index_t x_size, index_t y_size, index_t padding)
{
	extern __shared__ real_t shmem[];
	real_t* densities_sh = shmem;
	real_t* b_sh = shmem
				   + ((blockDim.x + substrates_count - 1) / substrates_count)
						 * (x_size * substrates_count + padding); // padding to remove bank conflicts

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	const index_t y_min = blockIdx.x * blockDim.x / substrates_count;
	const index_t y_max = min(((blockIdx.x + 1) * blockDim.x - 1) / substrates_count, y_size - 1);

	const index_t offset = x_size * substrates_count;
	const index_t offset2 = x_size * substrates_count + padding;

	for (index_t i = threadIdx.x; i < x_size * substrates_count; i += blockDim.x)
	{
		for (index_t y = y_min, y_l = 0; y <= y_max; y++, y_l++)
			densities_sh[y_l * offset2 + i] = densities[y * offset + i];
	}

	for (index_t i = threadIdx.x; i < x_size * substrates_count; i += blockDim.x)
	{
		b_sh[i] = b[i];
	}

	index_t y = id / substrates_count - y_min;
	index_t s = id % substrates_count;

	__syncthreads();

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
			tmp *= a;
			tmp = fmaf(b_, tmp, d_);
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

	__syncthreads();

	for (index_t i = threadIdx.x; i < x_size * substrates_count; i += blockDim.x)
	{
		for (index_t y = y_min, y_l = 0; y <= y_max; y++, y_l++)
			densities[y * offset + i] = densities_sh[y_l * offset2 + i];
	}
}

__global__ void solve_slice_2d_x_block(real_t* __restrict__ densities, const real_t* __restrict__ a,
									   const real_t* __restrict__ r_fwd, const real_t* __restrict__ c_fwd,
									   const real_t* __restrict__ a_bck, const real_t* __restrict__ c_bck,
									   const real_t* __restrict__ c_rdc, const real_t* __restrict__ r_rdc,
									   const bool* __restrict__ dirichlet_conditions_min,
									   const real_t* __restrict__ dirichlet_values_min,
									   const bool* __restrict__ dirichlet_conditions_max,
									   const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
									   index_t x_size, index_t y_size, index_t block_size)
{
	int lane_id = threadIdx.x % 32;
	int warp_id = threadIdx.x / 32;
	int warps = blockDim.x / 32;

	int id = lane_id + blockIdx.x * 32;

	if (id >= y_size * substrates_count)
		return;

	index_t y = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x_blocks = (x_size + block_size - 1) / block_size;

	if (warp_id == 0)
	{
		if (dirichlet_conditions_min[s])
			densities[(y * x_size) * substrates_count + s] = dirichlet_values_min[s];

		if (dirichlet_conditions_max[s])
			densities[(y * x_size + x_size - 1) * substrates_count + s] = dirichlet_values_max[s];
	}

	__syncthreads();

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

	__syncthreads();

	if (warp_id == 0)
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

	__syncthreads();

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

__global__ void solve_slice_3d_x(real_t* __restrict__ densities, const real_t* __restrict__ b,
								 const real_t* __restrict__ c, const bool* __restrict__ dirichlet_conditions_min,
								 const real_t* __restrict__ dirichlet_values_min,
								 const bool* __restrict__ dirichlet_conditions_max,
								 const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
								 index_t x_size, index_t y_size, index_t z_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void solve_slice_2d_y(real_t* __restrict__ densities, const real_t* __restrict__ b,
								 const real_t* __restrict__ c, const bool* __restrict__ dirichlet_conditions_min,
								 const real_t* __restrict__ dirichlet_values_min,
								 const bool* __restrict__ dirichlet_conditions_max,
								 const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
								 index_t x_size, index_t y_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void solve_slice_2d_y_block(real_t* __restrict__ densities, const real_t* __restrict__ a,
									   const real_t* __restrict__ r_fwd, const real_t* __restrict__ c_fwd,
									   const real_t* __restrict__ a_bck, const real_t* __restrict__ c_bck,
									   const real_t* __restrict__ c_rdc, const real_t* __restrict__ r_rdc,
									   const bool* __restrict__ dirichlet_conditions_min,
									   const real_t* __restrict__ dirichlet_values_min,
									   const bool* __restrict__ dirichlet_conditions_max,
									   const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
									   index_t x_size, index_t y_size, index_t block_size)
{
	int lane_id = threadIdx.x % 32;
	int warp_id = threadIdx.x / 32;
	int warps = blockDim.x / 32;

	int id = lane_id + blockIdx.x * 32;

	if (id >= x_size * substrates_count)
		return;

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	index_t y_blocks = (y_size + block_size - 1) / block_size;

	if (warp_id == 0)
	{
		if (dirichlet_conditions_min[s])
			densities[x * substrates_count + s] = dirichlet_values_min[s];

		if (dirichlet_conditions_max[s])
			densities[((y_size - 1) * x_size + x) * substrates_count + s] = dirichlet_values_max[s];
	}

	__syncthreads();

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

	__syncthreads();

	if (warp_id == 0)
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

	__syncthreads();

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

__global__ void solve_slice_2d_y_shared_full(real_t* __restrict__ densities, const real_t* __restrict__ b,
											 const real_t* __restrict__ c,
											 const bool* __restrict__ dirichlet_conditions_min,
											 const real_t* __restrict__ dirichlet_values_min,
											 const bool* __restrict__ dirichlet_conditions_max,
											 const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
											 index_t x_size, index_t y_size)
{
	extern __shared__ real_t shmem[];
	real_t* densities_sh = shmem;
	real_t* b_sh = shmem + blockDim.x * y_size;

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	for (index_t i = threadIdx.x; i < y_size * substrates_count; i += blockDim.x)
	{
		b_sh[i] = b[i];
	}

	__syncthreads();

	if (id >= x_size * substrates_count)
		return;

	for (index_t y = 0; y < y_size; y++)
	{
		densities_sh[y * blockDim.x + threadIdx.x] = densities[(y * x_size + x) * substrates_count + s];
	}

	const real_t a = -c[s];

	if (dirichlet_conditions_min[s])
		densities_sh[threadIdx.x] = dirichlet_values_min[s];

	real_t tmp = densities_sh[threadIdx.x];

	for (index_t y = 1; y < y_size - 1; y++)
	{
		real_t* density = densities_sh + (y * blockDim.x) + threadIdx.x;
		const real_t b_ = b_sh[(y - 1) * substrates_count + s];
		const real_t d_ = *density;
		tmp *= a;
		tmp = fmaf(b_, tmp, d_);
		*density = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities_sh[((y_size - 1) * blockDim.x) + threadIdx.x] = dirichlet_values_max[s];

	{
		const real_t density = densities_sh[(y_size - 1) * blockDim.x + threadIdx.x];
		const real_t b_2 = b_sh[(y_size - 2) * substrates_count + s];
		const real_t b_1 = b_sh[(y_size - 1) * substrates_count + s];
		tmp = (density + a * b_2 * tmp) * b_1;
		densities[((y_size - 1) * x_size + x) * substrates_count + s] = tmp;
	}

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		const real_t density = densities_sh[(y * blockDim.x) + threadIdx.x];
		const real_t b_ = b_sh[y * substrates_count + s];
		tmp = fmaf(a, tmp, density) * b_;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}

__global__ void solve_slice_2d_y_shared_async(real_t* __restrict__ densities, const real_t* __restrict__ b,
											  const real_t* __restrict__ c,
											  const bool* __restrict__ dirichlet_conditions_min,
											  const real_t* __restrict__ dirichlet_values_min,
											  const bool* __restrict__ dirichlet_conditions_max,
											  const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
											  index_t x_size, index_t y_size)
{
	__align__(128) __shared__ real_t densities_sh[32 * 128];

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	cooperative_groups::thread_block_tile<32> group =
		cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	for (index_t y = 0; y < y_size; y++)
	{
		cooperative_groups::memcpy_async(group, densities_sh + y * 32,
										 densities + y * x_size * substrates_count + blockIdx.x * 32,
										 cuda::aligned_size_t<4>(sizeof(real_t) * 32));
	}

	cooperative_groups::wait(group);

	if (id >= x_size * substrates_count)
		return;

	const real_t a = -c[s];

	if (dirichlet_conditions_min[s])
		densities_sh[threadIdx.x] = dirichlet_values_min[s];

	real_t tmp = densities_sh[threadIdx.x];

	for (index_t y = 1; y < y_size - 1; y++)
	{
		real_t* density = densities_sh + (y * 32) + threadIdx.x;
		const real_t b_ = b[(y - 1) * substrates_count + s];
		tmp = *density + a * b_ * tmp;
		*density = tmp;
	}

	if (dirichlet_conditions_max[s])
		densities_sh[((y_size - 1) * 32) + threadIdx.x] = dirichlet_values_max[s];

	{
		const real_t density = densities_sh[(y_size - 1) * 32 + threadIdx.x];
		const real_t b_2 = b[(y_size - 2) * substrates_count + s];
		const real_t b_1 = b[(y_size - 1) * substrates_count + s];
		tmp = (density + a * b_2 * tmp) * b_1;
		densities[((y_size - 1) * x_size + x) * substrates_count + s] = tmp;
	}

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		const real_t density = densities_sh[(y * 32) + threadIdx.x];
		const real_t b_ = b[y * substrates_count + s];
		tmp = (density + a * tmp) * b_;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}

__global__ void solve_slice_2d_y_shared_async_thread(real_t* __restrict__ densities, const real_t* __restrict__ b,
													 const real_t* __restrict__ c,
													 const bool* __restrict__ dirichlet_conditions_min,
													 const real_t* __restrict__ dirichlet_values_min,
													 const bool* __restrict__ dirichlet_conditions_max,
													 const real_t* __restrict__ dirichlet_values_max,
													 index_t substrates_count, index_t x_size, index_t y_size)
{
	__align__(128) __shared__ real_t densities_sh[32 * 128];
	__align__(128) __shared__ real_t b_sh[4 * 128];

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	index_t x = id / substrates_count;
	index_t s = id % substrates_count;

	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
	auto group = cooperative_groups::this_thread();

	for (index_t y = 0; y < 1; y++)
	{
		cooperative_groups::memcpy_async(group, densities_sh + y * 32 + threadIdx.x,
										 densities + y * x_size * substrates_count + blockIdx.x * 32 + threadIdx.x,
										 cuda::aligned_size_t<4>(sizeof(real_t)));
	}

	const real_t a = -c[s];

	cooperative_groups::wait(group);

	if (dirichlet_conditions_min[s])
		densities_sh[threadIdx.x] = dirichlet_values_min[s];

	real_t tmp = densities_sh[threadIdx.x];

	for (index_t y = 1; y < 2; y++)
	{
		pipeline.producer_acquire();
		cuda::memcpy_async(group, densities_sh + y * 32 + threadIdx.x,
						   densities + y * x_size * substrates_count + blockIdx.x * 32 + threadIdx.x,
						   cuda::aligned_size_t<4>(sizeof(real_t)), pipeline);

		cuda::memcpy_async(group, b_sh + (y - 1) * substrates_count + threadIdx.x,
						   b + (y - 1) * substrates_count + threadIdx.x, cuda::aligned_size_t<4>(sizeof(real_t)),
						   pipeline);

		pipeline.producer_commit();
	}

	for (index_t y = 1; y < y_size - 1; y++)
	{
		pipeline.producer_acquire();
		cuda::memcpy_async(group, densities_sh + (y + 1) * 32 + threadIdx.x,
						   densities + y * x_size * substrates_count + blockIdx.x * 32 + threadIdx.x,
						   cuda::aligned_size_t<4>(sizeof(real_t)), pipeline);

		cuda::memcpy_async(group, b_sh + (y - 1) * substrates_count + threadIdx.x,
						   b + (y - 1) * substrates_count + threadIdx.x, cuda::aligned_size_t<4>(sizeof(real_t)),
						   pipeline);

		pipeline.producer_commit();

		cuda::pipeline_consumer_wait_prior<1>(pipeline);

		real_t* density = densities_sh + (y * 32) + threadIdx.x;
		const real_t b_ = b_sh[(y - 1) * substrates_count + s];
		tmp = *density + a * b_ * tmp;
		*density = tmp;

		pipeline.consumer_release();
	}

	pipeline.consumer_wait();
	pipeline.consumer_release();

	if (dirichlet_conditions_max[s])
		densities_sh[((y_size - 1) * 32) + threadIdx.x] = dirichlet_values_max[s];

	{
		const real_t density = densities_sh[(y_size - 1) * 32 + threadIdx.x];
		const real_t b_2 = b[(y_size - 2) * substrates_count + s];
		const real_t b_1 = b[(y_size - 1) * substrates_count + s];
		tmp = (density + a * b_2 * tmp) * b_1;
		densities[((y_size - 1) * x_size + x) * substrates_count + s] = tmp;
	}

	for (index_t y = y_size - 2; y >= 0; y--)
	{
		const real_t density = densities_sh[(y * 32) + threadIdx.x];
		const real_t b_ = b_sh[y * substrates_count + s];
		tmp = (density + a * tmp) * b_;
		densities[(y * x_size + x) * substrates_count + s] = tmp;
	}
}

__global__ void solve_slice_3d_y(real_t* __restrict__ densities, const real_t* __restrict__ b,
								 const real_t* __restrict__ c, const bool* __restrict__ dirichlet_conditions_min,
								 const real_t* __restrict__ dirichlet_values_min,
								 const bool* __restrict__ dirichlet_conditions_max,
								 const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
								 index_t x_size, index_t y_size, index_t z_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void solve_slice_3d_z(real_t* __restrict__ densities, const real_t* __restrict__ b,
								 const real_t* __restrict__ c, const bool* __restrict__ dirichlet_conditions_min,
								 const real_t* __restrict__ dirichlet_values_min,
								 const bool* __restrict__ dirichlet_conditions_max,
								 const real_t* __restrict__ dirichlet_values_max, index_t substrates_count,
								 index_t x_size, index_t y_size, index_t z_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

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

void run_solve_slice_2d_x(real_t* densities, const real_t* b, const real_t* c, const real_t* a, const real_t* r_fwd,
						  const real_t* c_fwd, const real_t* a_bck, const real_t* c_bck, const real_t* c_rdc,
						  const real_t* r_rdc, const bool* dirichlet_conditions_min, const real_t* dirichlet_values_min,
						  const bool* dirichlet_conditions_max, const real_t* dirichlet_values_max,
						  index_t substrates_count, index_t x_size, index_t y_size, index_t work_block_size,
						  cudaStream_t& stream, index_t shared_mem_limit)
{
	size_t b_size = x_size * substrates_count * sizeof(real_t);

	int slice_remainder = (x_size * substrates_count * sizeof(real_t)) % 128;
	int s_remainder = (substrates_count * sizeof(real_t)) % 128;

	int padding;
	if (slice_remainder <= s_remainder)
	{
		padding = s_remainder - slice_remainder;
	}
	else
	{
		padding = 128 - slice_remainder + s_remainder;
	}

	index_t slice_size = x_size * substrates_count * sizeof(real_t) + padding;

	shared_mem_limit -= b_size;
	int max_slices = shared_mem_limit / slice_size;
	int slices_in_warp = (32 + substrates_count - 1) / substrates_count;
	int max_warps = max_slices / slices_in_warp;

	int work_items = y_size * substrates_count;

	if (max_warps <= 0)
	{
		int block_size = 256;
		int blocks = (work_items + 32 - 1) / 32;

		solve_slice_2d_x_block<<<blocks, block_size, 0, stream>>>(
			densities, a, r_fwd, c_fwd, a_bck, c_bck, c_rdc, r_rdc, dirichlet_conditions_min, dirichlet_values_min,
			dirichlet_conditions_max, dirichlet_values_max, substrates_count, x_size, y_size, work_block_size);
	}
	else
	{
		int block_size = std::min(512, max_warps * 32);
		int blocks = (work_items + block_size - 1) / block_size;
		int shmem = ((block_size + substrates_count - 1) / substrates_count) * slice_size + b_size;

		solve_slice_2d_x_shared_full<<<blocks, block_size, shmem, stream>>>(
			densities, b, c, dirichlet_conditions_min, dirichlet_values_min, dirichlet_conditions_max,
			dirichlet_values_max, substrates_count, x_size, y_size, padding / sizeof(real_t));
	}
}

void run_solve_slice_2d_y(real_t* densities, const real_t* b, const real_t* c, const real_t* a, const real_t* r_fwd,
						  const real_t* c_fwd, const real_t* a_bck, const real_t* c_bck, const real_t* c_rdc,
						  const real_t* r_rdc, const bool* dirichlet_conditions_min, const real_t* dirichlet_values_min,
						  const bool* dirichlet_conditions_max, const real_t* dirichlet_values_max,
						  index_t substrates_count, index_t x_size, index_t y_size, index_t work_block_size,
						  cudaStream_t& stream, index_t shared_mem_limit)
{
	index_t b_size = y_size * substrates_count * sizeof(real_t);
	index_t slice_size = y_size * sizeof(real_t);

	shared_mem_limit -= b_size;
	int max_slices = shared_mem_limit / slice_size;
	int max_warps = max_slices / 32;

	int work_items = x_size * substrates_count;

	if (max_warps <= 0)
	{
		int block_size = 256;
		int blocks = (work_items + 32 - 1) / 32;

		solve_slice_2d_y_block<<<blocks, block_size, 0, stream>>>(
			densities, a, r_fwd, c_fwd, a_bck, c_bck, c_rdc, r_rdc, dirichlet_conditions_min, dirichlet_values_min,
			dirichlet_conditions_max, dirichlet_values_max, substrates_count, x_size, y_size, work_block_size);
	}
	else
	{
		int block_size = std::min(512, max_warps * 32);
		int blocks = (work_items + block_size - 1) / block_size;
		int shmem = block_size * slice_size + b_size;

		solve_slice_2d_y_shared_full<<<blocks, block_size, shmem, stream>>>(
			densities, b, c, dirichlet_conditions_min, dirichlet_values_min, dirichlet_conditions_max,
			dirichlet_values_max, substrates_count, x_size, y_size);
	}
}

void run_solve_slice_3d_x(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  index_t z_size, cudaStream_t& stream)
{
	int work_items = z_size * y_size * substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	solve_slice_3d_x<<<blocks, block_size, 0, stream>>>(densities, b, c, dirichlet_conditions_min, dirichlet_values_min,
														dirichlet_conditions_max, dirichlet_values_max,
														substrates_count, x_size, y_size, z_size);
}

void run_solve_slice_3d_y(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  index_t z_size, cudaStream_t& stream)
{
	int work_items = z_size * x_size * substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	solve_slice_3d_y<<<blocks, block_size, 0, stream>>>(densities, b, c, dirichlet_conditions_min, dirichlet_values_min,
														dirichlet_conditions_max, dirichlet_values_max,
														substrates_count, x_size, y_size, z_size);
}

void run_solve_slice_3d_z(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  index_t z_size, cudaStream_t& stream)
{
	int work_items = y_size * x_size * substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	solve_slice_3d_z<<<blocks, block_size, 0, stream>>>(densities, b, c, dirichlet_conditions_min, dirichlet_values_min,
														dirichlet_conditions_max, dirichlet_values_max,
														substrates_count, x_size, y_size, z_size);
}
