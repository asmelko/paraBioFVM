#include <cstdio>

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

void run_solve_slice_2d_x(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  cudaStream_t& stream)
{
	int work_items = y_size * substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	solve_slice_2d_x<<<blocks, block_size, 0, stream>>>(densities, b, c, dirichlet_conditions_min, dirichlet_values_min,
														dirichlet_conditions_max, dirichlet_values_max,
														substrates_count, x_size, y_size);
}

void run_solve_slice_2d_y(real_t* densities, const real_t* b, const real_t* c, const bool* dirichlet_conditions_min,
						  const real_t* dirichlet_values_min, const bool* dirichlet_conditions_max,
						  const real_t* dirichlet_values_max, index_t substrates_count, index_t x_size, index_t y_size,
						  cudaStream_t& stream)
{
	int work_items = x_size * substrates_count;
	int block_size = 256;
	int blocks = (work_items + block_size - 1) / block_size;

	solve_slice_2d_y<<<blocks, block_size, 0, stream>>>(densities, b, c, dirichlet_conditions_min, dirichlet_values_min,
														dirichlet_conditions_max, dirichlet_values_max,
														substrates_count, x_size, y_size);
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
