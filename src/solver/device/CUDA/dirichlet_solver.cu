#include <device_launch_parameters.h>

#include "types.h"

using namespace biofvm;

__global__ void solve_interior_2d(real_t* __restrict__ substrate_densities,
								  const index_t* __restrict__ dirichlet_voxels,
								  const real_t* __restrict__ dirichlet_values,
								  const bool* __restrict__ dirichlet_conditions, index_t n, index_t substrates_count,
								  index_t x_size, index_t y_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
		return;

	index_t i = id / substrates_count;
	index_t s = id % substrates_count;

	index_t x = dirichlet_voxels[i * 2];
	index_t y = dirichlet_voxels[i * 2 + 1];

	if (dirichlet_conditions[i * substrates_count + s])
		substrate_densities[(y * x_size + x) * substrates_count + s] = dirichlet_values[i * substrates_count + s];
}

__global__ void solve_interior_3d(real_t* __restrict__ substrate_densities,
								  const index_t* __restrict__ dirichlet_voxels,
								  const real_t* __restrict__ dirichlet_values,
								  const bool* __restrict__ dirichlet_conditions, index_t n, index_t substrates_count,
								  index_t x_size, index_t y_size, index_t z_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= n * substrates_count)
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

void run_solve_interior_2d(real_t* substrate_densities, const index_t* dirichlet_voxels, const real_t* dirichlet_values,
						   const bool* dirichlet_conditions, index_t n, index_t substrates_count, index_t x_size,
						   index_t y_size, cudaStream_t& stream)
{
	int block_size = 256;
	solve_interior_2d<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
		substrate_densities, dirichlet_voxels, dirichlet_values, dirichlet_conditions, n, substrates_count, x_size,
		y_size);
}

void run_solve_interior_3d(real_t* substrate_densities, const index_t* dirichlet_voxels, const real_t* dirichlet_values,
						   const bool* dirichlet_conditions, index_t n, index_t substrates_count, index_t x_size,
						   index_t y_size, index_t z_size, cudaStream_t& stream)
{
	int block_size = 256;
	solve_interior_3d<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
		substrate_densities, dirichlet_voxels, dirichlet_values, dirichlet_conditions, n, substrates_count, x_size,
		y_size, z_size);
}
