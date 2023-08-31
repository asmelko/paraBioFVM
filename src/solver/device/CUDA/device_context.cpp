#include "device_context.h"

#include <iostream>

using namespace biofvm::solvers::device;

device_context::device_context() : capacity(0)
{
	CUCH(cudaStreamCreate(&substrates_queue));
	CUCH(cudaStreamCreate(&cell_data_queue));
}

device_context::~device_context()
{
	CUCH(cudaStreamDestroy(substrates_queue));
	CUCH(cudaStreamDestroy(cell_data_queue));
}

void device_context::initialize(microenvironment& m)
{
	cudaMalloc(&diffusion_substrates, m.mesh.voxel_count() * m.substrates_count * sizeof(real_t));

	capacity = 0;
}

void device_context::resize(std::size_t new_capacity, microenvironment& m)
{
	if (capacity != 0)
	{
		cudaFree(secretion_rates);
		cudaFree(saturation_densities);
		cudaFree(uptake_rates);
		cudaFree(net_export_rates);

		cudaFree(internalized_substrates);

		cudaFree(volumes);
		cudaFree(positions);
	}

	capacity = new_capacity;

	CUCH(cudaMalloc(&secretion_rates, capacity * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&saturation_densities, capacity * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&uptake_rates, capacity * m.substrates_count * sizeof(real_t)));
	CUCH(cudaMalloc(&net_export_rates, capacity * m.substrates_count * sizeof(real_t)));

	CUCH(cudaMalloc(&internalized_substrates, capacity * m.substrates_count * sizeof(real_t)));

	CUCH(cudaMalloc(&volumes, capacity * sizeof(real_t)));
	CUCH(cudaMalloc(&positions, capacity * m.mesh.dims * sizeof(real_t)));
}

void device_context::copy_to_device(microenvironment& m)
{
	CUCH(cudaMemcpyAsync(diffusion_substrates, m.substrate_densities.get(),
						 m.mesh.voxel_count() * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice,
						 substrates_queue));

	auto& data = get_agent_data(m);

	if (data.volumes.capacity() != capacity)
	{
		resize(data.volumes.capacity(), m);
	}

	if (capacity != 0)
	{
		CUCH(cudaMemcpyAsync(secretion_rates, data.secretion_rates.data(),
							 data.agents_count * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice,
							 cell_data_queue));

		CUCH(cudaMemcpyAsync(saturation_densities, data.saturation_densities.data(),
							 data.agents_count * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice,
							 cell_data_queue));

		CUCH(cudaMemcpyAsync(uptake_rates, data.uptake_rates.data(),
							 data.agents_count * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice,
							 cell_data_queue));

		CUCH(cudaMemcpyAsync(net_export_rates, data.net_export_rates.data(),
							 data.agents_count * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice,
							 cell_data_queue));

		CUCH(cudaMemcpyAsync(internalized_substrates, data.internalized_substrates.data(),
							 data.agents_count * m.substrates_count * sizeof(real_t), cudaMemcpyHostToDevice,
							 cell_data_queue));

		CUCH(cudaMemcpyAsync(volumes, data.volumes.data(), data.agents_count * sizeof(real_t), cudaMemcpyHostToDevice,
							 cell_data_queue));

		CUCH(cudaMemcpyAsync(positions, data.positions.data(), data.agents_count * m.mesh.dims * sizeof(real_t),
							 cudaMemcpyHostToDevice, cell_data_queue));
	}
}

void device_context::copy_to_host(microenvironment& m)
{
	CUCH(cudaMemcpyAsync(m.substrate_densities.get(), diffusion_substrates,
						 m.mesh.voxel_count() * m.substrates_count * sizeof(real_t), cudaMemcpyDeviceToHost,
						 substrates_queue));

	auto& data = get_agent_data(m);

	if (capacity != 0)
	{
		CUCH(cudaMemcpyAsync(data.internalized_substrates.data(), internalized_substrates,
							 data.agents_count * m.substrates_count * sizeof(real_t), cudaMemcpyDeviceToHost,
							 substrates_queue));
	}

	CUCH(cudaStreamSynchronize(substrates_queue));
	CUCH(cudaStreamSynchronize(cell_data_queue));
}
