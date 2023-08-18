#include "device_context.h"

#include <iostream>

#include <CL/cl.h>

using namespace biofvm::solvers::device;

device_context::device_context(bool print_device_info) : context(CL_DEVICE_TYPE_DEFAULT), queue(context), capacity(0)
{
	if (!print_device_info)
		return;

	std::vector<cl::Device> devices;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (std::size_t i = 0; i < platforms.size(); i++)
	{
		std::vector<cl::Device> devices_available;
		platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);

		if (devices_available.size() == 0)
			continue;

		for (std::size_t j = 0; j < devices_available.size(); j++)
			devices.push_back(devices_available[j]);
	}

	if (devices.size() == 0)
		throw std::runtime_error("Error: There are no OpenCL devices available!");

	for (std::size_t i = 0; i < devices.size(); i++)
		std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;

	std::cout << "Selected device: " << context.getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_NAME>() << std::endl;
}

void device_context::initialize(microenvironment& m)
{
	diffusion_substrates =
		cl::Buffer(context, CL_MEM_READ_WRITE, m.mesh.voxel_count() * m.substrates_count * sizeof(real_t));

	capacity = 0;
}

void device_context::resize(std::size_t new_capacity, microenvironment& m)
{
	capacity = new_capacity;
	secretion_rates = cl::Buffer(context, CL_MEM_READ_ONLY, capacity * m.substrates_count * sizeof(real_t));
	saturation_densities = cl::Buffer(context, CL_MEM_READ_ONLY, capacity * m.substrates_count * sizeof(real_t));
	uptake_rates = cl::Buffer(context, CL_MEM_READ_ONLY, capacity * m.substrates_count * sizeof(real_t));
	net_export_rates = cl::Buffer(context, CL_MEM_READ_ONLY, capacity * m.substrates_count * sizeof(real_t));

	internalized_substrates = cl::Buffer(context, CL_MEM_READ_WRITE, capacity * m.substrates_count * sizeof(real_t));

	volumes = cl::Buffer(context, CL_MEM_READ_ONLY, capacity * sizeof(real_t));
	positions = cl::Buffer(context, CL_MEM_READ_ONLY, capacity * m.mesh.dims * sizeof(real_t));
}

void device_context::copy_to_device(microenvironment& m)
{
	cl::copy(queue, m.substrate_densities.get(),
			 m.substrate_densities.get() + m.mesh.voxel_count() * m.substrates_count, diffusion_substrates);

	auto& data = get_agent_data(m);

	if (data.volumes.capacity() != capacity)
	{
		resize(data.volumes.capacity(), m);
	}

	if (capacity != 0)
	{
		cl::copy(queue, data.secretion_rates.begin(),
				 data.secretion_rates.begin() + data.agents_count * m.substrates_count, secretion_rates);
		cl::copy(queue, data.saturation_densities.begin(),
				 data.saturation_densities.begin() + data.agents_count * m.substrates_count, saturation_densities);
		cl::copy(queue, data.uptake_rates.begin(), data.uptake_rates.begin() + data.agents_count * m.substrates_count,
				 uptake_rates);
		cl::copy(queue, data.net_export_rates.begin(),
				 data.net_export_rates.begin() + data.agents_count * m.substrates_count, net_export_rates);

		cl::copy(queue, data.internalized_substrates.begin(),
				 data.internalized_substrates.begin() + data.agents_count * m.substrates_count,
				 internalized_substrates);

		cl::copy(queue, data.volumes.begin(), data.volumes.begin() + data.agents_count, volumes);
		cl::copy(queue, data.positions.begin(), data.positions.begin() + data.agents_count * m.mesh.dims, positions);
	}
}

void device_context::copy_to_host(microenvironment& m)
{
	cl::copy(queue, diffusion_substrates, m.substrate_densities.get(),
			 m.substrate_densities.get() + m.mesh.voxel_count() * m.substrates_count);

	auto& data = get_agent_data(m);

	if (capacity != 0)
	{
		cl::copy(queue, secretion_rates, data.secretion_rates.begin(),
				 data.secretion_rates.begin() + data.agents_count * m.substrates_count);
		cl::copy(queue, saturation_densities, data.saturation_densities.begin(),
				 data.saturation_densities.begin() + data.agents_count * m.substrates_count);
		cl::copy(queue, uptake_rates, data.uptake_rates.begin(),
				 data.uptake_rates.begin() + data.agents_count * m.substrates_count);
		cl::copy(queue, net_export_rates, data.net_export_rates.begin(),
				 data.net_export_rates.begin() + data.agents_count * m.substrates_count);

		cl::copy(queue, internalized_substrates, data.internalized_substrates.begin(),
				 data.internalized_substrates.begin() + data.agents_count * m.substrates_count);

		cl::copy(queue, volumes, data.volumes.begin(), data.volumes.begin() + data.agents_count);
		cl::copy(queue, positions, data.positions.begin(), data.positions.begin() + data.agents_count * m.mesh.dims);
	}
}
