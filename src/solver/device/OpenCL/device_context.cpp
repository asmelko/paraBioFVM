#include "device_context.h"

#include <iostream>

#include <CL/cl.h>

using namespace biofvm::solvers::device;

device_context::device_context(bool print_device_info)
	: context(CL_DEVICE_TYPE_DEFAULT), substrates_queue(context), cell_data_queue(context), capacity(0)
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

	std::cout << "Selected device: " << context.getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_NAME>() << " with "
			  << context.getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << "B of local memory"
			  << std::endl;

	local_mem_limit = context.getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
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
	substrates_queue.enqueueWriteBuffer(diffusion_substrates, CL_FALSE, 0,
										m.mesh.voxel_count() * m.substrates_count * sizeof(real_t),
										m.substrate_densities.get());

	auto& data = get_agent_data(m);

	if (data.volumes.capacity() != capacity)
	{
		resize(data.volumes.capacity(), m);
	}

	if (capacity != 0)
	{
		cell_data_queue.enqueueWriteBuffer(secretion_rates, CL_FALSE, 0,
										   data.agents_count * m.substrates_count * sizeof(real_t),
										   data.secretion_rates.data());
		cell_data_queue.enqueueWriteBuffer(saturation_densities, CL_FALSE, 0,
										   data.agents_count * m.substrates_count * sizeof(real_t),
										   data.saturation_densities.data());
		cell_data_queue.enqueueWriteBuffer(uptake_rates, CL_FALSE, 0,
										   data.agents_count * m.substrates_count * sizeof(real_t),
										   data.uptake_rates.data());
		cell_data_queue.enqueueWriteBuffer(net_export_rates, CL_FALSE, 0,
										   data.agents_count * m.substrates_count * sizeof(real_t),
										   data.net_export_rates.data());

		cell_data_queue.enqueueWriteBuffer(internalized_substrates, CL_FALSE, 0,
										   data.agents_count * m.substrates_count * sizeof(real_t),
										   data.internalized_substrates.data());

		cell_data_queue.enqueueWriteBuffer(volumes, CL_FALSE, 0, data.agents_count * sizeof(real_t),
										   data.volumes.data());
		cell_data_queue.enqueueWriteBuffer(positions, CL_FALSE, 0, data.agents_count * m.mesh.dims * sizeof(real_t),
										   data.positions.data());
	}
}

void device_context::copy_to_host(microenvironment& m)
{
	substrates_queue.enqueueReadBuffer(diffusion_substrates, CL_FALSE, 0,
									   m.mesh.voxel_count() * m.substrates_count * sizeof(real_t),
									   m.substrate_densities.get());

	auto& data = get_agent_data(m);

	if (capacity != 0)
	{
		cell_data_queue.enqueueReadBuffer(internalized_substrates, CL_FALSE, 0,
										  data.agents_count * m.substrates_count * sizeof(real_t),
										  data.internalized_substrates.data());
	}

	substrates_queue.finish();
	cell_data_queue.finish();
}
