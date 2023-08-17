#include "device_context.h"

#include <iostream>

using namespace biofvm::solvers::device;

device_context::device_context() : context(CL_DEVICE_TYPE_DEFAULT), queue(context)
{
	std::vector<cl::Device> devices;

	std::vector<cl::Platform> platforms; // get all platforms
	std::vector<cl::Device> devices_available;
	int n = 0; // number of available devices
	cl::Platform::get(&platforms);
	for (int i = 0; i < (int)platforms.size(); i++)
	{
		devices_available.clear();
		platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
		if (devices_available.size() == 0)
			continue; // no device found in plattform i
		for (int j = 0; j < (int)devices_available.size(); j++)
		{
			n++;
			devices.push_back(devices_available[j]);
		}
	}
	if (platforms.size() == 0 || devices.size() == 0)
	{
		std::cout << "Error: There are no OpenCL devices available!" << std::endl;
	}
	for (int i = 0; i < n; i++)
		std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
}

void device_context::initialize(microenvironment& m)
{
	diffusion_substrates =
		cl::Buffer(context, CL_MEM_READ_WRITE, m.mesh.voxel_count() * m.substrates_count * sizeof(real_t));
}

void device_context::copy_to_device(microenvironment& m)
{
	cl::copy(queue, m.substrate_densities.get(),
			 m.substrate_densities.get() + m.mesh.voxel_count() * m.substrates_count, diffusion_substrates);
}

void device_context::copy_to_host(microenvironment& m)
{
	cl::copy(queue, diffusion_substrates, m.substrate_densities.get(),
			 m.substrate_densities.get() + m.mesh.voxel_count() * m.substrates_count);
}
