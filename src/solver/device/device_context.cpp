#include "device_context.h"

using namespace biofvm::solvers::device;

device_context::device_context() : context(CL_DEVICE_TYPE_DEFAULT), queue(context) {}

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
