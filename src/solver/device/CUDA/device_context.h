#pragma once

#include <cuda_runtime.h>

#include "../../../../include/BioFVM/microenvironment.h"
#include "../../common_solver.h"

namespace biofvm {
namespace solvers {
namespace device {

inline void cuda_check(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
		abort();
	}
}

#define CUCH(err) cuda_check(err, __FILE__, __LINE__)

struct device_context : common_solver
{
private:
	void resize(std::size_t new_capacity, microenvironment& m);

public:
	cudaStream_t substrates_queue;
	cudaStream_t cell_data_queue;

	real_t* diffusion_substrates;

	// required agent data:
	std::size_t capacity;

	real_t* secretion_rates;
	real_t* saturation_densities;
	real_t* uptake_rates;
	real_t* net_export_rates;

	real_t* internalized_substrates;

	real_t* volumes;
	real_t* positions;

	device_context();
	~device_context();

	void initialize(microenvironment& m);

	void copy_to_device(microenvironment& m);
	void copy_to_host(microenvironment& m);
};

} // namespace device
} // namespace solvers
} // namespace biofvm
