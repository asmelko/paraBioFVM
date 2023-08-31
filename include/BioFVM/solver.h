#pragma once

#if BioFVM_SOLVER_IMPL_HOST
	#include "../../src/solver/host/solver.h"

namespace biofvm {
using solver = solvers::host::solver;
using cell_solver = solvers::host::cell_solver;
} // namespace biofvm

#endif

#if BioFVM_SOLVER_IMPL_OPENCL
	#include "../../src/solver/device/OpenCL/solver.h"

namespace biofvm {
using solver = solvers::device::solver;
using cell_solver = solvers::device::cell_solver;
} // namespace biofvm

#endif

#if BioFVM_SOLVER_IMPL_CUDA
	#include "../../src/solver/device/CUDA/solver.h"

namespace biofvm {
using solver = solvers::device::solver;
using cell_solver = solvers::device::cell_solver;
} // namespace biofvm

#endif
