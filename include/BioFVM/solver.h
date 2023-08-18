#pragma once

#define BioFVM_HOST_SOLVER 0
#define BioFVM_DEVICE_SOLVER 1

#if BioFVM_SOLVER_IMPL == BioFVM_HOST_SOLVER
	#include "../../src/solver/host/solver.h"

namespace biofvm {
using solver = solvers::host::solver;
} // namespace biofvm

#endif

#if BioFVM_SOLVER_IMPL == BioFVM_DEVICE_SOLVER
	#include "../../src/solver/device/solver.h"

namespace biofvm {
using solver = solvers::device::solver;
} // namespace biofvm

#endif