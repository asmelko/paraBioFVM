#pragma once

#ifndef BioFVM_SOLVER_IMPL
	#define BioFVM_SOLVER_IMPL host
#endif


#if BioFVM_SOLVER_IMPL == host
	#include "../../src/solver/host/solver.h"

namespace biofvm {
using solver = solvers::host::solver;
} // namespace biofvm

#elif BioFVM_SOLVER_IMPL == device
	#include "../../src/solver/device/solver.h"

namespace biofvm {
using solver = solvers::device::solver;
} // namespace biofvm
#endif
