#pragma once

#include "../../../include/BioFVM/microenvironment.h"

namespace biofvm {
namespace solvers {
namespace host {

class gradient_solver
{
public:
	static void solve(microenvironment& m);
};

} // namespace host
} // namespace solvers
} // namespace biofvm
