#pragma once

#include "microenvironment.h"

namespace biofvm {

class gradient_solver
{
public:
	static void initialize(microenvironment& m);

	static void solve(microenvironment& m);
};

} // namespace biofvm
