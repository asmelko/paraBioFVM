#pragma once

#include "../../../microenvironment.h"

class gradient_solver
{
public:
	static void initialize(microenvironment& m);

	static void solve(microenvironment& m);
};
