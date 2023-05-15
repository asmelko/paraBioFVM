#include "diffusion_solvers/host_solver.h"

int main()
{
	microenvironment m;

	diffusion_solver s;

	s.solve(m);

	s.initialize(m);
}