#include "microenvironment.h"

class diffusion_solver
{
public:
	template <int dims>
	void initialize(microenvironment<dims>&)
	{}

	template <int dims>
	void solve(microenvironment<dims>& m);
};
