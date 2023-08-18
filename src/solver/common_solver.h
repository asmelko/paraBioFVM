#pragma once

namespace biofvm {

struct agent_data;
struct microenvironment;

namespace solvers {

class common_solver
{
protected:
	agent_data& get_agent_data(microenvironment& m);
};

} // namespace solvers
} // namespace biofvm
