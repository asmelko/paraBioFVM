#include "common_solver.h"

#include "microenvironment.h"

using namespace biofvm;
using namespace solvers;

agent_data& common_solver::get_agent_data(microenvironment& m) { return m.agents->get_agent_data(); }
