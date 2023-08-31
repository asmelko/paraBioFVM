#include "cuda_solver.h"

using namespace biofvm::solvers::device;

cuda_solver::cuda_solver(device_context& ctx) : ctx_(ctx) {}
