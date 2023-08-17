#include "opencl_solver.h"

#include <iostream>

using namespace biofvm::solvers::device;

opencl_solver::opencl_solver(device_context& ctx, const std::string& file_name)
	: ctx_(ctx),
	  kernel_fs_(file_name),
	  kernel_code_(std::istreambuf_iterator<char>(kernel_fs_), (std::istreambuf_iterator<char>())),
	  program_(ctx_.context, kernel_code_, true)
{}
