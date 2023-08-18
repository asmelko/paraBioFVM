#include "opencl_solver.h"

#include <iostream>

using namespace biofvm::solvers::device;

opencl_solver::opencl_solver(device_context& ctx, const std::string& file_name)
	: ctx_(ctx),
	  kernel_fs_(file_name),
	  kernel_code_(std::istreambuf_iterator<char>(kernel_fs_), (std::istreambuf_iterator<char>())),
	  program_(ctx_.context, kernel_code_, false)
{
	try
	{
		program_.build("-cl-std=CL3.0 -w");
	}
	catch (...)
	{
		std::cout << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx_.context.getInfo<CL_CONTEXT_DEVICES>()[0])
				  << std::endl;

		throw;
	}
	if (!program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx_.context.getInfo<CL_CONTEXT_DEVICES>()[0]).empty())
	{
		std::cout << "Build Log: "
				  << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx_.context.getInfo<CL_CONTEXT_DEVICES>()[0])
				  << std::endl;
	}
}
