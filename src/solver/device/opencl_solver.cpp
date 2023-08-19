#include "opencl_solver.h"

#include <iostream>
#include <type_traits>

using namespace biofvm::solvers::device;

bool is_nvidia(const cl::Context& c)
{
	return c.getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>().starts_with("NVIDIA");
}

opencl_solver::opencl_solver(device_context& ctx, const std::string& file_name)
	: ctx_(ctx),
	  kernel_fs_(file_name),
	  kernel_code_(std::istreambuf_iterator<char>(kernel_fs_), (std::istreambuf_iterator<char>())),
	  program_(ctx_.context, kernel_code_, false)
{
	try
	{
		std::string build_parameters = "-cl-std=CL2.0 -w";

		if (is_nvidia(ctx_.context))
			build_parameters += " -DNVIDIA";

		if (std::is_same_v<real_t, double>)
			build_parameters += " -DDOUBLE";

		program_.build(build_parameters.c_str());
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
