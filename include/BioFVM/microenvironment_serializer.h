#pragma once
#include "microenvironment.h"

namespace biofvm {

struct microenvironment_serializer
{
	virtual void serialize_one_timestep(const microenvironment& m) = 0;
};

} // namespace biofvm
