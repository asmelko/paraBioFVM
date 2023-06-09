#pragma once

#include "agent_container_base.h"

namespace biofvm {

class agent_container : public agent_container_templated<agent>
{
public:
	agent_container(microenvironment& m);
};

} // namespace biofvm
