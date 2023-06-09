#pragma once

#include "agent_container_base.h"

namespace biofvm {

class agent_container : public agent_container_common<agent, agent_data>
{
	virtual agent_data& get_agent_data() override;

public:
	agent_container(microenvironment& m);
};

} // namespace biofvm
