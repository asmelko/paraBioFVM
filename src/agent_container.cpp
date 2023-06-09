#include "agent_container.h"

#include <algorithm>

#include "agent.h"
#include "agent_container.h"

using namespace biofvm;

agent_data& agent_container::get_agent_data() { return data_; };

agent_container::agent_container(microenvironment& m) : agent_container_templated<agent, agent_data>(m) {}
