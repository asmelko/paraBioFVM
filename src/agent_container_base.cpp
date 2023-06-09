#include "agent_container_base.h"

using namespace biofvm;

agent_container_base::agent_container_base() : next_agent_id_(0) {}

index_t& agent_container_base::get_agent_index(agent* agent) { return agent->index_; }
