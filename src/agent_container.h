#pragma once

#include <memory>

#include "agent_data.h"

class agent;

using agent_ptr = std::unique_ptr<agent>;

class agent_container
{
	agent_data data_;

	agent_id_t next_agent_id_;

	std::vector<agent_ptr> agents_;

public:
	agent_container(microenvironment& m);

	void add_agent();

	void remove_agent(agent_id_t id);
	void remove_agent(agent_ptr& agent);
};
