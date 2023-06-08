#pragma once

#include <memory>

#include "agent.h"
#include "agent_data.h"

namespace biofvm {

class cell_solver;

using agent_ptr = std::unique_ptr<agent>;

class agent_container
{
	friend cell_solver;

	agent_data data_;

	agent_id_t next_agent_id_;

	std::vector<agent_ptr> agents_;

public:
	agent_container(microenvironment& m);

	agent* add_agent();

	void remove_agent(agent_id_t id);
	void remove_agent(agent_ptr& agent);

	const agent_data& data() const;
};

} // namespace biofvm
