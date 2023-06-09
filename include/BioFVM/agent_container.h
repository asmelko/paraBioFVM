#pragma once

#include <memory>

#include "agent.h"
#include "agent_data.h"

namespace biofvm {

class cell_solver;

using agent_ptr = std::unique_ptr<agent>;

class agent_container;

using agent_container_ptr = std::unique_ptr<agent_container>;

class agent_container
{
	friend cell_solver;

protected:
	agent_data data_;

	agent_id_t next_agent_id_;

	std::vector<agent_ptr> agents_;

	index_t find_agent_index(const agent_ptr& agent) const;

public:
	agent_container(microenvironment& m);

	agent* add_agent();

	void remove_agent(agent_id_t id);
	void remove_agent(agent_ptr& agent);

	const agent_data& data() const;

	const std::vector<agent_ptr>& agents() const;

	virtual ~agent_container() = default;
};

} // namespace biofvm
