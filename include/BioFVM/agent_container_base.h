#pragma once

#include <memory>

#include "agent.h"
#include "agent_data.h"

namespace biofvm {

class cell_solver;

using agent_ptr = std::unique_ptr<agent>;

class agent_container_base;

using agent_container_ptr = std::unique_ptr<agent_container_base>;

class agent_container_base
{
	friend cell_solver;

protected:
	agent_data data_;

	agent_id_t next_agent_id_;

	agent_container_base(microenvironment& m);

	index_t& get_agent_index(agent* agent);

public:
	virtual agent* add_agent() = 0;

	virtual void remove_agent(agent* agent) = 0;

	const agent_data& data() const { return data_; }

	virtual ~agent_container_base() = default;
};

template <typename agent_t>
class agent_container_templated : public agent_container_base
{
protected:
	std::vector<std::unique_ptr<agent_t>> agents_;

	agent_container_templated(microenvironment& m) : agent_container_base(m) {}

public:
	virtual agent* add_agent() override
	{
		agents_.emplace_back(std::make_unique<agent_t>(next_agent_id_++, data_, data_.agents_count));

		data_.add();

		return agents_.back().get();
	}

	virtual void remove_agent(agent* agent) override
	{
		index_t index_to_remove = get_agent_index(agent);

		get_agent_index(agents_.back().get()) = index_to_remove;

		std::swap(agents_[index_to_remove], agents_.back());

		agents_.resize(agents_.size() - 1);

		data_.remove(index_to_remove);
	}

	const std::vector<agent_t>& agents() const { return agents_; }
};

} // namespace biofvm
