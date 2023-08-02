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
	virtual agent_data& get_agent_data() = 0;

	agent_id_t next_agent_id_;

	agent_container_base();

	index_t& get_agent_index(agent* agent);

public:
	virtual index_t agents_count() const = 0;

	virtual agent* create_agent() = 0;

	virtual void remove_agent(agent* agent) = 0;
	virtual void remove_at(index_t index) = 0;

	virtual agent* get_agent_at(index_t index) = 0;

	virtual ~agent_container_base() = default;
};

template <typename agent_t, typename agent_data_t>
class agent_container_common : public agent_container_base
{
protected:
	agent_data_t data_;

	std::vector<std::unique_ptr<agent_t>> agents_;

	template <typename... args_t>
	agent_container_common(args_t&&... args) : data_(std::forward<args_t>(args)...)
	{}

public:
	virtual index_t agents_count() const override { return (index_t)agents_.size(); }

	virtual agent* create_agent() override { return create(); }

	agent_t* create()
	{
		data_.add();

		agents_.emplace_back(std::make_unique<agent_t>(next_agent_id_++, data_, data_.agents_count - 1));

		return agents_.back().get();
	}

	virtual void remove_agent(agent* agent) override
	{
		index_t index_to_remove = get_agent_index(agent);

		remove_at(index_to_remove);
	}

	virtual void remove_at(index_t index) override
	{
		get_agent_index(agents_.back().get()) = index;

		std::swap(agents_[index], agents_.back());

		agents_.resize(agents_.size() - 1);

		data_.remove(index);
	}

	virtual agent* get_agent_at(index_t index) override { return get_at(index); }

	agent_t* get_at(index_t index) { return agents_[index].get(); }

	const std::vector<std::unique_ptr<agent_t>>& agents() const { return agents_; }

	const agent_data_t& data() const { return data_; }
};

} // namespace biofvm
