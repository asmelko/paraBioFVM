#pragma once

#include "agent_container.h"
#include "agent_data.h"
#include "types.h"

class agent_container;

class agent
{
	friend agent_container;

	index_t index_;
	agent_data& data_;

public:
	const agent_id_t id;

	agent(agent_id_t id, agent_data& data, index_t index);

	real_t* secretion_rates();
	real_t* saturation_densities();
	real_t* uptake_rates();
	real_t* net_export_rates();

	real_t* internalized_substrates();
	real_t* fraction_released_at_death();
	real_t* fraction_transferred_when_ingested();

	real_t& volume();

	point_t<real_t, 3> position() const;

	point_t<index_t, 3> voxel_index() const;

	real_t* nearest_density_vector();
	point_t<real_t, 3> nearest_gradient(index_t substrate_index) const;
};
