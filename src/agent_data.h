#pragma once

#include <vector>

#include "types.h"

using agent_id_t = uint32_t;

class microenvironment;

struct agent_data
{
public:
	std::vector<real_t> secretion_rates;
	std::vector<real_t> saturation_densities;
	std::vector<real_t> uptake_rates;
	std::vector<real_t> net_export_rates;

	std::vector<real_t> internalized_substrates;
	std::vector<real_t> fraction_released_at_death;
	std::vector<real_t> fraction_transferred_when_ingested;

    std::vector<real_t> volumes;
    std::vector<real_t> positions;

	index_t agents_count;

    microenvironment& m;
	
	agent_data(microenvironment& m);

	void add();
	void remove(index_t index);
};
