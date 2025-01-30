#include "agent_data.h"

#include "data_utils.h"
#include "microenvironment.h"

using namespace biofvm;

agent_data::agent_data(microenvironment& m) : agents_count(0), m(m) {}

void agent_data::add()
{
	agents_count++;

	secretion_rates.resize(agents_count * m.substrates_count);
	saturation_densities.resize(agents_count * m.substrates_count);
	uptake_rates.resize(agents_count * m.substrates_count);
	net_export_rates.resize(agents_count * m.substrates_count);

	internalized_substrates.resize(agents_count * m.substrates_count, 0);
	fraction_released_at_death.resize(agents_count * m.substrates_count);
	fraction_transferred_when_ingested.resize(agents_count * m.substrates_count);

	volumes.resize(agents_count, 0);
	positions.resize(agents_count * m.mesh.dims, 0);
}

void agent_data::remove(index_t index)
{
	agents_count--;

	if (index == agents_count)
		return;

	move_vector(secretion_rates.data() + index * m.substrates_count,
				secretion_rates.data() + agents_count * m.substrates_count, m.substrates_count);
	move_vector(saturation_densities.data() + index * m.substrates_count,
				saturation_densities.data() + agents_count * m.substrates_count, m.substrates_count);
	move_vector(uptake_rates.data() + index * m.substrates_count,
				uptake_rates.data() + agents_count * m.substrates_count, m.substrates_count);
	move_vector(net_export_rates.data() + index * m.substrates_count,
				net_export_rates.data() + agents_count * m.substrates_count, m.substrates_count);

	move_vector(internalized_substrates.data() + index * m.substrates_count,
				internalized_substrates.data() + agents_count * m.substrates_count, m.substrates_count);
	move_vector(fraction_released_at_death.data() + index * m.substrates_count,
				fraction_released_at_death.data() + agents_count * m.substrates_count, m.substrates_count);
	move_vector(fraction_transferred_when_ingested.data() + index * m.substrates_count,
				fraction_transferred_when_ingested.data() + agents_count * m.substrates_count, m.substrates_count);

	volumes[index] = volumes[agents_count];

	move_vector(positions.data() + index * m.mesh.dims, positions.data() + agents_count * m.mesh.dims, m.mesh.dims);
}
