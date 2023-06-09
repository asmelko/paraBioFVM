#include "agent_data.h"

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

	for (index_t s = 0; s < m.substrates_count; s++)
	{
		secretion_rates[index * m.substrates_count + s] = secretion_rates[agents_count * m.substrates_count + s];
		saturation_densities[index * m.substrates_count + s] =
			saturation_densities[agents_count * m.substrates_count + s];
		uptake_rates[index * m.substrates_count + s] = uptake_rates[agents_count * m.substrates_count + s];
		net_export_rates[index * m.substrates_count + s] = net_export_rates[agents_count * m.substrates_count + s];

		internalized_substrates[index * m.substrates_count + s] =
			internalized_substrates[agents_count * m.substrates_count + s];
		fraction_released_at_death[index * m.substrates_count + s] =
			fraction_released_at_death[agents_count * m.substrates_count + s];
		fraction_transferred_when_ingested[index * m.substrates_count + s] =
			fraction_transferred_when_ingested[agents_count * m.substrates_count + s];
	}

	volumes[index] = volumes[agents_count];

	for (index_t i = 0; i < m.mesh.dims; i++)
	{
		positions[index * m.mesh.dims + i] = positions[agents_count * m.mesh.dims + i];
	}
}
