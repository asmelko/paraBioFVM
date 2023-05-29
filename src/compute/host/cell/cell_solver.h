#pragma once

#include "../../../agent_data.h"

/*
Performs secretion and uptake of cells.
Updates substrate denisities of the cell's voxel and conditionally updates the cell's internalized substrates.
For each cell, the following is performed:

D = substrate densities
I = internalized substrates
S = secretion rates
U = uptake rates
T = saturation densities
N = net export rates
c = cell_volume
v = voxel_volume

D = (D + (c/v)*dt*S*T)/(1 + (c/v)*dt*(U+S)) + (1/v)*dt*N
I = I - ((-c*dt*(U+S)*D + c*dt*S*T)/(1 + c*dt*(U+S)) + dt*N)

Also handles release of internalized substrates:

F = fraction released at death

D = D + I*F/v
*/

class cell_solver
{
	static void simulate_secretion_and_uptake(agent_data& data);

	static void release_internalized_substrates(agent_data& data, index_t index);
};
