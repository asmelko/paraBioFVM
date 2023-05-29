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
*/

class cell_solver
{
	static void solve(agent_data& data);
};
