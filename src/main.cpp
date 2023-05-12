#include "solver.h"

int main() { 
    microenvironment<1> m;

    diffusion_solver s;

    s.solve(m);

    s.initialize(m);

}