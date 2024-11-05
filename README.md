# paraBioFVM: the optimized parallel implementation of BioFVM

This repository contains parallel CPU implementation of these algorithms:
- substrates diffusion,
- agent secretion and uptake and
- diffusion gradient computation.

The code is written using C++20. CPU parallelism is programmed using **OpenMP**. The implementation also takes advantage of compiler-generated vectorization. The repository also contains GPU implementation.

## Usage

The project functionalities are captured into several classes. These are the most important ones:

### Microenvironment
All the data required for the proper run of the aforementioned algorithms are stored in BIoFVM `microenvironment` C++ structure. It stores the sizes of the mesh domain, and all the information about the substrates (initial conditions, diffusion coefficients, decay rates). Most importantly, it contains the array of diffusion substrates (required for the diffusion and gradient) as well as the agents data (required for the secretion and uptake), which are stored in separate `agent_data` structure.

#### Microenvironment builder
For convenience, the microenvironment can be constructed via `microenvironment_builder`, which contains all the helping functions such as *add substrate*, *resize mesh* and others.

### Agent container

`microenvironment` structure holds a pointer to an `agent_container_base` class. It is a virtual class with an interface for the creation and fetching of bare `agent` objects. The more specialized class is `agent_container_common` template class, which implements the interface for a general templated `agent_t` (see `agent_container` as a specialization example).

The main task of the agent container and agent classes is to provide a nice object-oriented wrapper around the `agent_data` structure, which contains the data for all the created agents in SoA (Structure of Arrays) data format.

### Solver

The most important functions from the performance perspective are captured as separate solver classes. The responsibilities are as follows:
- `bulk_solver`: for bulk secretion/uptake (not used)
- `cell_solver`: for agent secretion/uptake
- `diffusion_solver`: for substrate diffusion
- `dirichlet_solver`: for Dirichlet boundary/interior conditions
- `gradient_solver`: for diffusion gradient

The specialized solvers are wrapped in the `solver` class for convenience.

---
The classes with their functionalities are provided in the form of header files (in `include` folder) and a static library `BioFVMCore`, which is exposed as a CMake target.

## Build and requirements
We use CMake as the build system. The buildable targets are the static library `BioFVMCore`, unit tests executable `unit_BioFVM` and `BioFVM` executable, which runs the solvers and outputs a simple benchmark.

The project uses git submodules, so these need to be initialized before building any target:

```sh
git submodule update --init --recursive
```

OpenMP is required only for parallel CPU run. If OpenMP can not be found in the system, the solvers will run a vectorized serial code.

The C++ compiler must be C++20 compliant.

The implementation uses single-precision floating point arithmetics. To enable double precision, just pass `-DDOUBLE_PRECISION` when doing CMake configuration step.

The following steps can be used to build the project in the `build` directory using CMake:

```sh
# add -DDOUBLE_PRECISION for double precision
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Experimental GPU support
When CMake option `BUILD_FOR_DEVICE` is set to `ON`, a GPU/accelerator solver implementation will be used. The GPU code is written using OpenCL v2.0. No special requirement is needed apart from the OpenCL capable device (the OpenCL headers and ICD-Loader will be fetched and compiled by CMake automatically).

Only diffusion and secretion are currently supported. Both are implemented very naively.
