#pragma once

#include <array>
#include <stdint.h>

namespace biofvm {

using agent_id_t = uint32_t;
using index_t = int32_t;

#ifdef USE_DOUBLES
using real_t = double;
#else
using real_t = float;
#endif

template <typename T, int dims>
using point_t = std::array<T, dims>;

} // namespace biofvm
