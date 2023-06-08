#pragma once

#include <array>
#include <stdint.h>

namespace biofvm {

using agent_id_t = uint32_t;
using real_t = float;
using index_t = int32_t;

template <typename T, int dims>
using point_t = std::array<T, dims>;

} // namespace biofvm
