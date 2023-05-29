#pragma once

#include <array>
#include <stdint.h>

using real_t = float;
using index_t = int32_t;

template <typename T, int dims>
using point_t = std::array<T, dims>;
