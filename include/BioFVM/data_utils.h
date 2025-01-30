#pragma once

#include <cstring>

#include "types.h"

template <typename T>
void move_scalar(T* dst, const T* src)
{
	dst[0] = src[0];
}

template <typename T>
void move_vector(T* dst, const T* src, biofvm::index_t size)
{
	std::memcpy(dst, src, size * sizeof(T));
}
