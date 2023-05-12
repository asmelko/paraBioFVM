#include "types.h"

template <int dims>
struct cartesian_mesh
{
	point_t<index_t, dims> bounding_box_mins; // [x_min, y_min, z_min]
	point_t<index_t, dims> bounding_box_maxs; // [x_max, y_max, z_max]

	point_t<index_t, dims> voxel_shape; // [dx, dy, dz]
    point_t<index_t, dims> voxel_dims; // [x_size, y_size, z_size]
};
