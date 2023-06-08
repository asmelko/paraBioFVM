#pragma once

#include "types.h"

namespace biofvm {

struct cartesian_mesh
{
	index_t dims; // 1 or 2 or 3

	point_t<index_t, 3> bounding_box_mins; // [x_min, y_min, z_min]
	point_t<index_t, 3> bounding_box_maxs; // [x_max, y_max, z_max]

	point_t<index_t, 3> voxel_shape; // [dx, dy, dz]
	point_t<index_t, 3> grid_shape;	 // [x_size, y_size, z_size]

	cartesian_mesh(index_t dims, point_t<index_t, 3> bounding_box_mins, point_t<index_t, 3> bounding_box_maxs,
				   point_t<index_t, 3> voxel_shape);

	std::size_t voxel_count() const;
	index_t voxel_volume() const;

	point_t<index_t, 3> voxel_position(point_t<real_t, 3> position) const;
};

} // namespace biofvm
