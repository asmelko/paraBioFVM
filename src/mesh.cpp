#include "mesh.h"

cartesian_mesh::cartesian_mesh(index_t dims, point_t<index_t, 3> bounding_box_mins,
							   point_t<index_t, 3> bounding_box_maxs, point_t<index_t, 3> voxel_shape)
	: dims(dims), bounding_box_mins(bounding_box_mins), bounding_box_maxs(bounding_box_maxs), voxel_shape(voxel_shape)
{
	grid_shape = { 1, 1, 1 };

	if (dims >= 1)
	{
		grid_shape[0] = (bounding_box_maxs[0] - bounding_box_mins[0]) / voxel_shape[0];
	}
	if (dims >= 2)
	{
		grid_shape[1] = (bounding_box_maxs[1] - bounding_box_mins[1]) / voxel_shape[1];
	}
	if (dims >= 3)
	{
		grid_shape[2] = (bounding_box_maxs[2] - bounding_box_mins[2]) / voxel_shape[2];
	}
}

index_t cartesian_mesh::voxel_count() const { return grid_shape[0] * grid_shape[1] * grid_shape[2]; }
