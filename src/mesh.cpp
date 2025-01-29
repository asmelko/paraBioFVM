#include "mesh.h"

#include <iostream>

using namespace biofvm;

cartesian_mesh::cartesian_mesh(index_t dims, point_t<index_t, 3> bounding_box_mins,
							   point_t<index_t, 3> bounding_box_maxs, point_t<index_t, 3> voxel_shape)
	: dims(dims), bounding_box_mins(bounding_box_mins), bounding_box_maxs(bounding_box_maxs), voxel_shape(voxel_shape)
{
	grid_shape = { 1, 1, 1 };

	if (dims >= 1)
	{
		grid_shape[0] = (bounding_box_maxs[0] - bounding_box_mins[0] + voxel_shape[0] - 1) / voxel_shape[0];
	}
	if (dims >= 2)
	{
		grid_shape[1] = (bounding_box_maxs[1] - bounding_box_mins[1] + voxel_shape[1] - 1) / voxel_shape[1];
	}
	if (dims >= 3)
	{
		grid_shape[2] = (bounding_box_maxs[2] - bounding_box_mins[2] + voxel_shape[2] - 1) / voxel_shape[2];
	}
}

std::size_t cartesian_mesh::voxel_count() const
{
	return (std::size_t)grid_shape[0] * (std::size_t)grid_shape[1] * (std::size_t)grid_shape[2];
}

index_t cartesian_mesh::voxel_volume() const { return voxel_shape[0] * voxel_shape[1] * voxel_shape[2]; }

template <>
point_t<index_t, 3> cartesian_mesh::voxel_position<1>(const real_t* position) const
{
	return { (index_t)((position[0] - bounding_box_mins[0]) / voxel_shape[0]) };
}

template <>
point_t<index_t, 3> cartesian_mesh::voxel_position<2>(const real_t* position) const
{
	return { (index_t)((position[0] - bounding_box_mins[0]) / voxel_shape[0]),
			 (index_t)((position[1] - bounding_box_mins[1]) / voxel_shape[1]) };
}

template <>
point_t<index_t, 3> cartesian_mesh::voxel_position<3>(const real_t* position) const
{
	return { (index_t)((position[0] - bounding_box_mins[0]) / voxel_shape[0]),
			 (index_t)((position[1] - bounding_box_mins[1]) / voxel_shape[1]),
			 (index_t)((position[2] - bounding_box_mins[2]) / voxel_shape[2]) };
}

point_t<real_t, 3> cartesian_mesh::voxel_center(point_t<index_t, 3> position) const
{
	return { (real_t)(position[0] * voxel_shape[0] + voxel_shape[0] / 2.0 + bounding_box_mins[0]),
			 (real_t)(position[1] * voxel_shape[1] + voxel_shape[1] / 2.0 + bounding_box_mins[1]),
			 (real_t)(position[2] * voxel_shape[2] + voxel_shape[2] / 2.0 + bounding_box_mins[2]) };
}

void cartesian_mesh::display_info()
{
	std::cout << std::endl << "Mesh information: " << std::endl;
	std::cout << "   dimensions: " << dims << std::endl;
	std::cout << "Domain: " << "[" << bounding_box_mins[0] << "," << bounding_box_maxs[0] << "] " << "units" << " x "
			  << "[" << bounding_box_mins[1] << "," << bounding_box_maxs[1] << "] " << "units" << " x " << "["
			  << bounding_box_mins[2] << "," << bounding_box_maxs[2] << "] " << "units" << std::endl
			  << "   resolution: dx = " << voxel_shape[0] << " " << "units" << ", dy = " << voxel_shape[1] << " "
			  << "units" << ", dz = " << voxel_shape[2] << " " << "units";
	std::cout << std::endl
			  << "   voxels: " << voxel_count() << std::endl
			  << "   volume: "
			  << (std::size_t)(bounding_box_maxs[0] - bounding_box_mins[0])
					 * (std::size_t)(bounding_box_maxs[1] - bounding_box_mins[1])
					 * (std::size_t)(bounding_box_maxs[2] - bounding_box_mins[2])
			  << " cubic " << "units" << std::endl;

	return;
}
