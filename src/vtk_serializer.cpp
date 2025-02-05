#include "vtk_serializer.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <noarr/structures/extra/funcs.hpp>
#include <noarr/structures/structs/blocks.hpp>
#include <vtkCellData.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>

#include "microenvironment.h"
#include "traits.h"

using namespace biofvm;

vtk_microenvironment_serializer::vtk_microenvironment_serializer(std::string_view output_dir, microenvironment& m)
	: iteration_(0),
	  output_dir_(output_dir),
	  vti_dir_(std::filesystem::path(vti_dir_) / "vtk_microenvironment"),
	  writer_(vtkSmartPointer<vtkXMLImageDataWriter>::New()),
	  image_data_(vtkSmartPointer<vtkImageData>::New())
{
	std::filesystem::create_directories(vti_dir_);

	auto x_extent_start = m.mesh.bounding_box_mins[0] / m.mesh.voxel_shape[0];
	auto x_extent_end = x_extent_start + m.mesh.grid_shape[0] - 1;

	auto y_extent_start = m.mesh.bounding_box_mins[1] / m.mesh.voxel_shape[1];
	auto y_extent_end = y_extent_start + m.mesh.grid_shape[1] - 1;

	auto z_extent_start = m.mesh.bounding_box_mins[2] / m.mesh.voxel_shape[2];
	auto z_extent_end = z_extent_start + m.mesh.grid_shape[2] - 1;

	image_data_->SetExtent(x_extent_start, x_extent_end, y_extent_start, y_extent_end, z_extent_start, z_extent_end);
	image_data_->SetSpacing(m.mesh.voxel_shape[0], m.mesh.voxel_shape[1], m.mesh.voxel_shape[2]);

	data_arrays_.reserve(m.substrates_count);
	for (auto i = 0; i < m.substrates_count; ++i)
	{
		data_arrays_.emplace_back(vtkSmartPointer<vtkFloatArray>::New());
		data_arrays_[i]->SetNumberOfComponents(1);
		data_arrays_[i]->SetNumberOfTuples(m.mesh.voxel_count());
		data_arrays_[i]->SetName(m.substrates_names[i].c_str());
		image_data_->GetCellData()->AddArray(data_arrays_[i]);
	}

	writer_->SetInputData(image_data_);

	pvd_contents_ = R"(<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>)";
}

void vtk_microenvironment_serializer::serialize_one_timestep(const microenvironment& m)
{
	auto l = layout_traits<3>::construct_density_layout(m.substrates_count, m.mesh.grid_shape);
	auto linearized_l = l ^ noarr::merge_blocks<'z', 'y'>() ^ noarr::merge_blocks<'y', 'x'>();

	for (size_t voxel_idx = 0; voxel_idx < (l | noarr::get_length<'x'>()); voxel_idx++)
		for (index_t s_idx = 0; s_idx < m.substrates_count; ++s_idx)
			data_arrays_[s_idx]->SetValue(
				voxel_idx, linearized_l | noarr::get_at<'x', 's'>(m.substrate_densities.get(), voxel_idx, s_idx));

	std::ostringstream ss;

	ss << (std::filesystem::path(vti_dir_) / "microenvironment_").string() << std::setw(6) << std::setfill('0')
	   << iteration_ << ".vti";

	writer_->SetFileName(ss.str().c_str());
	writer_->Write();

	pvd_contents_ +=
		"    <DataSet timestep=" + std::to_string(iteration_) + R"(" group="" part="0" file=")" + ss.str() + R"(" />
)";

	std::ofstream pvd_file(std::filesystem::path(vti_dir_) / "microenvironment.pvd");

	pvd_file << pvd_contents_ << R"(  </Collection>
</VTKFile>)";

	pvd_file.close();

	iteration_++;
}
