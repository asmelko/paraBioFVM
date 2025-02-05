#pragma once

#include <string_view>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataWriter.h>
#pragma GCC diagnostic pop

#include "microenvironment_serializer.h"

namespace biofvm {

class vtk_microenvironment_serializer : public microenvironment_serializer
{
	biofvm::index_t iteration_;

	std::string output_dir_;
	std::string vti_dir_;

	std::string pvd_contents_;

	vtkSmartPointer<vtkXMLImageDataWriter> writer_;
	vtkSmartPointer<vtkImageData> image_data_;

#ifdef USE_DOUBLES
	std::vector<vtkSmartPointer<vtkDoubleArray>> data_arrays_;
#else
	std::vector<vtkSmartPointer<vtkFloatArray>> data_arrays_;
#endif

public:
	vtk_microenvironment_serializer(std::string_view output_dir, microenvironment& m);

	void serialize_one_timestep(const microenvironment& m) override;
};

} // namespace biofvm
