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

#include "serializer.h"
#include "vtk_serializer_base.h"

namespace biofvm {

class vtk_serializer : public serializer, public vtk_serializer_base
{
	vtkSmartPointer<vtkXMLImageDataWriter> writer_;
	vtkSmartPointer<vtkImageData> image_data_;

#ifdef USE_DOUBLES
	std::vector<vtkSmartPointer<vtkDoubleArray>> data_arrays_;
#else
	std::vector<vtkSmartPointer<vtkFloatArray>> data_arrays_;
#endif

public:
	vtk_serializer(std::string_view output_dir, microenvironment& m);

	void serialize_one_timestep(const microenvironment& m) override;
};

} // namespace biofvm
