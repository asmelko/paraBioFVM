#pragma once

#include <string>
#include <string_view>

namespace biofvm {

class vtk_serializer_base
{
protected:
	std::size_t iteration_;
	std::string output_dir_;
	std::string vtks_dir_;

	std::string pvd_contents_;

	void append_to_pvd(std::string_view vtk_file_name);

public:
	vtk_serializer_base(std::string_view output_dir, std::string_view vtks_dir_name);
};

} // namespace biofvm
