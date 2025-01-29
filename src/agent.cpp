#include "agent.h"

#include "agent_data.h"
#include "microenvironment.h"
#include "traits.h"

using namespace biofvm;

agent::agent(agent_id_t id, agent_data& data, index_t index) : index_(index), data_(data), id(id) {}

real_t* agent::secretion_rates() { return data_.secretion_rates.data() + index_ * data_.m.substrates_count; }

real_t* agent::saturation_densities() { return data_.saturation_densities.data() + index_ * data_.m.substrates_count; }

real_t* agent::uptake_rates() { return data_.uptake_rates.data() + index_ * data_.m.substrates_count; }

real_t* agent::net_export_rates() { return data_.net_export_rates.data() + index_ * data_.m.substrates_count; }

real_t* agent::internalized_substrates()
{
	return data_.internalized_substrates.data() + index_ * data_.m.substrates_count;
}

real_t* agent::fraction_released_at_death()
{
	return data_.fraction_released_at_death.data() + index_ * data_.m.substrates_count;
}

real_t* agent::fraction_transferred_when_ingested()
{
	return data_.fraction_transferred_when_ingested.data() + index_ * data_.m.substrates_count;
}

real_t& agent::volume() { return data_.volumes[index_]; }

real_t* agent::position() { return data_.positions.data() + index_ * data_.m.mesh.dims; }

point_t<real_t, 3> agent::get_position() const
{
	if (data_.m.mesh.dims == 1)
		return point_t<real_t, 3> { data_.positions[index_ * data_.m.mesh.dims], 0, 0 };
	else if (data_.m.mesh.dims == 2)
		return point_t<real_t, 3> { data_.positions[index_ * data_.m.mesh.dims],
									data_.positions[index_ * data_.m.mesh.dims + 1], 0 };
	else
		return point_t<real_t, 3> { data_.positions[index_ * data_.m.mesh.dims],
									data_.positions[index_ * data_.m.mesh.dims + 1],
									data_.positions[index_ * data_.m.mesh.dims + 2] };
}

point_t<index_t, 3> agent::voxel_index() const
{
	if (data_.m.mesh.dims == 1)
		return data_.m.mesh.voxel_position<1>(get_position().data());
	else if (data_.m.mesh.dims == 2)
		return data_.m.mesh.voxel_position<2>(get_position().data());
	else
		return data_.m.mesh.voxel_position<3>(get_position().data());
}

real_t* agent::nearest_density_vector()
{
	auto index = voxel_index();
	if (data_.m.mesh.dims == 1)
	{
		auto dens_l = layout_traits<1>::construct_density_layout(data_.m.substrates_count, data_.m.mesh.grid_shape);
		auto offset = dens_l | noarr::offset<'x', 's'>(index[0], 0);
		return data_.m.substrate_densities.get() + offset / sizeof(real_t);
	}
	else if (data_.m.mesh.dims == 2)
	{
		auto dens_l = layout_traits<2>::construct_density_layout(data_.m.substrates_count, data_.m.mesh.grid_shape);
		auto offset = dens_l | noarr::offset<'x', 'y', 's'>(index[0], index[1], 0);
		return data_.m.substrate_densities.get() + offset / sizeof(real_t);
	}
	else
	{
		auto dens_l = layout_traits<3>::construct_density_layout(data_.m.substrates_count, data_.m.mesh.grid_shape);
		auto offset = dens_l | noarr::offset<'x', 'y', 'z', 's'>(index[0], index[1], index[2], 0);
		return data_.m.substrate_densities.get() + offset / sizeof(real_t);
	}
}

point_t<real_t, 3> agent::nearest_gradient(index_t substrate_index) const
{
	auto index = voxel_index();
	if (data_.m.mesh.dims == 1)
	{
		auto grad_l = layout_traits<1>::construct_gradient_layout(data_.m.substrates_count, data_.m.mesh.grid_shape);

		return point_t<real_t, 3> {
			grad_l | noarr::get_at<'x', 's', 'd'>(data_.m.gradients.get(), index[0], substrate_index, noarr::lit<0>),
			grad_l | noarr::get_at<'x', 's', 'd'>(data_.m.gradients.get(), index[0], substrate_index, noarr::lit<1>),
			grad_l | noarr::get_at<'x', 's', 'd'>(data_.m.gradients.get(), index[0], substrate_index, noarr::lit<2>)
		};
	}
	else if (data_.m.mesh.dims == 2)
	{
		auto grad_l = layout_traits<2>::construct_gradient_layout(data_.m.substrates_count, data_.m.mesh.grid_shape);

		return point_t<real_t, 3> { grad_l
										| noarr::get_at<'x', 'y', 's', 'd'>(data_.m.gradients.get(), index[0], index[1],
																			substrate_index, noarr::lit<0>),
									grad_l
										| noarr::get_at<'x', 'y', 's', 'd'>(data_.m.gradients.get(), index[0], index[1],
																			substrate_index, noarr::lit<1>),
									grad_l
										| noarr::get_at<'x', 'y', 's', 'd'>(data_.m.gradients.get(), index[0], index[1],
																			substrate_index, noarr::lit<2>) };
	}
	else
	{
		auto grad_l = layout_traits<3>::construct_gradient_layout(data_.m.substrates_count, data_.m.mesh.grid_shape);

		return point_t<real_t, 3> {
			grad_l
				| noarr::get_at<'x', 'y', 'z', 's', 'd'>(data_.m.gradients.get(), index[0], index[1], index[2],
														 substrate_index, noarr::lit<0>),
			grad_l
				| noarr::get_at<'x', 'y', 'z', 's', 'd'>(data_.m.gradients.get(), index[0], index[1], index[2],
														 substrate_index, noarr::lit<1>),
			grad_l
				| noarr::get_at<'x', 'y', 'z', 's', 'd'>(data_.m.gradients.get(), index[0], index[1], index[2],
														 substrate_index, noarr::lit<2>)
		};
	}
}

index_t agent::index() const { return index_; }
