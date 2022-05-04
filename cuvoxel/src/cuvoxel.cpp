// Copyright 2022 Zhihao Liang
#include <cstdint>
#include <iostream>
#include <pybind11/functional.h>

#include "cuvoxel.hpp"

torch::Tensor voxel::select_voxels(
    const torch::Tensor& points,           // [num_points, ndim]
    const torch::Tensor& point_to_voxel,   // [num_points]
    const int32_t num_voxels,
    const int32_t max_points
) {
    // check
    CHECK_INPUT(points);
    TORCH_CHECK(points.ndimension() == 2);
    TORCH_CHECK(points.scalar_type() == torch::kFloat)
    CHECK_INPUT(point_to_voxel);
    TORCH_CHECK(point_to_voxel.ndimension() == 1);
    TORCH_CHECK(point_to_voxel.scalar_type() == torch::kInt)

    const int32_t ndim = points.size(1);
    torch::Tensor voxels = torch::zeros({num_voxels, max_points, ndim},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor visited = torch::zeros({num_voxels + 1},
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    
    voxel::select_voxels_wrapper(points, point_to_voxel, visited, voxels);
    
    return voxels;
}
