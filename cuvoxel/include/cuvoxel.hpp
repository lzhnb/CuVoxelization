// Copyright 2022 Zhihao Liang
#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <thrust/sort.h>

using torch::Tensor;

namespace voxel {

struct PointToVoxelComparator
{
    PointToVoxelComparator(int32_t* __restrict__ _point_to_voxel):
        point_to_voxel(_point_to_voxel)
        {};

    int32_t *point_to_voxel;

    __host__ __device__
    bool operator()(const int32_t idx1, const int32_t idx2) const;
};

Tensor sort_point_ids(const Tensor&);
Tensor select_voxels(const Tensor&, const Tensor&, const int32_t, const int32_t);
void select_voxels_wrapper(const Tensor&, const Tensor&, Tensor&, Tensor&);

} // namespace voxel


// Utils
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)      \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x)

#define CHECK_CPU_INPUT(x)  \
    CHECK_CPU(x);           \
    CHECK_CONTIGUOUS(x)

