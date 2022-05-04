// Copyright 2022 Zhihao Liang
#include "cuvoxel.hpp"


__host__ __device__ bool voxel::PointToVoxelComparator::operator()(const int32_t idx1, const int32_t idx2) const
{
    if (point_to_voxel[idx1] < point_to_voxel[idx2]) {
        return true;
    } else if (point_to_voxel[idx1] == point_to_voxel[idx2]) {
        return idx1 < idx2;
    } else { return false; }
}

// KERNELS
__global__ void find_voxel_to_point_kernel(
    const int32_t* __restrict__ point_to_voxel, // [num_points]
    const int32_t* __restrict__ indices, // [num_points]
    const int32_t num_points,
    // output
    int32_t* __restrict__ visited
) {
	const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) { return; }

    const int32_t point_id = indices[tid];
    const int32_t voxel_id = point_to_voxel[point_id];
    const int32_t count = atomicAdd(visited + voxel_id, 1);
}

__global__ void select_voxels_kernel(
    const float* __restrict__ points, // [num_points, 3]
    const int32_t ndim,
    const int32_t num_voxels,
    const int32_t max_points,
    int32_t* __restrict__ indices, // [num_points]
    int32_t* __restrict__ offsets, // [num_voxels + 1]
    // output
    float* __restrict__ voxels // [num_voxels, max_points, ndim]
) {
	const int32_t voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= num_voxels) { return; }

    const int32_t start = offsets[voxel_id], end = offsets[voxel_id + 1];
    if (start == end) { return; }
    for (int32_t i = start; i < min(end, start + max_points); ++i) {
        const int32_t point_id = indices[i];
        for (int32_t j = 0; j < ndim; ++j) {
            voxels[voxel_id * (max_points * ndim) + (i - start) * ndim + j] = points[point_id * ndim + j];
        }
    }
}

torch::Tensor voxel::sort_point_ids(const torch::Tensor& point_to_voxel){ // [num_points]
    // check
    CHECK_INPUT(point_to_voxel);
    TORCH_CHECK(point_to_voxel.ndimension() == 1);
    TORCH_CHECK(point_to_voxel.scalar_type() == torch::kInt)

    const int32_t num_points = point_to_voxel.size(0);
    torch::Tensor indices = torch::arange(0, num_points, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    int32_t* point_to_voxel_ptr = point_to_voxel.data_ptr<int32_t>();
    int32_t* indices_ptr = indices.data_ptr<int32_t>();
    thrust::sort(
        thrust::device,
        indices_ptr,
        indices_ptr + num_points,
        PointToVoxelComparator(point_to_voxel_ptr)
    );
    
    return indices;
}

// WRAPPER
void voxel::select_voxels_wrapper(
    const torch::Tensor& points, // [num_points, ndim]
    const torch::Tensor& point_to_voxel, // [num_points]
    torch::Tensor& visited, // [num_voxels + 1]
    // output
    torch::Tensor& voxels // [num_voxels, max_points, ndim]
){
    // init for parallel
    const int32_t num_points = points.size(0), ndim = points.size(1);
    const int32_t num_voxels = voxels.size(0), max_points = voxels.size(1);
    const int32_t threads = 1024;
    const int32_t point_blocks = static_cast<int32_t>((num_points - 1) / threads) + 1;

    torch::Tensor indices = voxel::sort_point_ids(point_to_voxel);

    find_voxel_to_point_kernel<<<point_blocks, threads>>>(
        point_to_voxel.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(),
        num_points,
        // output
        visited.data_ptr<int32_t>() + 1
    );

    torch::Tensor offsets = torch::cumsum(visited, 0, torch::kInt); // [num_points + 1]

    const int32_t voxel_blocks = static_cast<int32_t>((num_voxels - 1) / threads) + 1;
	select_voxels_kernel<<<voxel_blocks, threads>>>(
        points.data_ptr<float>(),
        ndim,
        num_voxels,
        max_points,
        indices.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        // output
        voxels.data_ptr<float>()); // [num_voxels, max_points, ndim]
}

