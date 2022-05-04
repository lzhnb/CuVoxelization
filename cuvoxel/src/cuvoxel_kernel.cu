#include "cuvoxel.hpp"

// KERNELS
__global__ void select_voxels_kernel(
    const float* __restrict__ points,
    const int32_t* __restrict__ point_to_voxel,
    const int32_t num_points,
    const int32_t ndim,
    const int32_t num_voxels,
    const int32_t max_points,
    int32_t* __restrict__ visited,
    // output
    float* __restrict__ voxels
) {
	const int32_t point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_id >= num_points) { return; }
    
    const int32_t voxel_id = point_to_voxel[point_id];
    assert(voxel_id < num_voxels);
    const int32_t count = atomicAdd(visited + voxel_id, 1);
    if (count < max_points) {
// #pragma unroll
        for (int32_t i = 0; i < ndim; ++i) {
            voxels[voxel_id * (max_points * ndim) + count * ndim + i] = points[point_id * ndim + i];
        }
    }
}

// WRAPPER
void voxel::select_voxels_wrapper(
    const torch::Tensor& points,
    const torch::Tensor& point_to_voxel,
    torch::Tensor& visited,
    // output
    torch::Tensor& voxels
){
    // init for parallel
    const int32_t num_points = points.size(0), ndim = points.size(1);
    const int32_t num_voxels = voxels.size(0), max_points = voxels.size(1);
    const int32_t threads = 1024;
    const int32_t blocks = static_cast<int32_t>((num_points - 1) / threads) + 1;

	select_voxels_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        point_to_voxel.data_ptr<int32_t>(),
        num_points,
        ndim,
        num_voxels,
        max_points,
        visited.data_ptr<int32_t>(),
        // output
        voxels.data_ptr<float>());
}

