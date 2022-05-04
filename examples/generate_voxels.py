import os
from typing import Tuple

import torch
import numpy as np
import numba as nb

import cuvoxel

@nb.jit(nopython=True)
def select_voxels(
    points: np.ndarray,
    voxels: np.ndarray,
    point_to_voxel: np.ndarray,
    visited: np.ndarray,
    max_points=35
) -> None:
    """select voxels from points
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxels: [M, max_points, ndim] float tensor.
        point_to_voxel: [N] int tensor.
        visited: [N] int tensor
        max_points: int. indicate maximum points contained in a voxel.
    """
    # scan points
    for point_id in range(points.shape[0]):
        voxel_id = point_to_voxel[point_id]
        if visited[voxel_id] < max_points:
            inner_id = visited[voxel_id]
            voxels[voxel_id, inner_id, :] = points[point_id]
            visited[voxel_id] += 1
        else:
            continue

def points_to_voxel_new(
    points: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    max_points: int = 35,
    reverse_index=True,
    max_voxels: int = 20000
) -> Tuple[np.ndarray]:
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    # filter_ids = ((points[:, :3] >= coors_range[:3]) * (points[:, :3] <= coors_range[3:])).all(1)
    # points = points[filter_ids]

    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    quantize_coors = np.floor(
        (points[:, :3] - coors_range[:3]) / voxel_size).astype(np.int32)

    filter_ids = ((quantize_coors >= 0) * (quantize_coors < grid_size)).all(1)
    quantize_coors = quantize_coors[filter_ids]
    points = points[filter_ids]

    import ipdb; ipdb.set_trace()
    coords, unq_ind, unq_inv, num_points_per_voxel = np.unique(
        quantize_coors, return_inverse=True, return_counts=True, axis=0, return_index=True)

    sort_ind = np.argsort(unq_ind)
    coords_origin = quantize_coors[unq_ind[sort_ind]]
    ind_temp = np.arange(0, len(sort_ind))
    target = np.zeros_like(ind_temp)
    np.put(target, sort_ind, ind_temp)
    unq_inv_origin = target[unq_inv]
    num_points_per_voxel_origin = np.zeros_like(num_points_per_voxel)
    np.put(num_points_per_voxel_origin, target, num_points_per_voxel)

    if reverse_index:
        # coords = coords[:, ::-1]
        coords_origin = coords_origin[:, ::-1]

    voxel_num = coords.shape[0]
    if voxel_num > max_voxels:
        # coords = coords[:max_voxels]  # [max_voxels, 3]
        # filter_ids = unq_inv < max_voxels
        # unq_inv = unq_inv[filter_ids]
        # points = points[filter_ids]
        # num_points_per_voxel = num_points_per_voxel[:max_voxels]
        # voxel_num = max_voxels
        coords_origin = coords_origin[:max_voxels]  # [max_voxels, 3]
        filter_ids = unq_inv_origin < max_voxels
        unq_inv_origin = unq_inv_origin[filter_ids]
        points = points[filter_ids]
        num_points_per_voxel_origin = num_points_per_voxel_origin[:max_voxels]
        voxel_num = max_voxels

    voxels = np.zeros(
        shape=(voxel_num, max_points, points.shape[-1]), dtype=points.dtype
    )
    visited = np.zeros(shape=(voxel_num,), dtype=np.int32)

    select_voxels(points, voxels, unq_inv_origin, visited, max_points)
    voxels = voxels[:voxel_num]
    num_points_per_voxel_origin = np.clip(
        num_points_per_voxel_origin, 0, max_points)

    return voxels, coords_origin, num_points_per_voxel_origin

if __name__ == "__main__":
    points = np.load(os.path.join(os.path.dirname(__file__), "points.npy"))

    voxels, coords_origin, num_points_per_voxel_origin = points_to_voxel_new(
        points,
        np.array([0.1, 0.1, 0.1]),
        coors_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    )
    np.save("voxels_gt.npy", voxels)
