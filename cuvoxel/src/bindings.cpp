// Copyright 2022 Zhihao Liang
#include "cuvoxel.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("select_voxels", &voxel::select_voxels);
    m.def("sort_point_ids", &voxel::sort_point_ids);
}

