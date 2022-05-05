# CuVoxelization
`CuVoxelization` is an **CUDA** implementation of voxelization. The given points is a two-dimension `torch.Tensor # [num_points, 3]`, and the output voxels is a three-dimension `torch.Tensor # [num_voxels, max_points, 3]`.

## Requirements
The enviroment of my developer machine:
- Python 3.8.8+
- PyTorch 1.10.2
- CUDA 11.1


## Installation
```sh
python setup.py install
```
Or use:
```sh
pip install .
```
Or use:
```sh
pip install https://github.com/lzhnb/CuVoxelization
```

## TODO
- [x] Examples (More Example)
- [ ] Optimize the code
- [ ] More elegant Python Wrapper
- [ ] Support backward
- [ ] Visualization

## Example
Put the `points.npy` file under `examples` directory, then run
```sh
python verify.py
```
