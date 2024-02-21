# Edge matching puzzle

Continuous integration with [Travis-Ci](https://travis-ci.com/quicky2000/edge_matching_puzzle) : ![Build Status](https://travis-ci.com/quicky2000/edge_matching_puzzle.svg?branch=master)

The aim of this project is the exploration of edge matching puzzles


More information can be found on : https://www.logre.eu/wiki/Projet_EMP_E2


License
-------
Please see [LICENSE](LICENSE) for info on the license.

Build
-----

Build process is the same used in [Travis file](.travis.yml)
Reference build can be found [here](https://travis-ci.com/quicky2000/edge_matching_puzzle)


CUDA
-----
CUDA code is designed to run on a **Nvidia GPU Tesla T4**

### Build with CUDA code enabled

With CMake and CUDA 10, g++ > 9 is not supported so use the following command to force use of gcc 8

```
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 $QUICKY_REPOSITORY/edge_matching_puzzle/
```

With CMake and CUDA 11.1 nvcc is not in default path so specify location of nvcc with following command to force use of gcc 8

```
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc $QUICKY_REPOSITORY/edge_matching_puzzle/
```

On OpenSuse Tumbleweed with CUDA 12.3 and gcc-12.3

```
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_ARCHITECTURES=75  $QUICKY_REPOSITORY/edge_matching_puzzle/

```
