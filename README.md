# Edge matching puzzle

Continuous integration with [Travis-Ci](https://travis-ci.org/quicky2000/edge_matching_puzzle) : ![Build Status](https://travis-ci.org/quicky2000/edge_matching_puzzle.svg?branch=master)

The aim of this project is the exploration of edge matching puzzles


More information can be found on : https://www.logre.eu/wiki/Projet_EMP_E2


License
-------
Please see [LICENSE](LICENSE) for info on the license.

Build
-----

Build process is the same used in [Travis file](.travis.yml)
Reference build can be found [here](https://travis-ci.org/quicky2000/edge_matching_puzzle)


CUDA
-----
CUDA code is designed to run on a **Nvidia GPU Tesla T4**

### Build with CUDA code enabled

With CMake use command

```
cmake  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 $QUICKY_REPOSITORY/edge_matching_puzzle/
```
