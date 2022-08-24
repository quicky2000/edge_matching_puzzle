/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2021  Julien Thevenon ( julien_thevenon at yahoo.fr )

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#ifndef EDGE_MATCHING_PUZZLE_CUDA_COMMON_H
#define EDGE_MATCHING_PUZZLE_CUDA_COMMON_H

#include "my_cuda.h"
#ifdef ENABLE_CUDA_CODE
#include "thrust/version.h"
#include "quicky_exception.h"
#endif // ENABLE_CUDA_CODE
#include <iostream>

namespace edge_matching_puzzle
{
    inline
    void CUDA_info()
    {
#ifdef ENABLE_CUDA_CODE
        std::cout << "CUDA version  : " << CUDART_VERSION << std::endl;
        std::cout << "THRUST version: " << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << "." << THRUST_SUBMINOR_VERSION << std::endl;

        int l_cuda_device_nb = 0;
        cudaError_t l_cuda_status = cudaGetDeviceCount(&l_cuda_device_nb);
        if (cudaSuccess != l_cuda_status)
        {
            throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);
        }
        std::cout << "Number of CUDA devices: " << l_cuda_device_nb << std::endl;

        for (int l_device_index = 0; l_device_index < l_cuda_device_nb; ++l_device_index)
        {
            std::cout << "Cuda device[" << l_device_index << "]" << std::endl;
            cudaDeviceProp l_properties;
            cudaGetDeviceProperties(&l_properties, l_device_index);
            std::cout << R"(\tName                      : ")" << l_properties.name << R"(")" << std::endl;
            std::cout << "\tDevice compute capability : " << l_properties.major << "." << l_properties.minor << std::endl;
            std::cout << "\tWarp size                 : " << l_properties.warpSize << std::endl;
            if (l_properties.warpSize != 32)
            {
                throw quicky_exception::quicky_logic_exception("Unsupported warp size" + std::to_string(l_properties.warpSize), __LINE__, __FILE__);
            }
            std::cout << "\tMultiprocessor count      : " << l_properties.multiProcessorCount << std::endl;
            std::cout << "\tManaged Memory            : " << l_properties.managedMemory << std::endl;
            if (!l_properties.managedMemory)
            {
                throw quicky_exception::quicky_logic_exception("Managed memory is not supported", __LINE__, __FILE__);
            }
            std::cout << std::endl;
        }
#else // ENABLE_CUDA_CODE
        std::cout << "Software not compiled with CUDA" << std::endl;
#endif // ENABLE_CUDA_CODE

    }
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_COMMON_H
// EOF