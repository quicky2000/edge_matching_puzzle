/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2020  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_ITEM_H
#define EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_ITEM_H

#ifdef __NVCC__

#include "quicky_exception.h"
#include <memory>
#include <inttypes.h>

namespace edge_matching_puzzle
{
    /**
     * Class to be used with CUDA compiler.
     * It overloads new operator and destructor to use CUDA memory managed
     * functions.
     * To make a class CUDA memory managed ti has to derived from this one
     */
    class CUDA_memory_managed_item
    {
      public:

        inline
        void * operator new(size_t p_size);

        inline
        void * operator new[](size_t p_size);

        inline
        void operator delete(void * p_ptr);

        inline
        void operator delete[](void * p_ptr);

      protected:
	inline static
	void * allocate(size_t p_size);

	inline static
	void deallocate(void * p_ptr);
    };

    //-------------------------------------------------------------------------
    void * CUDA_memory_managed_item::operator new(size_t p_size)
    {
	return allocate(p_size);
    }

    //-------------------------------------------------------------------------
    void * CUDA_memory_managed_item::operator new[](size_t p_size)
    {
	return allocate(p_size);
    }

    //-------------------------------------------------------------------------
    void * CUDA_memory_managed_item::allocate(size_t p_size)
    {
        void * l_result;
        cudaError_t l_cuda_status = cudaMallocManaged(&l_result, p_size);
        if(cudaSuccess != l_cuda_status)
        {
            throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);
        }
        l_cuda_status = cudaDeviceSynchronize();
        if(cudaSuccess != l_cuda_status)
        {
            throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);
        }
#ifdef CUDA_DEBUG
	printf("Cuda malloc size " PRIx32 " @ %" PRIx64 "\n", p_size, l_result);
#endif // CUDA_DEBUG
        return l_result;
    }

    //-------------------------------------------------------------------------
    void CUDA_memory_managed_item::operator delete(void *p_ptr)
    {
         deallocate(p_ptr);
    }

    //-------------------------------------------------------------------------
    void CUDA_memory_managed_item::operator delete[](void *p_ptr)
    {
         deallocate(p_ptr);
    }

    //-------------------------------------------------------------------------
    void CUDA_memory_managed_item::deallocate(void *p_ptr)
    {
        cudaError_t l_cuda_status = cudaDeviceSynchronize();
        if(cudaSuccess != l_cuda_status)
        {
            throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);
        }
        l_cuda_status = cudaFree(p_ptr);
        if(cudaSuccess != l_cuda_status)
        {
            throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);
        }
    }
}

#endif // __NVCC__
#endif //EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_ITEM_H
// EOF
