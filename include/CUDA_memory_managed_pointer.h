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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_POINTER_H
#define EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_POINTER_H


#include "CUDA_memory_managed_item.h"

namespace edge_matching_puzzle
{
    /**
     * Class stored to CUDA managed memory used as pointer
     * The overload of new and delete operator from CUDA_memory_managed_item
     * hide CUDA API memory managment calls making these objects look like
     * normal objects
     * @tparam T pointed type
     */
    template<typename T>
    class CUDA_memory_managed_ptr
#ifdef __NVCC__
    : public CUDA_memory_managed_item
#endif // __NVCC__
    {
      public:

        inline
        CUDA_memory_managed_ptr();

        inline
        CUDA_memory_managed_ptr(T * p_ptr);

        inline
#ifdef __NVCC__
        __device__ __host__
#endif // __NVCC__
        const T & operator*() const;

        inline
#ifdef __NVCC__
        __device__ __host__
#endif // __NVCC__
        T & operator*();

        inline
#ifdef __NVCC__
        __device__ __host__
#endif // __NVCC__
        T * operator->();

        inline
#ifdef __NVCC__
        __device__ __host__
#endif // __NVCC__
        const T * operator->() const;

        inline
#ifdef __NVCC__
        __device__ __host__
#endif // __NVCC__
        T * get();

        inline
#ifdef __NVCC__
        __device__ __host__
#endif // __NVCC__
        const T * get() const;

      private:
        T * m_ptr;
    };

    //-------------------------------------------------------------------------
    template<typename T>
    CUDA_memory_managed_ptr<T>::CUDA_memory_managed_ptr()
    : m_ptr(nullptr)
    {
    }

    //-------------------------------------------------------------------------
    template <typename T>
#ifdef __NVCC__
    __device__ __host__
#endif // __NVCC__
    const T &
    CUDA_memory_managed_ptr<T>::operator*() const
    {
        assert(m_ptr);
        return *m_ptr;
    }

    //-------------------------------------------------------------------------
    template <typename T>
#ifdef __NVCC__
    __device__ __host__
#endif // __NVCC__
    T &
    CUDA_memory_managed_ptr<T>::operator*()
    {
        assert(m_ptr);
        return *m_ptr;
    }

    //-------------------------------------------------------------------------
    template <typename T>
#ifdef __NVCC__
    __device__ __host__
#endif // __NVCC__
    T *
    CUDA_memory_managed_ptr<T>::operator->()
    {
        return m_ptr;
    }

    //-------------------------------------------------------------------------
    template <typename T>
#ifdef __NVCC__
    __device__ __host__
#endif // __NVCC__
    const T *
    CUDA_memory_managed_ptr<T>::operator->() const
    {
        return m_ptr;
    }

    //-------------------------------------------------------------------------
    template <typename T>
#ifdef __NVCC__
    __device__ __host__
#endif // __NVCC__
    T *
    CUDA_memory_managed_ptr<T>::get()
    {
        return m_ptr;
    }

    //-------------------------------------------------------------------------
    template <typename T>
#ifdef __NVCC__
    __device__ __host__
#endif // __NVCC__
    const T *
    CUDA_memory_managed_ptr<T>::get() const
    {
        return m_ptr;
    }

    //-------------------------------------------------------------------------
    template <typename T>
    CUDA_memory_managed_ptr<T>::CUDA_memory_managed_ptr(T * p_ptr)
    : m_ptr(p_ptr)
    {
    }
}

#endif //EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_POINTER_H
// EOF