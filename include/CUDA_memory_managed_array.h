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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_ARRAY_H
#define EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_ARRAY_H

#include "my_cuda.h"
#include "CUDA_memory_managed_item.h"
#include <algorithm>

namespace edge_matching_puzzle
{
    /**
     * Class to deal with CUDA memory managed arrays of simple types like
     * uint32_t that are not derived from CUDA_memory_managed_item
     * @tparam T array type
     */
    template<typename T>
    class CUDA_memory_managed_array
#ifdef ENABLE_CUDA_CODE
    : public CUDA_memory_managed_item
#endif // 0
    {
      public:

        explicit
        CUDA_memory_managed_array(size_t p_size);

        /**
         * Constuctor with itnit value applye to all array items
         * @param p_size size of array
         * @param p_init_value init value for each item
         */
        explicit
        CUDA_memory_managed_array(size_t p_size
                                 ,T p_init_value
                                 );

        __host__ __device__
        T & operator[](std::size_t p_index);

        __host__ __device__
        const T & operator[](std::size_t p_index)const;

        ~CUDA_memory_managed_array();

      private:
#ifdef ENABLE_CUDA_CODE
        static_assert(!std::is_base_of<CUDA_memory_managed_item, T>::value, "CUDA_memory_managed_array is for basic types not derived from CUDA_memory_managed_item");
#endif // ENABLE_CUDA_CODE
        T * m_array_ptr;
    };

    //-------------------------------------------------------------------------
    template<typename T>
    CUDA_memory_managed_array<T>::CUDA_memory_managed_array(size_t p_size)
#ifdef ENABLE_CUDA_CODE
    :m_array_ptr(static_cast<T*>(CUDA_memory_managed_item::allocate(p_size * sizeof(T))))
#else // ENABLE_CUDA_CODE
    :m_array_ptr(new T[p_size])
#endif // ENABLE_CUDA_CODE
    {
    }

    //-------------------------------------------------------------------------
    template<typename T>
    CUDA_memory_managed_array<T>::CUDA_memory_managed_array(size_t p_size
                                                           ,T p_init_value
                                                           )
    :CUDA_memory_managed_array(p_size)
    {
        std::transform(&m_array_ptr[0], &m_array_ptr[p_size], &m_array_ptr[0], [&](T p_item){return p_init_value;});
    }

    //-------------------------------------------------------------------------
    template <typename T>
    CUDA_memory_managed_array<T>::~CUDA_memory_managed_array()
    {
#ifdef ENABLE_CUDA_CODE
        deallocate(m_array_ptr);
#else // ENABLE_CUDA_CODE
        delete[] m_array_ptr;
#endif // ENABLE_CUDA_CODE
    }

    //-------------------------------------------------------------------------
    template <typename T>
    __host__ __device__
    T &
    CUDA_memory_managed_array<T>::operator[](std::size_t p_index)
    {
        // Assert generate illegal warp instruction
        assert(m_array_ptr);
        return m_array_ptr[p_index];
    }

    //-------------------------------------------------------------------------
    template <typename T>
    __host__ __device__
    const T &
    CUDA_memory_managed_array<T>::operator[](std::size_t p_index) const
    {
        // Assert generate illegal warp instruction
        assert(m_array_ptr);
        return m_array_ptr[p_index];
    }
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_MEMORY_MANAGED_ARRAY_H
// EOF