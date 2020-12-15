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
#ifndef EMP_CUDA_SITUATION_CAPABILITY_H
#define EMP_CUDA_SITUATION_CAPABILITY_H

#ifndef __NVCC__
#error This code should be compiled with nvcc
#endif // __NVCC__

#include "CUDA_piece_position_info.h"

namespace edge_matching_puzzle
{

    template <unsigned int SIZE>
    class situation_capability;

    template <unsigned int SIZE>
    std::ostream & operator<<(std::ostream & p_stream, const situation_capability<SIZE> & p_capability);

    /**
     * Represent capabilities for a situation:
     * _ for a position which oriented pieces are possible
     * _ for a piece which positions with orientation are possible
     * @tparam SIZE twice the number of pieces/positions as we have info for both
     */
    template <unsigned int SIZE>
    class CUDA_situation_capability
    : public CUDA_memory_managed_item
    {

        friend
        std::ostream & operator<< <>(std::ostream & p_stream, const situation_capability<SIZE> & p_capability);

      public:

        CUDA_situation_capability() = default;
        CUDA_situation_capability(const CUDA_situation_capability &) = default;
        CUDA_situation_capability & operator=(const CUDA_situation_capability &) = default;

        inline __host__ __device__
        const CUDA_piece_position_info &
        get_capability(unsigned int p_index) const;

        inline __host__ __device__
        CUDA_piece_position_info &
        get_capability(unsigned int p_index);

        inline
        void apply_and( const CUDA_situation_capability & p_a
                      , const CUDA_situation_capability & p_b
                      );

        inline __host__ __device__
        bool operator==(const CUDA_situation_capability &) const;

      private:
        CUDA_piece_position_info m_capability[SIZE];

        static_assert(!(SIZE % 2),"Situation capability size is odd whereas it should be even");
    };

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    __host__ __device__
    const CUDA_piece_position_info &
    CUDA_situation_capability<SIZE>::get_capability(unsigned int p_index) const
    {
        assert(p_index < SIZE);
        return m_capability[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    __host__ __device__
    CUDA_piece_position_info &
    CUDA_situation_capability<SIZE>::get_capability(unsigned int p_index)
    {
        assert(p_index < SIZE);
        return m_capability[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    void
    CUDA_situation_capability<SIZE>::apply_and( const CUDA_situation_capability & p_a
                                              , const CUDA_situation_capability & p_b
                                              )
    {
        std::transform( &(p_a.m_capability[0])
                      , &(p_a.m_capability[SIZE])
                      , &(p_b.m_capability[0])
                      , &(m_capability[0])
                      , [=](const CUDA_piece_position_info & p_first, const CUDA_piece_position_info & p_second)
                        {CUDA_piece_position_info l_result;
                        l_result.apply_and(p_first, p_second);
                         return l_result;
                        }
                      );
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    bool
    CUDA_situation_capability<SIZE>::operator==(const CUDA_situation_capability & p_operator) const
    {
        unsigned int l_index = 0;
        while(l_index < SIZE)
        {
            if(!(m_capability[l_index] == p_operator.m_capability[l_index]))
            {
                return false;
            }
            ++l_index;
        }
        return true;
        return false;
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    std::ostream & operator<<(std::ostream & p_stream, const CUDA_situation_capability<SIZE> & p_capability)
    {
        for(unsigned int l_index = 0; l_index < SIZE; ++l_index)
        {
            p_stream << "[" << l_index << "] =>" << std::endl << p_capability.m_capability[l_index] << std::endl;
        }
        p_stream << std::endl;
        return p_stream;
    }
}
#endif //EMP_CUDA_situation_capability_H
// EOF