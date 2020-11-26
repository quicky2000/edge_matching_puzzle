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
#ifndef EMP_SITUATION_CAPABILITY_H
#define EMP_SITUATION_CAPABILITY_H

#include "piece_position_info.h"
#include <array>

#ifndef __NVCC__
#define __host__
#define __device__
#endif // __NVCC__

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
    class situation_capability
#ifdef __NVCC__
    : public CUDA_memory_managed_item
#endif // __NVCC__
    {

        friend
        std::ostream & operator<< <>(std::ostream & p_stream, const situation_capability<SIZE> & p_capability);

      public:

        situation_capability() = default;
        situation_capability(const situation_capability &) = default;
        situation_capability & operator=(const situation_capability &) = default;

        inline __host__ __device__
        const piece_position_info &
        get_capability(unsigned int p_index) const;

        inline __host__ __device__
        piece_position_info &
        get_capability(unsigned int p_index);

        inline
        void apply_and( const situation_capability & p_a
                      , const situation_capability & p_b
                      );

        inline __host__ __device__
        bool operator==(const situation_capability &) const;

      private:
#ifndef __NVCC__
        std::array<piece_position_info, SIZE> m_capability;
#else // __NVCC__
        piece_position_info m_capability[SIZE];
#endif // __NVCC__

        static_assert(!(SIZE % 2),"Situation capability size is odd whereas it should be even");
    };

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    __host__ __device__
    const piece_position_info &
    situation_capability<SIZE>::get_capability(unsigned int p_index) const
    {
        assert(p_index < SIZE);
        return m_capability[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    __host__ __device__
    piece_position_info &
    situation_capability<SIZE>::get_capability(unsigned int p_index)
    {
        assert(p_index < SIZE);
        return m_capability[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    void
    situation_capability<SIZE>::apply_and( const situation_capability & p_a
                                         , const situation_capability & p_b
                                         )
    {
        for(unsigned int l_index = 0; l_index < SIZE; ++l_index)
        {
            m_capability[l_index].apply_and(p_a.m_capability[l_index], p_b.m_capability[l_index]);
        }
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    bool
    situation_capability<SIZE>::operator==(const situation_capability & p_operator) const
    {
#ifndef __NVCC__
        return m_capability == p_operator.m_capability;
#else // __NVCC__
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
#endif // __NVCC__
        return false;
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    std::ostream & operator<<(std::ostream & p_stream, const situation_capability<SIZE> & p_capability)
    {
        for(unsigned int l_index = 0; l_index < SIZE; ++l_index)
        {
            p_stream << "[" << l_index << "] =>" << std::endl << p_capability.m_capability[l_index] << std::endl;
        }
        p_stream << std::endl;
        return p_stream;
    }
}
#endif //EMP_SITUATION_CAPABILITY_H
