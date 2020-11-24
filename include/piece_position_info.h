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

#ifndef EMP_PIECE_POSITION_INFO_H
#define EMP_PIECE_POSITION_INFO_H

#ifdef __NVCC__
#include "CUDA_memory_managed_item.h"
#endif // __NVCC__
#include "emp_types.h"
#include <array>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace edge_matching_puzzle
{
    /**
     * Class storing informations related to a position or a piece
     * In case of position each bit represent a piece and an orientation
     * possible at this position
     * In case of piece each bit represent a position and an orientation
     * possible for this piece
     * Class is sized to be able to deal with Eternity2 puzzle
     * Eternity2 has 256 pieces/positions, pieces are square so have 4 possible
     * orientations so we need 1024 bits. They are split 32 32 bits words to be
     * manageable by a CUDA warp
     * 256 pieces/positions for a specific orientation need 8 words to be represented:
     * |   NORTH  ||   EAST   ||   SOUTH  ||   WEST   |
     * -----------||----------||----------||----------|
     * | W0 .. W7 || W0 .. W7 || W0 .. W7 || W0 .. W7 |
     */
    class piece_position_info
#ifdef __NVCC__
    : public CUDA_memory_managed_item
#endif // __NVCC__
    {
        friend
        std::ostream & operator<<(std::ostream & p_stream, const piece_position_info & p_info);

      public:

        inline
        piece_position_info();

        inline explicit
        piece_position_info(uint32_t p_value);

        piece_position_info(const piece_position_info & ) = default;
        piece_position_info & operator=(const piece_position_info &) = default;

        inline
        void clear_bit(unsigned int p_bit_index);

        inline
        void clear_bit( unsigned int p_id
                      , emp_types::t_orientation p_orientation
                      );

        inline
        void set_bit(unsigned int p_bit_index);

        inline
        void set_bit( unsigned int p_index
                    , emp_types::t_orientation p_orientation
                    );

        inline static
        void set_init_value(uint32_t p_value);

        inline
        void clear();

        /**
         * Word access for CUDA warp operations
         * @param p_index Index of word
         * @return word value
         */
        inline
#ifdef __NVCC__
        __host__ __device__
#endif // __NVCC__
        uint32_t get_word(unsigned int p_index) const;

        /**
         * Word access for CUDA warp operation
         * @param p_index Index of ward
         * @param p_word value to assign to word
         */
        inline
#ifdef __NVCC__
        __host__ __device__
#endif // __NVCC__
        void set_word(unsigned int p_index, uint32_t p_word);

        inline
        void apply_and( const piece_position_info & p_a
                      , const piece_position_info & p_b
                      );

        inline
#ifdef __NVCC__
        __host__ __device__
#endif // __NVCC__
        bool operator==(const piece_position_info &) const;

      private:

        inline
        void clear_bit( unsigned int p_word_index
                      , unsigned int p_bit_index
                      );

        inline
        void set_bit( unsigned int p_word_index
                    , unsigned int p_bit_index
                    );

#ifndef __NVCC__
        std::array<uint32_t, 32> m_info;
#else // __NVCC__
        uint32_t m_info[32];
#endif // __NVCC__

        // To do : replace by inline variable ( need to have recent CMake for CUDA c++17
        static
        uint32_t s_init_value;

    };

    //-------------------------------------------------------------------------
    piece_position_info::piece_position_info()
            : piece_position_info(s_init_value)
    {

    }

    //-------------------------------------------------------------------------
    piece_position_info::piece_position_info( uint32_t p_value)
    : m_info{ p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            , p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            , p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            , p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            }
    {

    }

    //-------------------------------------------------------------------------
    uint32_t
#ifdef __NVCC__
    __host__ __device__
#endif // __NVCC__
    piece_position_info::get_word(unsigned int p_index) const
    {
#ifndef __NVCC__
        assert(p_index < m_info.size());
#else // __NVCC__
        //assert(p_index < 32);
#endif // __NVCC__
        return m_info[p_index];
    }

    //-------------------------------------------------------------------------
    void
#ifdef __NVCC__
    __host__ __device__
#endif // __NVCC__
    piece_position_info::set_word( unsigned int p_index
                                 , uint32_t p_word
                                 )
    {
#ifndef __NVCC__
        assert(p_index < m_info.size());
#else // __NVCC__
        //assert(p_index < 32);
#endif // __NVCC__
        m_info[p_index] = p_word;

    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::apply_and( const piece_position_info & p_a
                                  , const piece_position_info & p_b
                                  )
    {
        for(unsigned int l_index = 0; l_index < 32; ++l_index)
        {
            m_info[l_index] = p_a.m_info[l_index] & p_b.m_info[l_index];
        }
    }

    //-------------------------------------------------------------------------
    bool
    piece_position_info::operator==(const piece_position_info & p_operand) const
    {
        unsigned int l_index = 0;
        while(l_index < 32)
        {
            if(m_info[l_index] != p_operand.m_info[l_index])
            {
                return false;
            }
            ++l_index;
        }
        return true;
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::clear_bit(unsigned int p_word_index,
                                   unsigned int p_bit_index
                                  )
    {
        assert(p_word_index < 32);
        assert(p_bit_index < 32);
        m_info[p_word_index] &= ~(1u << p_bit_index);
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::clear_bit(unsigned int p_bit_index)
    {
        assert(p_bit_index < 1024);
        clear_bit(p_bit_index / 32, p_bit_index % 32);
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::clear_bit( unsigned int p_id
                                  , emp_types::t_orientation p_orientation
                                  )
    {
        assert(p_id < 256);
        clear_bit(8 * static_cast<unsigned int>(p_orientation) + p_id / 32, p_id % 32);
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::set_bit( unsigned int p_word_index
                                , unsigned int p_bit_index
                                )
    {
        assert(p_word_index < 32);
        assert(p_bit_index < 32);
        m_info[p_word_index] |= (1u << p_bit_index);
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::set_bit(unsigned int p_bit_index)
    {
        assert(p_bit_index < 1024);
        set_bit(p_bit_index / 32, p_bit_index % 32);
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::set_bit( unsigned int p_index
                                , emp_types::t_orientation p_orientation
                                )
    {
        assert(p_index < 256);
        set_bit(8 * static_cast<unsigned int>(p_orientation) + p_index / 32, p_index % 32);
    }

    //-------------------------------------------------------------------------
    inline
    std::ostream & operator<<(std::ostream & p_stream, const piece_position_info & p_info)
    {
        for(unsigned int l_index = 0; l_index < 32; ++l_index)
        {
            if(0 == (l_index % 4))
            {
                p_stream << std::endl;
            }
            p_stream << "\t[" << l_index << "] = 0x" << std::hex << p_info.m_info[l_index] << std::dec;
        }
        return p_stream;
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::set_init_value(uint32_t p_value)
    {
        s_init_value = p_value;
    }

    //-------------------------------------------------------------------------
    void
    piece_position_info::clear()
    {
#ifndef __NVCC__
        std::transform(m_info.begin(), m_info.end(), m_info.begin(), [](uint32_t p_item){return 0x0;});
#else // __NVCC__
        for(unsigned int l_index = 0; l_index < 32; ++l_index)
        {
            m_info[l_index] = 0;
        }
#endif // __NVCC__
    }

}
#endif //EMP_PIECE_POSITION_INFO_H
// EOF
