/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_PIECE_POSITION_INFO_BASE_H
#define EDGE_MATCHING_PUZZLE_CUDA_PIECE_POSITION_INFO_BASE_H

#include "my_cuda.h"
#ifdef ENABLE_CUDA_CODE
#include "CUDA_memory_managed_item.h"
#endif // ENABLE_CUDA_CODE
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
     * The exact layout of bits is implemented by derived class
     * This class is agnostic and provide raw methods to access words and bits
     */
     class CUDA_piece_position_info_base
#ifdef ENABLE_CUDA_CODE
     : public my_cuda::CUDA_memory_managed_item
#endif // ENABLE_CUDA_CODE
     {

        friend
        std::ostream & operator<<(std::ostream & p_stream, const CUDA_piece_position_info_base & p_info);

       public:

         /**
          * Define word initial value used when calling empty constructor
          * @param p_value word init value
          */
         inline static
         void set_init_value(uint32_t p_value);

         /**
          * Word access for CUDA warp operations
          * @param p_index Index of word
          * @return word value
          */
         inline
         __host__ __device__
         uint32_t get_word(unsigned int p_index) const;

         /**
          * Word access for CUDA warp operation
          * @param p_index Index of ward
          * @param p_word value to assign to word
          */
         inline
         __host__ __device__
         void set_word(unsigned int p_index, uint32_t p_word);

         /**
          * Apply result of xor operator between information contained in 2 oprerands
          * @param p_a first operand
          * @param p_b second operand
          */
         inline
         __device__
         void apply_xor(const CUDA_piece_position_info_base & p_a
                       ,const CUDA_piece_position_info_base & p_b
                       );

         inline
         __device__
         void CUDA_and(const CUDA_piece_position_info_base & p_a
                      ,const CUDA_piece_position_info_base & p_b
                      );

         /**
          * AND operator to be used by CUDA threads to use the result in local variable and not in memory
          * @param p_a the mask to apply
          * @return result of AND between internal info and provided mask
          */
         [[nodiscard]]
         inline
         __device__
         uint32_t
         CUDA_and(const CUDA_piece_position_info_base & p_a);

       protected:

         inline
         CUDA_piece_position_info_base();

         /**
          * Constructor specifying init value for words
          * @param p_value word init value
          */
         inline explicit
         CUDA_piece_position_info_base(uint32_t p_value);

         CUDA_piece_position_info_base(const CUDA_piece_position_info_base & ) = default;
         CUDA_piece_position_info_base & operator=(const CUDA_piece_position_info_base &) = default;

         inline
         __host__ __device__
         void clear_bit(unsigned int p_word_index
                       ,unsigned int p_bit_index
                       );

         inline
         __host__ __device__
         void set_bit(unsigned int p_word_index
                     ,unsigned int p_bit_index
                     );

       private:

         inline
         void clear_bit(unsigned int p_bit_index);

         inline
         void set_bit(unsigned int p_bit_index);

         inline
         void clear();

         inline
         void apply_and(const CUDA_piece_position_info_base & p_a
                       ,const CUDA_piece_position_info_base & p_b
                       );

         inline
         __host__ __device__
         bool operator==(const CUDA_piece_position_info_base &) const;

         uint32_t m_info[32];

         // To do : replace by inline variable ( need to have recent CMake for CUDA c++17
         static
         uint32_t s_init_value;
     };


    //-------------------------------------------------------------------------
    CUDA_piece_position_info_base::CUDA_piece_position_info_base()
    :CUDA_piece_position_info_base(s_init_value)
    {
    }

    //-------------------------------------------------------------------------
    CUDA_piece_position_info_base::CUDA_piece_position_info_base( uint32_t p_value)
    : m_info{ p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            , p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            , p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            , p_value ,p_value , p_value, p_value, p_value, p_value, p_value, p_value
            }
    {
    }

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info_base::clear_bit(unsigned int p_bit_index)
    {
        assert(p_bit_index < 1024);
        clear_bit(p_bit_index / 32, p_bit_index % 32);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info_base::set_bit(unsigned int p_bit_index)
    {
        assert(p_bit_index < 1024);
        set_bit(p_bit_index / 32, p_bit_index % 32);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info_base::clear()
    {
        //std::transform(&m_info[0], &m_info[32], &m_info[0], [](uint32_t p_item){return 0x0;});
        for(unsigned int l_index = 0; l_index < 32; ++l_index)
        {
            m_info[l_index] = 0;
        }
    }

    //-------------------------------------------------------------------------
    uint32_t
    __host__ __device__
    CUDA_piece_position_info_base::get_word(unsigned int p_index) const
    {
        //assert(p_index < 32);
        return m_info[p_index];
    }

    //-------------------------------------------------------------------------
    void
    __host__ __device__
    CUDA_piece_position_info_base::set_word(unsigned int p_index
                                           ,uint32_t p_word
                                           )
    {
        //assert(p_index < 32);
        m_info[p_index] = p_word;
    }

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info_base::apply_and(const CUDA_piece_position_info_base & p_a
                                            ,const CUDA_piece_position_info_base & p_b
                                            )
    {
        std::transform(&(p_a.m_info[0]), &(p_a.m_info[32]), &(p_b.m_info[0]), &(m_info[0]), [=](uint32_t p_first, uint32_t p_second){return p_first & p_second;});
    }

    //-------------------------------------------------------------------------
    bool
    CUDA_piece_position_info_base::operator==(const CUDA_piece_position_info_base & p_operand) const
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
    __host__ __device__
    void
    CUDA_piece_position_info_base::clear_bit(unsigned int p_word_index
                                            ,unsigned int p_bit_index
                                            )
    {
        assert(p_word_index < 32);
        assert(p_bit_index < 32);
        m_info[p_word_index] &= ~(1u << p_bit_index);
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    void
    CUDA_piece_position_info_base::set_bit(unsigned int p_word_index
                                          ,unsigned int p_bit_index
                                          )
    {
        assert(p_word_index < 32);
        assert(p_bit_index < 32);
        m_info[p_word_index] |= (1u << p_bit_index);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info_base::set_init_value(uint32_t p_value)
    {
        s_init_value = p_value;
    }

    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_piece_position_info_base::apply_xor(const CUDA_piece_position_info_base & p_a
                                            ,const CUDA_piece_position_info_base & p_b
                                            )
    {
#ifdef ENABLE_CUDA_CODE
        m_info[threadIdx.x] = p_a.m_info[threadIdx.x] ^ p_b.m_info[threadIdx.x];
#else // ENABLE_CUDA_CODE
        for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
        {
            m_info[l_threadIdx_x] = p_a.m_info[l_threadIdx_x] ^ p_b.m_info[l_threadIdx_x];
        }
#endif // ENABLE_CUDA_CODE
    }

    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_piece_position_info_base::CUDA_and(const CUDA_piece_position_info_base & p_a
                                           ,const CUDA_piece_position_info_base & p_b
                                           )
    {
#ifdef ENABLE_CUDA_CODE
        m_info[threadIdx.x] = p_a.m_info[threadIdx.x] & p_b.m_info[threadIdx.x];
#else // ENABLE_CUDA_CODE
        for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
        {
            m_info[l_threadIdx_x] = p_a.m_info[l_threadIdx_x] & p_b.m_info[l_threadIdx_x];
        }
#endif // ENABLE_CUDA_CODE
    }

    //-------------------------------------------------------------------------
    __device__
    uint32_t
    CUDA_piece_position_info_base::CUDA_and(const CUDA_piece_position_info_base & p_a)
    {
#ifdef ENABLE_CUDA_CODE
        return m_info[threadIdx.x] & p_a.m_info[threadIdx.x];
#else // ENABLE_CUDA_CODE
        throw std::logic_error("No CPU implementation");
#endif // ENABLE_CUDA_CODE
    }

    //-------------------------------------------------------------------------
    inline
    std::ostream & operator<<(std::ostream & p_stream, const CUDA_piece_position_info_base & p_info)
    {
        p_stream << std::endl;
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

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_PIECE_POSITION_INFO_BASE_H
// EOF
