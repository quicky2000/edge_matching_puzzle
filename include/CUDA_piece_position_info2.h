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

#ifndef EMP_CUDA_PIECE_POSITION_INFO2_H
#define EMP_CUDA_PIECE_POSITION_INFO2_H

#include "CUDA_piece_position_info_base.h"
#include "emp_types.h"
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
     * 256 pieces/positions for a specific orientation need 8 words to be represented:
     * Each word represent 8 pieces with their 4 orientations
     * |                 Word 0                        |
     * |-----------------------------------------------|
     * | Piece 0 | Piece 1 | ... |  Piece 7 | Piece 8  |
     * |---------|---------|-----|----------|----------|
     * | b0 - b3 | b4 - b7 | ... | b24 - b27| b28 - b31|
     */

    class CUDA_piece_position_info2
    : public CUDA_piece_position_info_base
    {
        friend
        std::ostream & operator<<(std::ostream & p_stream
                                 ,const CUDA_piece_position_info2 & p_info
                                 );

      public:

        inline
        void clear_bit(unsigned int p_index
                      ,emp_types::t_orientation p_orientation
                      );

        inline
        __device__
        void
        clear_bit(unsigned int p_word_index
                 ,unsigned int p_bit_index
                 );

        inline
        __host__ __device__
        void set_bit(unsigned int p_index
                    ,emp_types::t_orientation p_orientation
                    );

        inline static
        __host__ __device__
        unsigned int compute_word_index(unsigned int p_index
                                       ,emp_types::t_orientation p_orientation
                                       );

        inline static
        __host__ __device__
        unsigned int compute_bit_index(unsigned int p_index
                                       ,emp_types::t_orientation p_orientation
                                       );

        inline static
        __host__ __device__
        unsigned int compute_piece_index(unsigned int p_word_index
                                        ,unsigned int p_bit_index
                                        );

        /**
         * Compute index of word where info corresponding to piece is store
         * @param p_piece_index : Piece index ( 0 to n-1 with n the piece number)
         * @return index of word containing piece information
         */
        inline static
        __host__ __device__
        unsigned int
        compute_piece_word_index(unsigned int p_piece_index);

        /**
         * Compute index of first bit in word representing information for the
         * piece whose id is provided as parameter
         * @param p_piece_index : Piece index ( 0 to n-1 with n the piece number)
         * @return index of first bit in word containing piece information
         */
        inline static
        __host__ __device__
        unsigned int
        compute_piece_bit_index(unsigned int p_piece_index);

        inline static
        emp_types::t_orientation compute_orientation(unsigned int p_word_index
                                                    ,unsigned int p_bit_index
                                                    );

        inline static
        __host__ __device__
        unsigned int compute_orientation_index(unsigned int p_word_index
                                              ,unsigned int p_bit_index
                                              );

        inline static
        __host__ __device__
        uint32_t compute_piece_mask(unsigned int p_bit_index);

      private:



    };

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info2::clear_bit(unsigned int p_index
                                        ,emp_types::t_orientation p_orientation
                                        )
    {
        assert(p_index < 256);
        CUDA_piece_position_info_base::clear_bit(compute_word_index(p_index, p_orientation), compute_bit_index(p_index, p_orientation));
    }

    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_piece_position_info2::clear_bit(unsigned int p_word_index
                                        ,unsigned int p_bit_index
                                        )
    {
        CUDA_piece_position_info_base::clear_bit(p_word_index, p_bit_index);
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    void
    CUDA_piece_position_info2::set_bit(unsigned int p_index
                                      ,emp_types::t_orientation p_orientation
                                      )
    {
        assert(p_index < 256);
        CUDA_piece_position_info_base::set_bit(compute_word_index(p_index, p_orientation), compute_bit_index(p_index, p_orientation));
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    unsigned int
    CUDA_piece_position_info2::compute_word_index(unsigned int p_index
                                                 ,emp_types::t_orientation
                                                 )
    {
        return p_index / 8;
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    unsigned int
    CUDA_piece_position_info2::compute_bit_index(unsigned int p_index,
                                                 emp_types::t_orientation p_orientation
                                                )
    {
        return static_cast<unsigned int>(p_orientation) + 4 * (p_index % 8);
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    unsigned int
    CUDA_piece_position_info2::compute_piece_index(unsigned int p_word_index
                                                  ,unsigned int p_bit_index
                                                  )
    {
        return 8 * p_word_index + p_bit_index / 4;
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    unsigned int
    CUDA_piece_position_info2::compute_piece_word_index(unsigned int p_piece_index)
    {
        return p_piece_index / 8;
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    unsigned int
    CUDA_piece_position_info2::compute_piece_bit_index(unsigned int p_piece_index)
    {
        return 4 * (p_piece_index % 8);
    }

    //-------------------------------------------------------------------------
    emp_types::t_orientation
    CUDA_piece_position_info2::compute_orientation(unsigned int p_word_index
                                                  ,unsigned int p_bit_index
                                                  )
    {
        return static_cast<emp_types::t_orientation>(compute_orientation_index(p_word_index, p_bit_index));
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    unsigned int
    CUDA_piece_position_info2::compute_orientation_index(unsigned int
                                                        ,unsigned int p_bit_index
                                                        )
    {
        return p_bit_index % 4;
    }

    //-------------------------------------------------------------------------
    inline
    std::ostream & operator<<(std::ostream & p_stream
                             ,const CUDA_piece_position_info2 & p_info
                             )
    {
        for(unsigned int l_word_index = 0; l_word_index < 32; ++l_word_index)
        {
            uint32_t l_word = p_info.get_word(l_word_index);
            if(l_word)
            {
		p_stream << "/" << std::string(8,'-') << " Word[" << std::setw(2) << l_word_index << "] : 0x" << std::setfill('0') << std::setw(8) << std::hex << l_word << std::dec << std::setfill(' ') << " " << std::string(8, '-') << "\\" << std::endl;
                for (unsigned int l_internal_index = 0; l_internal_index < 8; ++l_internal_index)
                {
                    p_stream << "|" << std::setw(4) << 8 * l_word_index + l_internal_index;
                }
                p_stream << "|" << std::endl;
                for (unsigned int l_internal_index = 0; l_internal_index < 8; ++l_internal_index)
                {
                    p_stream << "|";
                    uint32_t l_piece_info = l_word & 0xF;
                    std::string l_string;
                    for (auto l_iter: emp_types::get_orientations())
                    {
                        if ((1u << static_cast<uint32_t>(l_iter)) & l_piece_info)
                        {
                            l_string += emp_types::orientation2short_string(l_iter);
                        }
                        else
                        {
                            l_string += " ";
                        }
                    }
                    p_stream << l_string;
                    l_word = l_word >> 4u;
                }
                p_stream << "|" << std::endl;
            }
        }
        return p_stream;
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    uint32_t
    CUDA_piece_position_info2::compute_piece_mask(unsigned int p_bit_index)
    {
        return 0xFu << 4 * (p_bit_index / 4);
    }

}
#endif //EMP_CUDA_PIECE_POSITION_INFO2_H
// EOF
