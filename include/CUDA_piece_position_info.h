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

#ifndef EMP_CUDA_PIECE_POSITION_INFO_H
#define EMP_CUDA_PIECE_POSITION_INFO_H

#ifndef __NVCC__
#error This code should be compiled with nvcc
#endif // __NVCC__

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
     * |   NORTH  ||   EAST   ||   SOUTH  ||   WEST   |
     * -----------||----------||----------||----------|
     * | W0 .. W7 || W0 .. W7 || W0 .. W7 || W0 .. W7 |
     */
    class CUDA_piece_position_info
    : public CUDA_piece_position_info_base
    {
        friend
        std::ostream & operator<<(std::ostream & p_stream, const CUDA_piece_position_info & p_info);

      public:

        inline
        void clear_bit(unsigned int p_id
                      ,emp_types::t_orientation p_orientation
                      );

        inline
        void set_bit(unsigned int p_index
                    ,emp_types::t_orientation p_orientation
                    );

      private:



    };

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info::clear_bit(unsigned int p_id
                                       ,emp_types::t_orientation p_orientation
                                       )
    {
        assert(p_id < 256);
        CUDA_piece_position_info_base::clear_bit(8 * static_cast<unsigned int>(p_orientation) + p_id / 32, p_id % 32);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_piece_position_info::set_bit(unsigned int p_index
                                     ,emp_types::t_orientation p_orientation
                                     )
    {
        assert(p_index < 256);
        CUDA_piece_position_info_base::set_bit(8 * static_cast<unsigned int>(p_orientation) + p_index / 32, p_index % 32);
    }

    //-------------------------------------------------------------------------
    inline
    std::ostream & operator<<(std::ostream & p_stream
                             ,const CUDA_piece_position_info & p_info
                             )
    {
        for(auto l_orientation_index = static_cast<unsigned int>(emp_types::t_orientation::NORTH);
            l_orientation_index <= static_cast<unsigned int>(emp_types::t_orientation::WEST);
            ++l_orientation_index
           )
        {
            p_stream << "|";
            for (unsigned int l_index = 0; l_index < 8; ++l_index)
            {
                p_stream << " " << emp_types::orientation2short_string(static_cast<emp_types::t_orientation>(l_orientation_index)) << "  " << std::setw(3) << (32 * (l_index + 1) - 1) << "-" << std::setw(3) << (32 * l_index) << " |";
            }
            p_stream << std::endl;
            p_stream << "|";
            for (unsigned int l_index = 0; l_index < 8; ++l_index)
            {
                p_stream << " 0x" << std::hex << std::setfill('0') << std::setw(8) << p_info.get_word(8 * l_orientation_index + l_index) << std::dec << " |";
            }
            p_stream << std::setfill(' ') << std::endl;
        }
        return p_stream;
    }

}
#endif //EMP_CUDA_piece_position_info_H
// EOF
