/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2024  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H

#include "CUDA_common_struct_glutton.h"

namespace edge_matching_puzzle
{
    class CUDA_glutton_situation: public CUDA_common_struct_glutton
    {
    public:

        inline explicit
        CUDA_glutton_situation(uint32_t p_level
                              ,uint32_t p_puzzle_size
                              );

#ifdef STRICT_CHECKING
        [[nodiscard]]
        inline
        uint32_t
        get_level() const;
#endif // STRICT_CHECKING

        //-------------------------------------------------------------------------
        [[nodiscard]]
        inline
        __device__ __host__
        bool
        is_position_free(position_index_t p_position_index) const;

    private:

    };

    //-------------------------------------------------------------------------
    CUDA_glutton_situation::CUDA_glutton_situation(uint32_t p_level
                                                  ,uint32_t p_puzzle_size
                                                  )
    :CUDA_common_struct_glutton(p_puzzle_size - p_level, p_level, p_puzzle_size, p_puzzle_size - p_level)
    {
    }

    //-------------------------------------------------------------------------
#ifdef STRICT_CHECKING
    uint32_t
    CUDA_glutton_situation::get_level() const
    {
        return CUDA_common_struct_glutton::get_nb_played_info();
    }
#endif // STRICT_CHECKING

    //-------------------------------------------------------------------------
    __device__ __host__
    bool
    CUDA_glutton_situation::is_position_free(position_index_t p_position_index) const
    {
        return CUDA_common_struct_glutton::_is_position_free(p_position_index);
    }
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H
//EOF