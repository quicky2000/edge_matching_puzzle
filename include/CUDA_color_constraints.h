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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_COLOR_CONSTRAINTS_H
#define EDGE_MATCHING_PUZZLE_CUDA_COLOR_CONSTRAINTS_H

#include "my_cuda.h"
#ifdef ENABLE_CUDA_CODE
#include "CUDA_memory_managed_item.h"
#endif // ENABLE_CUDA_CODE
#include "CUDA_piece_position_info2.h"
#include "common.h"

namespace edge_matching_puzzle
{
    class CUDA_color_constraints
#ifdef ENABLE_CUDA_CODE
    : public my_cuda::CUDA_memory_managed_item
#endif // ENABLE_CUDA_CODE
    {
      public:

        inline explicit
        CUDA_color_constraints(unsigned int p_nb_colors);

        [[nodiscard]]
        inline
        __host__ __device__
        const CUDA_piece_position_info2 & get_info(uint32_t p_color_index
                                                  ,uint32_t p_orientation_index
                                                  )const;

        [[nodiscard]]
        inline
        CUDA_piece_position_info2 & get_info(uint32_t p_color_index
                                            ,uint32_t p_orientation_index
                                            );

        inline
        ~CUDA_color_constraints();

      private:

        CUDA_piece_position_info2 * m_infos;

    };

    //-------------------------------------------------------------------------
    CUDA_color_constraints::CUDA_color_constraints(unsigned int p_nb_colors)
    :m_infos{new CUDA_piece_position_info2[p_nb_colors * 4]}
    {

    }

    //-------------------------------------------------------------------------
    __host__ __device__
    const CUDA_piece_position_info2 &
    CUDA_color_constraints::get_info(uint32_t p_color_index
                                    ,uint32_t p_orientation_index
                                    )const
    {
        return m_infos[p_color_index * 4 + p_orientation_index];
    }

    //-------------------------------------------------------------------------
    CUDA_piece_position_info2 &
    CUDA_color_constraints::get_info(uint32_t p_color_index
                                    ,uint32_t p_orientation_index
                                    )
    {
        return m_infos[p_color_index * 4 + p_orientation_index];
    }

    //-------------------------------------------------------------------------
    CUDA_color_constraints::~CUDA_color_constraints()
    {
        delete[] m_infos;
    }
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_COLOR_CONSTRAINTS_H
// EOF