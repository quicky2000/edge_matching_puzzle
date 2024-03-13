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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_WIDE_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_WIDE_H

/**
 * This file declare functions that will be implemented for
 * CUDA: performance. Corresponding implementation is in CUDA_glutton_wide.cu
 * CPU: alternative implementation to debug algorithm. Corresponding implementation is in CUDA_glutton_wide.cpp
 */

#include "emp_piece_db.h"
#include "emp_FSM_info.h"

namespace edge_matching_puzzle
{
    class CUDA_glutton_wide
    {
    public:

        inline
        CUDA_glutton_wide(const emp_piece_db & p_piece_db
                         ,const emp_FSM_info & p_info
        )
                :m_piece_db{p_piece_db}
                ,m_info(p_info)
        {

        }

        void
        run();

    private:

        const emp_piece_db & m_piece_db;
        const emp_FSM_info & m_info;

    };

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_WIDE_H
// EOF
