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
#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_H

#include "quicky_exception.h"

/**
 * This file declare functions that will be implemented for
 * CUDA: performance. Corresponding implementation is in CUDA_glutton_max.cu
 * CPU: alternative implementation to debug algorithm. Corresponding implementation is in CUDA_glutton_max.cpp
 */
namespace edge_matching_puzzle
{
    class emp_piece_db;
    class emp_FSM_info;

    template<unsigned int NB_PIECES>
    void template_run()
    {
        throw quicky_exception::quicky_logic_exception("You must enable CUDA core for this feature", __LINE__, __FILE__);
    }

    /**
     * Launch CUDA kernels
     */
    void launch_CUDA_glutton_max(const emp_piece_db & p_piece_db
                                ,const emp_FSM_info & p_info
                                );

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_H
// EOF
