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

/**
 * CPU alternative implementation to debug algorithm.
 * Corresponding CUDA implementation is in CUDA_glutton_wide.cu
 */
#ifndef ENABLE_CUDA_CODE
#include "CUDA_glutton_wide.h"

namespace edge_matching_puzzle
{
    void
    CUDA_glutton_wide::run()
    {
        prepare_constants();
        std::unique_ptr<CUDA_color_constraints> l_color_constraints = prepare_color_constraints();
        emp_situation l_start_situation;
        auto l_situation = prepare_situation(this->get_piece_db(), this->get_info(), l_start_situation);
#ifdef STRICT_CHECKING
        std::cout << *l_situation << std::endl;
#endif // STRICT_CHECKING
    }

    void launch_CUDA_glutton_wide(const emp_piece_db & p_piece_db
                                 ,const emp_FSM_info & p_info
                                 )
    {
        CUDA_glutton_wide l_glutton_wide(p_piece_db, p_info);
        l_glutton_wide.run();
    }

}
#endif // ENABLE_CUDA_CODE

// EOF