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

#include "CUDA_glutton_wide.h"
#include "emp_FSM_info.h"
#include "emp_piece_db.h"

namespace edge_matching_puzzle
{

    //-------------------------------------------------------------------------
    void launch_CUDA_glutton_wide(const emp_piece_db & p_piece_db
                                 ,const emp_FSM_info & p_info
                                 )
    {
        CUDA_glutton_wide::prepare_constants(p_piece_db, p_info);
        std::unique_ptr<CUDA_color_constraints> l_color_constraints = CUDA_glutton_wide::prepare_color_constraints(p_piece_db, p_info);
        emp_situation l_start_situation;
        auto l_situation = CUDA_glutton_wide::prepare_situation(p_piece_db, p_info, l_start_situation);
#ifdef STRICT_CHECKING
        std::cout << *l_situation << std::endl;
#endif // STRICT_CHECKING
    }
}
// EOF
