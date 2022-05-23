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

/**
 * CPU alternative implementation to debug algorithm.
 * Corresponding CUDA implementation is in CUDA_glutton_max.cu
 */
#ifndef ENABLE_CUDA_CODE

#include "CUDA_glutton_max.h"
#include "emp_FSM_info.h"

namespace edge_matching_puzzle
{

    void launch_CUDA_glutton_max(const emp_piece_db & p_piece_db
                                ,const emp_FSM_info & p_info
                                )
    {
        std::unique_ptr<emp_strategy_generator> l_strategy_generator(emp_strategy_generator_factory::create("basic", p_info));
        CUDA_glutton_max l_glutton_max{p_piece_db, p_info, l_strategy_generator};

        switch(p_info.get_nb_pieces())
        {
            case 9:
                l_glutton_max.template_run<9>();
                break;
            case 16:
                l_glutton_max.template_run<16>();
                break;
            case 25:
                l_glutton_max.template_run<25>();
                break;
            case 36:
                l_glutton_max.template_run<36>();
                break;
            case 72:
                l_glutton_max.template_run<72>();
                break;
            case 256:
                l_glutton_max.template_run<256>();
                break;
            default:
                throw quicky_exception::quicky_logic_exception("Unsupported size " + std::to_string(p_info.get_width()) + "x" + std::to_string(p_info.get_height()), __LINE__, __FILE__);
        }
    }

}
#endif // ENABLE_CUDA_CODE
// EOF