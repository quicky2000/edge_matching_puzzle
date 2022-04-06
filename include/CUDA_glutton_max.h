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

#include "feature_sys_equa_CUDA_base.h"
#include "emp_strategy_generator_factory.h"
#include "quicky_exception.h"
#include "situation_capability.h"
#include "transition_manager.h"
#include <map>

/**
 * This file declare functions that will be implemented for
 * CUDA: performance. Corresponding implementation is in CUDA_glutton_max.cu
 * CPU: alternative implementation to debug algorithm. Corresponding implementation is in CUDA_glutton_max.cpp
 */
namespace edge_matching_puzzle
{
    class emp_piece_db;
    class emp_FSM_info;

    class CUDA_glutton_max: public feature_sys_equa_CUDA_base
    {

      public:

        inline
        CUDA_glutton_max(const emp_piece_db & p_piece_db
                        ,const emp_FSM_info & p_info
                        ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                        ):
        feature_sys_equa_CUDA_base(p_piece_db, p_info, p_strategy_generator, "")
        {

        }

        template<unsigned int NB_PIECES>
        void template_run()
        {
            situation_capability<2 * NB_PIECES> l_situation_capability;
            std::map<unsigned int, unsigned int> l_variable_translator;
            std::unique_ptr<const transition_manager<NB_PIECES>> l_transition_manager{prepare_run<NB_PIECES>(l_situation_capability, l_variable_translator)};

            std::cout << l_situation_capability << std::endl;

            throw quicky_exception::quicky_logic_exception("You must enable CUDA core for this feature", __LINE__, __FILE__);
        }

    };

    /**
     * Launch CUDA kernels
     */
    void launch_CUDA_glutton_max(const emp_piece_db & p_piece_db
                                ,const emp_FSM_info & p_info
                                );

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_H
// EOF
