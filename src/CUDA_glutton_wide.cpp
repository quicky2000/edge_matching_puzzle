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
        auto l_situations = prepare_situation(this->get_piece_db(), this->get_info(), l_start_situation);
        CUDA_glutton_situation l_glutton_situation{*l_situations, 0};
        std::cout << l_glutton_situation << std::endl;

        unsigned int l_corner_index = this->get_info().get_position_index(0, 0);
        auto l_corner_info_index = l_situations->get_info_index(0, position_index_t{l_corner_index});
        auto l_corner_info = l_situations->get_position_info(0, l_corner_info_index);
        std::cout << "Corner info : " << std::endl;
        std::cout << l_corner_info << std::endl;

#if __cplusplus >= 201402
        auto l_next_situations = std::make_unique<CUDA_glutton_situations>(1, this->get_info().get_nb_pieces(), 4);
#else // __cplusplus
        auto l_next_situations = std::unique_ptr<CUDA_glutton_situations>(new CUDA_glutton_situations(1, this->get_info().get_nb_pieces(), 4));
#endif // __cplusplus

        uint32_t l_ffs = 0;
        uint32_t l_situation_index = 0;
        while((l_ffs = ffs(l_corner_info, l_ffs)))
        {
            std::cout << l_ffs << std::endl;
            l_next_situations->play_from(l_situation_index
                                        ,*l_situations
                                        ,l_situation_index
                                        ,l_corner_info_index
                                        ,CUDA_piece_position_info2::compute_piece_index(l_ffs - 1)
                                        ,CUDA_piece_position_info2::compute_piece_orientation(l_ffs - 1)
                                        ,*l_color_constraints
                                        );
        CUDA_glutton_situation l_result_situation{*l_next_situations, l_situation_index};
        std::cout << l_result_situation << std::endl;
        }

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