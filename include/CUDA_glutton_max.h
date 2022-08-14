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
#include "situation_string_formatter.h"
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
            std::unique_ptr<std::array<situation_capability<2 * NB_PIECES>, NB_PIECES + 1>> l_situation_capabilities_ptr{new std::array<situation_capability<2 * NB_PIECES>, NB_PIECES + 1>()};
            std::array<situation_capability<2 * NB_PIECES>, NB_PIECES + 1> & l_situation_capabilities = *l_situation_capabilities_ptr;
            std::map<unsigned int, unsigned int> l_variable_translator;
            std::unique_ptr<const transition_manager<NB_PIECES>> l_transition_manager{prepare_run<NB_PIECES>(l_situation_capabilities[0], l_variable_translator)};

            std::cout << l_situation_capabilities[0] << std::endl;

            emp_situation l_situation[NB_PIECES + 1];

            unsigned int l_level = 0;
            while(l_level < get_info().get_nb_pieces())
            {
                std::cout << "Level : " << l_level << "\t" << situation_string_formatter<emp_situation>::to_string(l_situation[l_level]) << std::endl;
                unsigned int l_min_total = 0;
                unsigned int l_min_x;
                unsigned int l_min_y;
                unsigned int l_min_piece_index;
                emp_types::t_orientation l_min_orientation;
                for (unsigned int l_position_index = 0; l_position_index < get_info().get_nb_pieces(); ++l_position_index)
                {
                    unsigned int l_x, l_y;
                    std::tie(l_x,l_y) = get_strategy_generator().get_position(l_position_index);
                    piece_position_info & l_position_info = l_situation_capabilities[l_level].get_capability(l_position_index);
                    for (unsigned int l_word_index = 0; l_word_index < 32; ++l_word_index)
                    {
                        uint32_t l_word = l_position_info.get_word(l_word_index);
                        while (l_word)
                        {
                            unsigned int l_bit_index = ffs(l_word) - 1;
                            l_word &= ~(1u << l_bit_index);
                            unsigned int l_piece_index;
                            emp_types::t_orientation l_orientation;
                            std::tie(l_piece_index,l_orientation) = piece_position_info::convert(l_word_index, l_bit_index);
#ifdef VERBOSE
                            std::cout << "(" << l_x << "," << l_y << ") => Piece index: " << l_piece_index << "\tOrientation: " << emp_types::orientation2short_string(l_orientation) << std::endl;
#endif // VERBOSE
                            situation_capability<2 * NB_PIECES> l_new_situation_capability;
                            emp_situation l_new_situation{l_situation[l_level]};
                            set_piece(l_new_situation, l_situation_capabilities[l_level], l_x, l_y, l_piece_index, l_orientation, l_new_situation_capability, *l_transition_manager);
                            situation_profile l_profile = l_new_situation_capability.compute_profile(l_level + 1);
                            if (l_profile.is_valid())
                            {
                                unsigned int l_total = l_profile.compute_total();
#ifdef VERBOSE
                                std::cout << situation_string_formatter<emp_situation>::to_string(l_new_situation) << " : " << l_total << std::endl;
#endif // VERBOSE
                                if (l_min_total <= l_total)
                                {
                                    l_min_total = l_total;
                                    l_min_x = l_x;
                                    l_min_y = l_y;
                                    l_min_orientation = l_orientation;
                                    l_min_piece_index = l_piece_index;
                                }
                            }
                            else
                            {
                                l_position_info.clear_bit(l_piece_index,l_orientation);
                            }
                        }
                    }
                }
                if (l_level + 1 < get_info().get_nb_pieces() && !l_min_total)
                {
                    if(l_level)
                    {
                        --l_level;
                    }
                    else
                    {
                        std::cout << "Unable to find a solution" << std::endl;
                        std::exit(-1);
                    }
                }
                else
                {
                    ++l_level;
                    l_situation[l_level] = l_situation[l_level - 1];
                    l_situation_capabilities[l_level - 1].get_capability(get_strategy_generator().get_position_index(l_min_x, l_min_y)).clear_bit(l_min_piece_index, l_min_orientation);
                    set_piece(l_situation[l_level], l_situation_capabilities[l_level - 1], l_min_x, l_min_y, l_min_piece_index, l_min_orientation, l_situation_capabilities[l_level], *l_transition_manager);
                }
            }
            std::cout << "Solution : " << situation_string_formatter<emp_situation>::to_string(l_situation[NB_PIECES]) << std::endl;
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
