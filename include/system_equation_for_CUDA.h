/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2020  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_SYSTEM_EQUATION_FOR_CUDA_H
#define EDGE_MATCHING_PUZZLE_SYSTEM_EQUATION_FOR_CUDA_H

#include "emp_FSM_info.h"
#include "emp_variable_generator.h"
#include "emp_strategy_generator.h"

#include <tuple>

namespace edge_matching_puzzle
{
    class system_equation_for_CUDA
    {
      public:

        template <unsigned int NB_PIECES, typename INITIAL, typename TRANSITION_MANAGER>
        static
        std::tuple<INITIAL *, const TRANSITION_MANAGER *>
        prepare_data_structure( unsigned int p_nb_stack
                              , const emp_FSM_info & p_info
                              , const emp_variable_generator & p_variable_generator
                              , const emp_strategy_generator & p_strategy_generator
                              , std::map<unsigned int, unsigned int> & p_variable_translator
                              );

        static inline
        unsigned int compute_raw_variable_id( unsigned int p_x
                                            , unsigned int p_y
                                            , unsigned int p_piece_index
                                            , emp_types::t_orientation p_orientation
                                            , const emp_FSM_info & p_info
                                            );

      private:

        static inline
        unsigned int compute_raw_variable_id( const simplex_variable & p_var
                                            , const emp_FSM_info & p_info
                                            );


    };

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES, typename INITIAL, typename TRANSITION_MANAGER>
    std::tuple<INITIAL *, const TRANSITION_MANAGER *>
    system_equation_for_CUDA::prepare_data_structure( unsigned int p_nb_stack
                                                    , const emp_FSM_info & p_info
                                                    , const emp_variable_generator & p_variable_generator
                                                    , const emp_strategy_generator & p_strategy_generator
                                                    , std::map<unsigned int, unsigned int> & p_variable_translator
                                                    )
    {
        //unsigned int l_raw_variable_nb = p_info.get_width() * p_info.get_height() * p_info.get_width() * p_info.get_height() * 4;
        unsigned int l_raw_variable_nb = NB_PIECES * 256 * 4;

        std::cout << "Max variables nb: " << l_raw_variable_nb << std::endl;
        std::cout << "Nb variables: " << p_variable_generator.get_variables().size() << std::endl;

        INITIAL::info_t::set_init_value(0);
        auto * l_stacks = new INITIAL[p_nb_stack];

        // Init first step of stack
        for(unsigned int l_stack_index = 0; l_stack_index < p_nb_stack; ++l_stack_index)
        {
            for(auto l_var_iter: p_variable_generator.get_variables())
            {
                unsigned int l_position_index = p_strategy_generator.get_position_index(l_var_iter->get_x(), l_var_iter->get_y());
                l_stacks[l_stack_index].get_available_variables(0).get_capability(l_position_index).set_bit(l_var_iter->get_piece_id() - 1, l_var_iter->get_orientation());
                l_stacks[l_stack_index].get_available_variables(0).get_capability(NB_PIECES + l_var_iter->get_piece_id() - 1).set_bit(l_position_index, l_var_iter->get_orientation());
            }
        }

        // Allocate an array to do the link between theorical variables and real variables
        TRANSITION_MANAGER * l_transition_manager{new TRANSITION_MANAGER(l_raw_variable_nb)};

        // Allocate transition vectors for real variables
        // All transition vector bits are set to 1 by default as they will be
        // used with and operator
        INITIAL::info_t::set_init_value(std::numeric_limits<uint32_t>::max());
        for(auto l_var_iter: p_variable_generator.get_variables())
        {
            unsigned int l_raw_variable_id = compute_raw_variable_id(*l_var_iter, p_info);
            p_variable_translator.insert(std::make_pair(l_raw_variable_id, l_var_iter->get_id()));
            std::cout << "Variable[" << l_var_iter->get_id() << ":" << l_raw_variable_id << "] : " << *l_var_iter << std::endl;
            l_transition_manager->create_transition(l_raw_variable_id);

            unsigned int l_position_index = p_strategy_generator.get_position_index(l_var_iter->get_x(), l_var_iter->get_y());

            // Mask bits corresponding to other variables with same position
            for(auto l_other_variable: p_variable_generator.get_position_variables(l_position_index))
            {
                l_transition_manager->get_transition(l_raw_variable_id).get_capability(l_position_index).clear_bit(l_other_variable->get_piece_id() - 1, l_other_variable->get_orientation());
                l_transition_manager->get_transition(l_raw_variable_id).get_capability(NB_PIECES + l_other_variable->get_piece_id() - 1).clear_bit(l_position_index, l_other_variable->get_orientation());
            }

            // Mask bits corresponding to other variables with same piece id
            for(auto l_other_variable: p_variable_generator.get_piece_variables(l_var_iter->get_piece_id()))
            {
                l_transition_manager->get_transition(l_raw_variable_id).get_capability(NB_PIECES + l_other_variable->get_piece_id() - 1).clear_bit(p_strategy_generator.get_position_index(l_other_variable->get_x(), l_other_variable->get_y()), l_other_variable->get_orientation());
                l_transition_manager->get_transition(l_raw_variable_id).get_capability(p_strategy_generator.get_position_index(l_other_variable->get_x(), l_other_variable->get_y())).clear_bit(l_other_variable->get_piece_id() - 1, l_other_variable->get_orientation());
            }
        }
        INITIAL::info_t::set_init_value(0);

        auto l_lamda = [=, & l_transition_manager, & p_strategy_generator, & p_info]( const simplex_variable & p_var1
                                                                                    , const simplex_variable & p_var2
                                                                                    )
        {
            unsigned int l_raw_id1 = compute_raw_variable_id(p_var1, p_info);
            unsigned int l_position_index2 = p_strategy_generator.get_position_index(p_var2.get_x(), p_var2.get_y());
            l_transition_manager->get_transition(l_raw_id1).get_capability(l_position_index2).clear_bit(p_var2.get_piece_id() - 1, p_var2.get_orientation());
            l_transition_manager->get_transition(l_raw_id1).get_capability(NB_PIECES + p_var2.get_piece_id() - 1).clear_bit(l_position_index2, p_var2.get_orientation());
            unsigned int l_position_index1 = p_strategy_generator.get_position_index(p_var1.get_x(), p_var1.get_y());
            unsigned int l_raw_id2 = compute_raw_variable_id(p_var2, p_info);
            l_transition_manager->get_transition(l_raw_id2).get_capability(l_position_index1).clear_bit(p_var1.get_piece_id() - 1, p_var1.get_orientation());
            l_transition_manager->get_transition(l_raw_id2).get_capability(NB_PIECES + p_var1.get_piece_id() - 1).clear_bit(l_position_index1, p_var1.get_orientation());
        };

        // Mask variables due to incompatible borders
        p_variable_generator.treat_piece_relations(l_lamda);

        return std::make_tuple(l_stacks, l_transition_manager);
    }

    //-------------------------------------------------------------------------
    unsigned int system_equation_for_CUDA::compute_raw_variable_id( unsigned int p_x
                                                                  , unsigned int p_y
                                                                  , unsigned int p_piece_index
                                                                  , emp_types::t_orientation p_orientation
                                                                  , const emp_FSM_info & p_info
                                                                  )
    {
        //unsigned int l_nb_pieces = p_info.get_width() * p_info.get_height();
        unsigned int l_nb_pieces = 256;
        unsigned int l_position_index = p_x + p_y * p_info.get_width();
        return l_position_index * l_nb_pieces * 4 + ((unsigned int)p_orientation) * l_nb_pieces + p_piece_index;
    }

    //-------------------------------------------------------------------------
    unsigned int system_equation_for_CUDA::compute_raw_variable_id( const simplex_variable & p_var
                                                                  , const emp_FSM_info & p_info
                                                                  )
    {
        return compute_raw_variable_id(p_var.get_x(), p_var.get_y(), p_var.get_piece_id() - 1, p_var.get_orientation(), p_info);
    }


}
#endif //EDGE_MATCHING_PUZZLE_SYSTEM_EQUATION_FOR_CUDA_H
