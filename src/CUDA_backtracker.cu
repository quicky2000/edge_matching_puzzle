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

#include "feature_CUDA_backtracker.h"
#include "CUDA_backtracker_stack.h"
#include "my_cuda.h"
#include "transition_manager.h"
#include <chrono>

namespace edge_matching_puzzle
{
    template <unsigned int NB_PIECES>
    __device__
    bool compute_piece_info( unsigned int p_start_index
                           , unsigned int p_end_index
                           , const situation_capability<NB_PIECES * 2> & p_situation_capability
                           , const situation_capability<NB_PIECES * 2> & p_transition_capability
                           , situation_capability<NB_PIECES * 2> & p_result_capability
                           )
    {
        bool l_valid = true;
        for(unsigned int l_piece_index = p_start_index; l_piece_index < p_end_index; ++l_piece_index)
        {
            uint32_t l_thread_situation_capability = p_situation_capability.get_capability(NB_PIECES + l_piece_index).get_word(threadIdx.x);
            // Apply mask only if available piece
            if(__any_sync(0xFFFFFFFF, l_thread_situation_capability))
            {
                uint32_t l_thread_transition_capability = p_transition_capability.get_capability(NB_PIECES + l_piece_index).get_word(threadIdx.x);
                uint32_t l_thread_result_capability = l_thread_situation_capability & l_thread_transition_capability;
                // Check result of mask except for selected piece and current poistion
                if(__any_sync(0xFFFFFFFF, l_thread_result_capability))
                {
                    p_result_capability.get_capability(NB_PIECES + l_piece_index).set_word(threadIdx.x, l_thread_result_capability);
                }
                else // Exit if position is locked
                {
                    l_valid = false;
                    break;
                }
            }

        }
        return l_valid;
    }

    // threadIdx.x : position in the warp
    // ThreadIdx.y : position index
    // blockIdx
    template <unsigned int NB_PIECES>
    __global__
    void kernel_and_info( CUDA_backtracker_stack<NB_PIECES> * p_stacks
                        , const transition_manager<NB_PIECES> & p_transition_capability
                        , unsigned int p_nb_stack
                        )
    {
        assert(warpSize == blockDim.x);
        unsigned int l_stack_index = threadIdx.y + blockIdx.x * blockDim.y;

        if(l_stack_index >= p_nb_stack)
        {
            return;
        }

        CUDA_backtracker_stack<NB_PIECES> & l_stack = p_stacks[l_stack_index];

        // Indicate the level in stack ( ie number of pieces placed)
        unsigned int l_stack_step = 0;

        do
        {
            // Search for first bit set
            uint32_t l_thread_available_variables = l_stack.get_available_variables(l_stack_step).get_capability(l_stack_step).get_word(threadIdx.x);

            // Sync between threads to determine whose as some available variables
            unsigned l_ballot_result = __ballot_sync(0xFFFFFFFF, (int) l_thread_available_variables);

            // Occur in case no more variable available
            if(!l_ballot_result)
            {
                --l_stack_step;
                continue;
            }

            // Determine first lane having an available variable
            unsigned l_elected_thread = __ffs((int)l_ballot_result) - 1;

            // Elected thread shared the available variable word
            l_thread_available_variables = __shfl_sync(0xFFFFFFFF, l_thread_available_variables, (int)l_elected_thread);

            // Variable Id bitfied
            // || Stack step ( 8 bits) || Elected thread ( 5 bits ) || selected bit ( 5 bits) ||
            // Selected bit is greater than 0 due to the vote between threads
            unsigned int l_bit_index = __ffs(l_thread_available_variables) - 1;
            uint32_t l_variable_id = (l_stack_step << 10u) | (l_elected_thread << 5u) | l_bit_index;

            unsigned int l_piece_id = (l_elected_thread % 8) * 32 + l_bit_index;

            // Only elected thread store selected variable
            if(threadIdx.x == l_elected_thread)
            {
                l_stack.set_variable_index(l_stack_step, l_variable_id);
                // Mark piece as used
                l_thread_available_variables &= ~(1u << l_bit_index);
                l_stack.get_available_variables(l_stack_step).get_capability(l_stack_step).set_word(threadIdx.x, l_thread_available_variables);
            }

            if(l_stack_step == NB_PIECES - 1)
            {
                break;
            }
            // Apply vector related to variable_id
            const situation_capability<NB_PIECES * 2> & l_situation_capability = l_stack.get_available_variables(l_stack_step);
            const situation_capability<NB_PIECES * 2> & l_transition_capability = p_transition_capability.get_transition(l_variable_id);
            situation_capability<NB_PIECES * 2> & l_result_capability = l_stack.get_available_variables(l_stack_step + 1);

            bool l_situation_valid = true;
            // Compute new positions after current one as we now that current one will be all 0
            for (unsigned int l_position_piece_index = l_stack_step + 1; l_position_piece_index < NB_PIECES; ++l_position_piece_index)
            {
                uint32_t l_thread_situation_capability = l_situation_capability.get_capability(l_position_piece_index).get_word(threadIdx.x);
                uint32_t l_thread_transition_capability = l_transition_capability.get_capability(l_position_piece_index).get_word(threadIdx.x);
                uint32_t l_thread_result_capability = l_thread_situation_capability & l_thread_transition_capability;

                // Check result of mask except for selected piece and current poistion
                if(__any_sync(0xFFFFFFFF, l_thread_result_capability))
                {
                    l_result_capability.get_capability(l_position_piece_index).set_word(threadIdx.x, l_thread_result_capability);
                }
                else // Exit if position is locked
                {
                    l_situation_valid = false;
                    break;
                }
            }

            // Compute available locations for pieces whose id is less than selected piece
            if(l_situation_valid)
            {
                l_situation_valid = compute_piece_info<NB_PIECES>(0, l_piece_id, l_transition_capability, l_transition_capability,l_result_capability);
            }

            // Compute available locations for pieces whose id is greater than selected piece
            if(l_situation_valid)
            {
                l_situation_valid = compute_piece_info<NB_PIECES>(l_piece_id + 1, NB_PIECES, l_transition_capability, l_transition_capability,l_result_capability);
            }

            if(l_situation_valid)
            {
                l_result_capability.get_capability(NB_PIECES + l_piece_id).set_word(threadIdx.x, 0x0);
                ++l_stack_step;
            }
        }
        while(l_stack_step < NB_PIECES);
    }

    //-------------------------------------------------------------------------
    unsigned int compute_raw_variable_id( unsigned int p_x
            , unsigned int p_y
            , unsigned int p_piece_index
            , emp_types::t_orientation p_orientation
            , const emp_FSM_info & p_info
                                        )
    {
        unsigned int l_nb_pieces = p_info.get_width() * p_info.get_height();
        unsigned int l_position_index = p_x + p_y * p_info.get_width();
        return l_position_index * l_nb_pieces * 4 + p_piece_index * 4 + (unsigned int)p_orientation;
    }

    //-------------------------------------------------------------------------
    unsigned int compute_raw_variable_id( const simplex_variable & p_var
                                        , const emp_FSM_info & p_info
                                        )
    {
        return compute_raw_variable_id(p_var.get_x(), p_var.get_y(), p_var.get_piece_id() - 1, p_var.get_orientation(), p_info);
    }

    //-------------------------------------------------------------------------
    void launch( const emp_piece_db & p_piece_db
               , const emp_FSM_info & p_info
               , const emp_variable_generator & p_variable_generator
               , const emp_strategy_generator & p_strategy_generator
               )
    {
        unsigned int l_raw_variable_nb = p_info.get_width() * p_info.get_height() * p_info.get_width() * p_info.get_height() * 4;
        unsigned int l_variable_nb = p_piece_db.get_nb_pieces(emp_types::t_kind::CORNER) * p_piece_db.get_nb_pieces(emp_types::t_kind::CORNER);
        l_variable_nb += p_piece_db.get_nb_pieces(emp_types::t_kind::BORDER) * p_piece_db.get_nb_pieces(emp_types::t_kind::BORDER);
        l_variable_nb += p_piece_db.get_nb_pieces(emp_types::t_kind::CENTER) * p_piece_db.get_nb_pieces(emp_types::t_kind::CENTER) * 4;

        std::cout << "Max variables nb: " << l_raw_variable_nb << std::endl;
        std::cout << "Nb variables: " << l_variable_nb << std::endl;
        unsigned int l_nb_stack = 1;
        auto * l_stacks = new CUDA_backtracker_stack<256>[l_nb_stack];

	    // Init first step of stack
	    for(unsigned int l_stack_index = 0; l_stack_index < l_nb_stack; ++l_stack_index)
	    {
	        for(auto l_var_iter: p_variable_generator.get_variables())
	        {
		        std::cout << "Variable : " << *l_var_iter << std::endl;
	            std::cout << "Before " << l_stacks[l_stack_index].get_available_variables(0) << std::endl;
                unsigned int l_position_index = p_strategy_generator.get_position_index(l_var_iter->get_x(), l_var_iter->get_y());
                l_stacks[l_stack_index].get_available_variables(0).get_capability(l_position_index).set_bit(l_var_iter->get_piece_id() - 1, l_var_iter->get_orientation());
                l_stacks[l_stack_index].get_available_variables(0).get_capability(256 + l_var_iter->get_piece_id() - 1).set_bit(l_position_index, l_var_iter->get_orientation());
	            std::cout << "After " << l_stacks[l_stack_index].get_available_variables(0) << std::endl;
	        }
	    }

        // Allocate an array to do the link between theorical variables and real variables
        auto l_transition_manager = new transition_manager<256>(l_raw_variable_nb);

        // Allocate transition vectors for real variables
        // All transition vector bits are set to 1 by default as they will be
        // used with and operator
        piece_position_info::set_init_value(std::numeric_limits<uint32_t>::max());
        for(auto l_var_iter: p_variable_generator.get_variables())
        {
            unsigned int l_raw_variable_id = compute_raw_variable_id(*l_var_iter, p_info);
            l_transition_manager->create_transition(l_raw_variable_id);

            unsigned int l_position_index = p_strategy_generator.get_position_index(l_var_iter->get_x(), l_var_iter->get_y());

            // Mask bits corresponding to other variables with same position
            l_transition_manager->get_transition(l_raw_variable_id).get_capability(l_position_index).clear();

            // Mask bits corresponding to other variables with same piece id
            l_transition_manager->get_transition(l_raw_variable_id).get_capability(256 + l_var_iter->get_piece_id() - 1).clear();
        }

        auto l_lamda = [=, & l_transition_manager, & p_strategy_generator, & p_info]( const simplex_variable & p_var1
                                                                                    , const simplex_variable & p_var2
                                                                                    )
        {
            unsigned int l_raw_id1 = compute_raw_variable_id(p_var1, p_info);
            unsigned int l_position_index2 = p_strategy_generator.get_position_index(p_var2.get_x(), p_var2.get_y());
            l_transition_manager->get_transition(l_raw_id1).get_capability(l_position_index2).clear_bit(p_var2.get_piece_id() - 1, p_var2.get_orientation());
            l_transition_manager->get_transition(l_raw_id1).get_capability(256 + p_var2.get_piece_id() - 1).clear_bit(l_position_index2, p_var2.get_orientation());
            unsigned int l_position_index1 = p_strategy_generator.get_position_index(p_var1.get_x(), p_var1.get_y());
            unsigned int l_raw_id2 = compute_raw_variable_id(p_var2, p_info);
            l_transition_manager->get_transition(l_raw_id2).get_capability(l_position_index1).clear_bit(p_var1.get_piece_id() - 1, p_var1.get_orientation());
            l_transition_manager->get_transition(l_raw_id2).get_capability(256 + p_var1.get_piece_id() - 1).clear_bit(l_position_index1, p_var1.get_orientation());
        };

        // Mask variables due to incompatible borders
        p_variable_generator.treat_piece_relations(l_lamda);

        dim3 l_block_info(32, 1);
        dim3 l_grid_info(1);

        auto l_start = std::chrono::steady_clock::now();

        // Reset CUDA error status
        cudaGetLastError();
        std::cout << "Launch kernels" << std::endl;
        kernel_and_info<256><<<l_grid_info,l_block_info>>>(l_stacks, *l_transition_manager, l_nb_stack);
        cudaDeviceSynchronize();
        gpuErrChk(cudaGetLastError());

        auto l_run_end = std::chrono::steady_clock::now();

        delete l_transition_manager;

        auto l_total_end = std::chrono::steady_clock::now();
        auto l_elapsed_seconds = l_total_end - l_start;
        std::cout << "Total elapsed time: " << l_elapsed_seconds.count() << "s" << std::endl;
    }

}
// EOF
