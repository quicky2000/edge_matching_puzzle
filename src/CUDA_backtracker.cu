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

#include "CUDA_backtracker_stack.h"
#include "CUDA_transition_manager.h"
#include "system_equation_for_CUDA.h"
#include "my_cuda.h"
#include <chrono>
#include "thrust/version.h"

namespace edge_matching_puzzle
{
    template <unsigned int NB_PIECES>
    __device__
    bool compute_piece_info( unsigned int p_start_index
                           , unsigned int p_end_index
                           , const CUDA_situation_capability<NB_PIECES * 2> & p_situation_capability
                           , const CUDA_situation_capability<NB_PIECES * 2> & p_transition_capability
                           , CUDA_situation_capability<NB_PIECES * 2> & p_result_capability
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
                        , const CUDA_transition_manager<NB_PIECES> & p_transition_capability
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
	        printf("Thread %i : selected variable %i at step %i\n", threadIdx.x, l_variable_id, l_stack_step);
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
            const CUDA_situation_capability<NB_PIECES * 2> & l_situation_capability = l_stack.get_available_variables(l_stack_step);
            const CUDA_situation_capability<NB_PIECES * 2> & l_transition_capability = p_transition_capability.get_transition(l_variable_id);
            CUDA_situation_capability<NB_PIECES * 2> & l_result_capability = l_stack.get_available_variables(l_stack_step + 1);

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
    template<unsigned int NB_PIECES>
    void template_launch( const emp_piece_db & p_piece_db
                        , const emp_FSM_info & p_info
                        , const emp_variable_generator & p_variable_generator
                        , const emp_strategy_generator & p_strategy_generator
                        )
    {
        std::cout << "CUDA version  : " << CUDART_VERSION << std::endl;
        std::cout << "THRUST version: " << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << "." << THRUST_SUBMINOR_VERSION << std::endl;

        int l_cuda_device_nb = 0;
        cudaError_t l_cuda_status = cudaGetDeviceCount(&l_cuda_device_nb);
        if(cudaSuccess != l_cuda_status)
        {
            throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);
        }
        std::cout << "Number of CUDA devices: " << l_cuda_device_nb << std::endl;

        for(int l_device_index = 0; l_device_index < l_cuda_device_nb; ++l_device_index)
        {
            std::cout << "Cuda device[" << l_device_index << "]" << std::endl;
            cudaDeviceProp l_properties;
            cudaGetDeviceProperties(&l_properties, l_device_index);
            std::cout << R"(\tName                      : ")" << l_properties.name << R"(")" << std::endl;
            std::cout <<   "\tDevice compute capability : " << l_properties.major << "." << l_properties.minor << std::endl;
            std::cout <<    "\tWarp size                 : " << l_properties.warpSize << std::endl;
            if(l_properties.warpSize != 32)
            {
                throw quicky_exception::quicky_logic_exception("Unsupported warp size" + std::to_string(l_properties.warpSize), __LINE__, __FILE__);
            }
            std::cout <<    "\tMultiprocessor count      : " << l_properties.multiProcessorCount << std::endl;
            std::cout <<    "\tManaged Memory            : " << l_properties.managedMemory << std::endl;
            if(!l_properties.managedMemory)
            {
                throw quicky_exception::quicky_logic_exception("Managed memory is not supported", __LINE__, __FILE__);
            }
            std::cout << std::endl;
        }

        unsigned int l_nb_stack = 1;
        auto * l_stacks = new CUDA_backtracker_stack<NB_PIECES>[l_nb_stack];

        // Init first step of stack
        for(unsigned int l_stack_index = 0; l_stack_index < l_nb_stack; ++l_stack_index)
        {
            system_equation_for_CUDA::prepare_initial<NB_PIECES, CUDA_situation_capability<2 * NB_PIECES>>(p_variable_generator, p_strategy_generator, l_stacks[l_stack_index].get_available_variables(0));
        }


        const CUDA_transition_manager<NB_PIECES> * l_transition_manager;
        std::map<unsigned int, unsigned int> l_variable_translator;
        l_transition_manager = system_equation_for_CUDA::prepare_transitions<NB_PIECES, CUDA_situation_capability<2 * NB_PIECES>, CUDA_transition_manager<NB_PIECES>>(p_info, p_variable_generator, p_strategy_generator, l_variable_translator);

        dim3 l_block_info(32, 1);
        dim3 l_grid_info(1);

        auto l_start = std::chrono::steady_clock::now();

        // Reset CUDA error status
        cudaGetLastError();
        std::cout << "Launch kernels" << std::endl;
        kernel_and_info<NB_PIECES><<<l_grid_info,l_block_info>>>(l_stacks, *l_transition_manager, l_nb_stack);
        cudaDeviceSynchronize();
        gpuErrChk(cudaGetLastError());

        auto l_variables = p_variable_generator.get_variables();
        emp_FSM_situation l_situation;
        l_situation.set_context(*new emp_FSM_context(NB_PIECES));
        for(unsigned int l_step_index = 0; l_step_index < p_info.get_height() * p_info.get_width(); ++l_step_index)
        {
            auto l_iter = l_variable_translator.find(l_stacks[0].get_variable_index(l_step_index));
            assert(l_variable_translator.end() != l_iter);
            simplex_variable & l_variable = *l_variables.at(l_iter->second);
            l_situation.set_piece(l_variable.get_x(), l_variable.get_y(), l_variable.get_oriented_piece());
        }
        std::cout << l_situation.to_string() <<std::endl;
        delete l_transition_manager;
        delete[] l_stacks;

        auto l_total_end = std::chrono::steady_clock::now();
        auto l_elapsed_seconds = l_total_end - l_start;
        std::cout << "Total elapsed time: " << l_elapsed_seconds.count() << "s" << std::endl;
    }
    //-------------------------------------------------------------------------
    void launch( const emp_piece_db & p_piece_db
               , const emp_FSM_info & p_info
               , const emp_variable_generator & p_variable_generator
               , const emp_strategy_generator & p_strategy_generator
               )
    {
        assert(3 == p_info.get_width());
        assert(3 == p_info.get_height());
        template_launch<9>(p_piece_db, p_info, p_variable_generator, p_strategy_generator);
    }

}
// EOF
