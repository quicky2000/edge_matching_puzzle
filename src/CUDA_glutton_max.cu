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

#include "emp_FSM_info.h"
#include "emp_piece_db.h"
#include "CUDA_glutton_max_stack.h"
#include "CUDA_color_constraints.h"
#include "CUDA_memory_managed_array.h"
#include "my_cuda.h"
#include "CUDA_common.h"
#include "emp_situation.h"

namespace edge_matching_puzzle
{

    /**
     * Store piece representation.
     * First dimension is piece index ( ie piece id -1 )
     * Second dimension is border orientation
     */
    __constant__ uint32_t g_pieces[256][4];

    /**
     * Return position offset for each orientation
     * NORTH : 0 EAST:1 SOUTH:2 WEST:3
     * Position offset depend on puzzle dimensions
     */
    __constant__ int g_position_offset[4];

    /**
     * Number of pieces remaining to set
     */
    __constant__ unsigned int g_nb_pieces;

    __global__
    void test_kernel(CUDA_glutton_max_stack * p_stacks
                    ,unsigned int p_nb_stack
                    ,CUDA_memory_managed_array<uint32_t> & p_array
                    );

    inline
    __device__
    uint32_t reduce_add_sync(uint32_t p_word)
    {
        unsigned l_mask = 0xFFFF;
        unsigned int l_width = 16;
        do
        {
            p_word += __shfl_down_sync(l_mask, p_word, l_width);
            l_width = l_width >> 1;
            l_mask = l_mask >> l_width;
        }
        while(l_width);
        return __shfl_sync(0xFFFFFFFFu, p_word, 0);
    }

    inline
    __device__
    uint32_t reduce_min_sync(uint32_t p_word)
    {
        unsigned l_mask = 0xFFFF;
        unsigned int l_width = 16;
        do
        {
            uint32_t l_received_word = __shfl_down_sync(l_mask, p_word, l_width);
            p_word = l_received_word < p_word ? l_received_word : p_word;
            l_width = l_width >> 1;
            l_mask = l_mask >> l_width;
        }
        while(l_width);
        return __shfl_sync(0xFFFFFFFFu, p_word, 0);
    }

    inline
    __device__
    uint32_t reduce_max_sync(uint32_t p_word)
    {
        unsigned l_mask = 0xFFFF;
        unsigned int l_width = 16;
        do
        {
            uint32_t l_received_word = __shfl_down_sync(l_mask, p_word, l_width);
            p_word = l_received_word > p_word ? l_received_word : p_word;
            l_width = l_width >> 1;
            l_mask = l_mask >> l_width;
        }
        while(l_width);
        return __shfl_sync(0xFFFFFFFFu, p_word, 0);
    }

    __global__
    void kernel(CUDA_glutton_max_stack * p_stacks
               ,unsigned int p_nb_stack
               )
    {
        assert(warpSize == blockDim.x);

        unsigned int l_stack_index = threadIdx.y + blockIdx.x * blockDim.y;

        if(l_stack_index >= p_nb_stack)
        {
            return;
        }

        CUDA_glutton_max_stack & l_stack = p_stacks[l_stack_index];

        while(l_stack.get_level() < l_stack.get_size())
        {
            // Iterate on all level position information to compute the score of each available transition
            for(unsigned int l_info_index = 0; l_info_index < l_stack.get_nb_position(); ++l_info_index)
            {
                // At the beginning all threads participates to ballot
                unsigned int l_ballot_result = 0xFFFFFFFF;

                // Each thread get its word in position info
                uint32_t l_thread_available_variables = l_stack.get_position_info(l_info_index).get_word(threadIdx.x);

                // Iterate on non null position info words determined by ballot between threads
                do
                {
                    // Sync between threads to determine who as some available variables
                    l_ballot_result = __ballot_sync(l_ballot_result, (int) l_thread_available_variables);

                    // Ballot result cannot be NULL because we are by construction in a valid situation
                    assert(l_ballot_result);

                    // Determine first lane/thread having an available variable. Result is greater than 0 due to assert
                    unsigned l_elected_thread = __ffs((int)l_ballot_result) - 1;

                    // Eliminate thread from next ballot
                    l_ballot_result &= ~(1u << l_elected_thread);

                    // Copy available variables because we will iterate on it
                    uint32_t l_current_available_variables = l_thread_available_variables;

                    // Share current available variables with all other threads so they can select the same variable
                    l_current_available_variables = __shfl_sync(0xFFFFFFFF, l_current_available_variables, (int)l_elected_thread);

                    // Iterate on available variables of elected thread
                    do
                    {
                        // Determine first available variable. Result  cannot be 0 due to ballot
                        unsigned l_bit_index = __ffs((int)l_current_available_variables) - 1;

                        // Set variable bit to zero
                        uint32_t l_mask = ~(1u << l_bit_index);
                        l_current_available_variables &= l_mask;

                        // Compute piece index
                        uint32_t l_piece_index = CUDA_piece_position_info2::compute_piece_index(l_elected_thread, l_bit_index);

                        // Piece orientation
                        uint32_t l_piece_orientation = CUDA_piece_position_info2::compute_orientation_index(l_elected_thread, l_bit_index);

                        for(unsigned int l_orientation_index = 0; l_orientation_index < 4; ++l_orientation_index)
                        {
                            uint32_t l_color_id = g_pieces[l_piece_index][l_color_id];
                            if(l_color_id)
                            {

                            }
                        }
                    }  while(l_current_available_variables);

                } while(l_ballot_result);
            }
        }
    }

    //-------------------------------------------------------------------------
    void launch_CUDA_glutton_max(const emp_piece_db & p_piece_db
                                ,const emp_FSM_info & p_info
                                )
    {
        // Prepare piece description
        std::array<uint32_t, 256 * 4> l_pieces{};
        for(unsigned int l_piece_index = 0; l_piece_index < p_info.get_nb_pieces(); ++l_piece_index)
        {
            for(auto l_orientation: emp_types::get_orientations())
            {
                l_pieces[l_piece_index * 4 + static_cast<unsigned int>(l_orientation)] = p_piece_db.get_piece(l_piece_index + 1).get_color(l_orientation);
            }
        }

        // Prepare position offset
        std::array<int,4> l_x_offset{- static_cast<int>(p_info.get_width()), 1, static_cast<int>(p_info.get_width()), -1};
        unsigned int l_nb_pieces = p_info.get_nb_pieces();

        CUDA_info();

        // Fill constant variables
        cudaMemcpyToSymbol(g_pieces, l_pieces.data(), l_pieces.size() * sizeof(uint32_t ));
        cudaMemcpyToSymbol(g_position_offset, l_x_offset.data(), l_x_offset.size() * sizeof(int));
        cudaMemcpyToSymbol(g_nb_pieces, &l_nb_pieces, sizeof(unsigned int));

        // Prepare color constraints
        CUDA_piece_position_info2::set_init_value(0xFFFFFFFF);
        std::unique_ptr<CUDA_color_constraints> l_color_constraints{new CUDA_color_constraints(static_cast<unsigned int>(p_piece_db.get_colors().size()))};
        for(auto l_iter_color: p_piece_db.get_colors())
        {
            unsigned int l_color_index = l_iter_color - 1;
            for(auto l_color_orientation: emp_types::get_orientations())
            {
                auto l_opposite_orientation = emp_types::get_opposite(l_color_orientation);
                for(unsigned int l_piece_index = 0; l_piece_index < p_info.get_nb_pieces(); ++l_piece_index)
                {
                    for(auto l_piece_orientation: emp_types::get_orientations())
                    {
                        emp_types::t_color_id l_color_id{p_piece_db.get_piece(l_piece_index + 1).get_color(l_opposite_orientation, l_piece_orientation)};
                        if(l_color_id != l_iter_color)
                        {
                            l_color_constraints->get_info(l_color_index, static_cast<unsigned int>(l_color_orientation)).clear_bit(l_piece_index, l_piece_orientation);
                        }
                    }
                }
            }
        }

        // Prepare initial situation vector
        CUDA_piece_position_info2::set_init_value(0x0);
        auto * l_initial_capability = new CUDA_piece_position_info2[p_info.get_nb_pieces()];
        for(unsigned int l_position_index = 0; l_position_index < p_info.get_nb_pieces(); ++l_position_index)
        {
            switch(p_info.get_position_kind(p_info.get_x(l_position_index), p_info.get_y(l_position_index)))
            {
                case emp_types::t_kind::CORNER:
                {
                    emp_types::t_orientation l_border1;
                    emp_types::t_orientation l_border2;
                    std::tie(l_border1,l_border2) = p_info.get_corner_orientation(l_position_index);
                    for (unsigned int l_corner_index = 0; l_corner_index < 4; ++l_corner_index)
                    {
                        const emp_piece_corner & l_corner = p_piece_db.get_corner(l_corner_index);
                        l_initial_capability[l_position_index].set_bit(l_corner.get_id() - 1, l_corner.compute_orientation(l_border1, l_border2));
                    }
                }
                    break;
                case emp_types::t_kind::BORDER:
                {
                    emp_types::t_orientation l_border_orientation = p_info.get_border_orientation(l_position_index);
                    for(unsigned int l_border_index = 0; l_border_index < p_info.get_nb_borders(); ++l_border_index)
                    {
                        const emp_piece_border & l_border = p_piece_db.get_border(l_border_index);
                        l_initial_capability[l_position_index].set_bit(l_border.get_id() - 1, l_border.compute_orientation(l_border_orientation));
                    }
                }
                    break;
                case emp_types::t_kind::CENTER:
                    for(unsigned int l_center_index = 0; l_center_index < p_info.get_nb_centers(); ++l_center_index)
                    {
                        const emp_piece & l_center = p_piece_db.get_center(l_center_index);
                        for (auto l_iter: emp_types::get_orientations())
                        {
                            l_initial_capability[l_position_index].set_bit(l_center.get_id() - 1, l_iter);
                        }
                    }
                    break;
                case emp_types::t_kind::UNDEFINED:
                    throw quicky_exception::quicky_logic_exception("Undefined position type", __LINE__, __FILE__);
                default:
                    throw quicky_exception::quicky_logic_exception("Unknown position type", __LINE__, __FILE__);
            }
        }

        CUDA_memory_managed_array<uint32_t> l_cuda_array(32);
        for(unsigned int l_index = 0; l_index < 32 ; ++l_index)
        {
            l_cuda_array[l_index] = 0;
        }

        emp_situation l_start_situation;

        unsigned int l_size = l_nb_pieces - l_start_situation.get_level();
        CUDA_glutton_max_stack l_stack(l_size,l_nb_pieces);

        // Prepare stack with info of initial situation
        uint32_t l_info_index = 0;
        for(unsigned int l_position_index = 0; l_position_index < l_nb_pieces; ++l_position_index)
        {
            if(!l_start_situation.contains_piece(p_info.get_x(l_position_index), p_info.get_y(l_position_index)))
            {
                l_stack.set_position_index(l_info_index, l_position_index);
                l_stack.set_position_info(l_info_index, l_initial_capability[l_position_index]);
                ++l_info_index;
            }
        }


        // Reset CUDA error status
        cudaGetLastError();
        std::cout << "Launch kernels" << std::endl;
        dim3 l_block_info(32, 1);
        dim3 l_grid_info(1);
        test_kernel<<<l_grid_info, l_block_info>>>(&l_stack, 1, l_cuda_array);
        cudaDeviceSynchronize();
        gpuErrChk(cudaGetLastError());

        for(unsigned int l_index = 0; l_index < l_size; ++l_index)
        {
            std::cout << l_stack.get_position_info(l_index) << std::endl;
            l_stack.push();
        }

        std::cout << "CUDA array content" << std::endl;
        for(unsigned int l_index = 0; l_index < 32; ++l_index)
        {
            std::cout << "cuda_array[" << l_index << "] = " << l_cuda_array[l_index] << std::endl;
        }

    }

    //-------------------------------------------------------------------------
    __global__
    void test_kernel(CUDA_glutton_max_stack * p_stacks
                    ,unsigned int p_nb_stack
                    ,CUDA_memory_managed_array<uint32_t> & p_array
                    )
    {
        assert(warpSize == blockDim.x);

        p_array[threadIdx.x] = 31 -  threadIdx.x;

        return;
        if(!threadIdx.x)
        {
            for(unsigned int l_piece_index = 0; l_piece_index < g_nb_pieces; ++l_piece_index)
            {
                printf("Piece[%i]={%i, %i, %i, %i}\n", l_piece_index + 1, g_pieces[l_piece_index][0], g_pieces[l_piece_index][1], g_pieces[l_piece_index][2], g_pieces[l_piece_index][3]);
            }
        }
        return;
        unsigned int l_stack_index = threadIdx.y + blockIdx.x * blockDim.y;

        if(l_stack_index >= p_nb_stack)
        {
            return;
        }

        CUDA_glutton_max_stack & l_stack = p_stacks[l_stack_index];


        for(unsigned int l_index = 0; l_index < l_stack.get_size(); ++l_index)
        {
            l_stack.get_position_info(l_index);
            l_stack.push();
        }
        for(unsigned int l_index = 0; l_index < l_stack.get_size(); ++l_index)
        {
            l_stack.pop();
        }
    }

}
// EOF
