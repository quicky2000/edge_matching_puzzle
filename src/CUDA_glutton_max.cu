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

#include "CUDA_glutton_max.h"
#include "emp_FSM_info.h"
#include "emp_piece_db.h"
#include "CUDA_glutton_max_stack.h"
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

    [[maybe_unused]]
    __device__
    void print_best_candidate_info(unsigned int p_indent_level
                                  ,const CUDA_glutton_max_stack & p_stack
                                  )
    {
        print_single(p_indent_level, "Best candidate info:");
        CUDA_glutton_max::print_position_info(p_indent_level, p_stack, &CUDA_glutton_max_stack::get_best_candidate_info);
    }

    //-------------------------------------------------------------------------
    void launch_CUDA_glutton_max(const emp_piece_db & p_piece_db
                                ,const emp_FSM_info & p_info
                                )
    {
        CUDA_glutton_max::prepare_constants(p_piece_db, p_info);
        std::unique_ptr<CUDA_color_constraints> l_color_constraints = CUDA_glutton_max::prepare_color_constraints(p_piece_db, p_info);

        emp_situation l_start_situation;
        std::unique_ptr<CUDA_glutton_max_stack> l_stack = CUDA_glutton_max::prepare_stack(p_piece_db, p_info, l_start_situation);

#ifdef TEST_KERNEL
        std::unique_ptr<CUDA_memory_managed_array<uint32_t>> l_cuda_array{new CUDA_memory_managed_array<uint32_t>(32)};
        for(unsigned int l_index = 0; l_index < 32 ; ++l_index)
        {
            (*l_cuda_array)[l_index] = 0;
        }
#endif // TEST_KERNEL

        // Reset CUDA error status
        cudaGetLastError();
        std::cout << "Launch kernels" << std::endl;
        dim3 l_block_info(32, 1);
        dim3 l_grid_info(1);
        kernel<<<l_grid_info, l_block_info>>>(l_stack.get(), 1, *l_color_constraints);
#ifdef TEST_KERNEL
        test_kernel<<<l_grid_info, l_block_info>>>(l_stack.get(), 1, *l_cuda_array);
#endif // TEST_KERNEL
        cudaDeviceSynchronize();
        gpuErrChk(cudaGetLastError());

#if 0
        if(l_stack->is_empty())
        {
            std::cout << "Empty stack" << std::endl;
        }
        else
        {
            unsigned int l_max_level = l_stack->get_level() - (unsigned int)l_stack->is_full();
            for(unsigned int l_level = 0; l_level <= l_max_level; ++l_level)
            {
                CUDA_glutton_max_stack::played_info_t l_played_info = l_stack->get_played_info(l_level);
                unsigned int l_x = p_info.get_x(static_cast<uint32_t>(CUDA_glutton_max_stack::decode_position_index(l_played_info)));
                unsigned int l_y = p_info.get_y(static_cast<uint32_t>(CUDA_glutton_max_stack::decode_position_index(l_played_info)));
                assert(!l_start_situation.contains_piece(l_x, l_y));
                l_start_situation.set_piece(l_x
                                           ,l_y
                                           ,emp_types::t_oriented_piece{static_cast<emp_types::t_piece_id >(1 + CUDA_glutton_max_stack::decode_piece_index(l_played_info))
                                                                       ,static_cast<emp_types::t_orientation>(CUDA_glutton_max_stack::decode_orientation_index(l_played_info))
                                                                       }
                                           );
            }
            std::cout << "Situation with stack played info:" << std::endl;
            std::cout << situation_string_formatter<emp_situation>::to_string(l_start_situation) << std::endl;
        }
        for(info_index_t l_index{0u}; l_index < l_stack->get_level_nb_info(); ++l_index)
        {
            std::cout << l_stack->get_position_info(l_index) << std::endl;
            //l_stack->push();
        }
#else // 0
        CUDA_glutton_max::display_result(*l_stack, l_start_situation, p_info);
#endif // 0

#ifdef TEST_KERNEL
        std::cout << "CUDA array content" << std::endl;
        for(unsigned int l_index = 0; l_index < 32; ++l_index)
        {
            std::cout << "cuda_array[" << l_index << "] = " << (*l_cuda_array)[l_index] << std::endl;
        }
#endif // TEST_KERNEL
    }


    //-------------------------------------------------------------------------
    __global__
    void test_kernel(CUDA_glutton_max_stack * //p_stacks
                    ,unsigned int //p_nb_stack
                    ,CUDA_memory_managed_array<uint32_t> & p_array
                    )
    {
        assert(warpSize == blockDim.x);

        p_array[threadIdx.x] = 31 -  threadIdx.x;

        print_all(0, "managed_array[%i] = %i", threadIdx.x, p_array[threadIdx.x]);

        for(unsigned int l_piece_index = 0; l_piece_index < g_nb_pieces; ++l_piece_index)
        {
            print_single(1, "Piece[%i]={%i, %i, %i, %i}\n", l_piece_index + 1, g_pieces[l_piece_index][0], g_pieces[l_piece_index][1], g_pieces[l_piece_index][2], g_pieces[l_piece_index][3]);
        }
#if 0
        unsigned int l_stack_index = threadIdx.y + blockIdx.x * blockDim.y;

        if(l_stack_index >= p_nb_stack)
        {
            return;
        }

        CUDA_glutton_max_stack & l_stack = p_stacks[l_stack_index];


        for(unsigned int l_index = 0; l_index < l_stack.get_size(); ++l_index)
        {
            l_stack.get_position_info(info_index_t(l_index));
            l_stack.push(info_index_t(0),position_index_t(0),0,0);
        }
        for(unsigned int l_index = 0; l_index < l_stack.get_size(); ++l_index)
        {
            l_stack.pop();
        }
#endif // 0
    }

}
// EOF
