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
#include "situation_string_formatter.h"

#define LOG_EXECUTION

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

    __device__
    unsigned int my_strlen(const char * p_string)
    {
        unsigned int l_index = 0;
        while('\0' != p_string[l_index])
        {
            ++l_index;
        }
        return l_index;
    }

    __device__
    unsigned int count_cr(const char * p_string)
    {
        unsigned int l_nb_cr = 0;
        unsigned int l_index = 0;
        while('\0' != p_string[l_index])
        {
            l_nb_cr += ('\n' == p_string[l_index++]);
        }
        return l_nb_cr;
    }

    __device__
    char * prepare_format(unsigned int p_level
                         ,const char * p_header
                         ,const char * p_format
                         )
    {
        unsigned int l_len = my_strlen(p_format);
        unsigned int l_nb_cr = count_cr(p_format);
        bool l_additional_cr = (!l_len) || (p_format[l_len - 1] != '\n');
        unsigned int l_total_cr = l_nb_cr + l_additional_cr;
        unsigned int l_header_len = my_strlen(p_header);
        char * l_format = static_cast<char*>(malloc(l_len + l_additional_cr + l_total_cr * (l_header_len + 2 * p_level)));
        unsigned int l_index = 0;
        unsigned int l_char_index = 0;
        for(unsigned int l_line_index = 0; l_line_index < l_total_cr; ++l_line_index)
        {
            for(unsigned int l_level_index = 0; l_level_index < 2 * p_level; ++l_level_index)
            {
                l_format[l_index++] = ' ';
            }
            for(unsigned int l_header_index = 0; l_header_index < l_header_len; ++l_header_index)
            {
                l_format[l_index++] = l_line_index ? ' ' : p_header[l_header_index];
            }
            while('\0' != p_format[l_char_index] && '\n' != p_format[l_char_index])
            {
                l_format[l_index++] = p_format[l_char_index++];
            }
            ++l_char_index;
            l_format[l_index++] = '\n';
        }
        l_format[l_index++] = '\0';
        return l_format;
    }

    template<typename... Targs>
    __device__
    void print_all(unsigned int p_level
                  ,const char * p_format
                  ,Targs... Fargs
                  )
    {
#ifdef LOG_EXECUTION
        char * l_format = prepare_format(p_level, "Thread%3i : ", p_format);
        printf(l_format, threadIdx.x, Fargs...);
        free(l_format);
#endif // LOG_EXECUTION
    }

    template<typename... Targs>
    __device__
    void print_mask(unsigned int p_level
                   ,uint32_t p_mask
                   ,const char * p_format
                   ,Targs... Fargs
                   )
    {
#ifdef LOG_EXECUTION
        // Do format treatment in all threads to minimise divergence
        char * l_format = prepare_format(p_level, "Thread%3i : ", p_format);
        if((1u << threadIdx.x) & p_mask)
        {
            printf(l_format, threadIdx.x, Fargs...);
        }
        free(l_format);
#endif // LOG_EXECUTION
    }

    template<typename... Targs>
    __device__
    void print_single(unsigned int p_level
                     ,const char * p_format
                     ,Targs... Fargs
                     )
    {
#ifdef LOG_EXECUTION
        char * l_format = prepare_format(p_level, "", p_format);
        if(!threadIdx.x)
        {
            printf(l_format, Fargs...);
        }
        free(l_format);
#endif // LOG_EXECUTION
    }

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

    inline
    __device__
    void update_stats(uint32_t p_value
                     ,uint32_t & p_min
                     ,uint32_t & p_max
                     ,uint32_t & p_total
                     )
    {
        p_max = p_value > p_max ? p_value : p_max;
        p_min = p_value < p_min ? p_value : p_min;
        p_total += p_value;
    }

    inline
    __device__
    bool analyze_info(uint32_t p_capability
                     ,uint32_t p_constraint_capability
                     ,uint32_t & p_min
                     ,uint32_t & p_max
                     ,uint32_t & p_total
                     ,CUDA_glutton_max_stack::t_piece_infos & p_piece_info
                     )
    {
        uint32_t l_result_capability = p_capability & p_constraint_capability;

        // Check result of mask except for selected piece and current position
        if(__any_sync(0xFFFFFFFFu, l_result_capability))
        {
            uint32_t l_info_bits = reduce_add_sync(__popc(l_result_capability));
            update_stats(l_info_bits, p_min, p_max, p_total);
            for(unsigned short & l_piece_index : p_piece_info)
            {
                l_piece_index += static_cast<CUDA_glutton_max_stack::t_piece_info>(__popc(static_cast<int>(l_result_capability & 0xFu)));
                l_result_capability = l_result_capability >> 4;
            }
            return false;
        }
        return true;
    }

    __device__
    void print_position_info(unsigned int p_indent_level
                            ,const CUDA_glutton_max_stack & p_stack
                            ,const CUDA_piece_position_info2 & (CUDA_glutton_max_stack::*p_accessor)(unsigned int) const
                            )
    {
        for(unsigned int l_display_index = 0; l_display_index < (p_stack.get_size() - p_stack.get_level()); ++l_display_index)
        {
            print_all(6,"Info = 0x%" PRIx32, (p_stack.*p_accessor)(l_display_index).get_word(threadIdx.x));
        }
    }
    __device__
    void print_position_info(unsigned int p_indent_level
                            ,const CUDA_glutton_max_stack & p_stack
                            )
    {
        print_position_info(p_indent_level, p_stack, &CUDA_glutton_max_stack::get_position_info);
    }

    __device__
    void print_best_candidate_info(unsigned int p_indent_level
                                  ,const CUDA_glutton_max_stack & p_stack
                                  )
    {
        print_position_info(p_indent_level, p_stack, &CUDA_glutton_max_stack::get_best_candidate_info);
    }

    /**
     * Print information relating info index and position index
     * @param p_indent_level indentation level
     * @param p_stack
     */
    __device__
    void
    print_device_info_position_index(unsigned int p_indent_level
                                    ,const CUDA_glutton_max_stack & p_stack
                                    )
    {
        print_single(p_indent_level, "====== Position index <-> Info index ======\n");
        for(unsigned int l_index = 0; l_index <= (p_stack.get_nb_pieces() / 32); ++l_index)
        {
            unsigned int l_thread_index = 32 * l_index + threadIdx.x;
            print_mask(p_indent_level, __ballot_sync(0xFFFFFFFF, l_thread_index < p_stack.get_nb_pieces()), "Position[%" PRIu32 "] -> Index %" PRIu32 , l_thread_index, l_thread_index < p_stack.get_nb_pieces() ? p_stack.get_info_index(l_thread_index) : 0xDEADCAFE);
        }
        for(unsigned int l_index = 0; l_index <= (p_stack.get_size() / 32); ++l_index)
        {
            unsigned int l_thread_index = 32 * l_index + threadIdx.x;
            print_mask(p_indent_level, __ballot_sync(0xFFFFFFFF, l_thread_index < p_stack.get_size()), "%c Index[%" PRIu32 "] -> Position %" PRIu32 , l_thread_index < p_stack.get_size() - p_stack.get_level() ? '*' : ' ', l_thread_index, l_thread_index < p_stack.get_size() ? p_stack.get_position_index(l_thread_index) : 0xDEADCAFE);
        }
    }

    /**
     * Print information relating info index and position index
     * @param p_indent_level indentation level
     * @param p_stack
     */
    void
    print_host_info_position_index(unsigned int p_indent_level
                                  ,const CUDA_glutton_max_stack & p_stack
                                  )
    {
        std::cout << std::string(p_indent_level,' ') <<  "====== Position index <-> Info index ======" << std::endl;
        for(unsigned int l_index = 0; l_index < p_stack.get_nb_pieces(); ++l_index)
        {
            std::cout << std::string(p_indent_level,' ') << "Position[" << l_index << "] -> Index " << p_stack.get_info_index(l_index) << std::endl;
        }
        for(unsigned int l_index = 0; l_index < p_stack.get_size(); ++l_index)
        {
            std::cout << std::string(p_indent_level,' ') << (l_index < (p_stack.get_size() - p_stack.get_level()) ? '*' : ' ') << " Index[" << l_index << "] -> Position " << p_stack.get_position_index(l_index) << std::endl;
        }
    }

    __global__
    void kernel(CUDA_glutton_max_stack * p_stacks
               ,unsigned int p_nb_stack
               ,const CUDA_color_constraints & p_color_constraints
               )
    {
        assert(warpSize == blockDim.x);

        unsigned int l_stack_index = threadIdx.y + blockIdx.x * blockDim.y;

        if(l_stack_index >= p_nb_stack)
        {
            return;
        }

        CUDA_glutton_max_stack & l_stack = p_stacks[l_stack_index];

        bool l_new_level = true;
        unsigned int l_best_start_index = 0;

        while(l_stack.get_level() < l_stack.get_size())
        {

            print_single(0,"Stack level = %i", l_stack.get_level());


            if(l_new_level)
            {
                print_single(0,"Search for best score");
                uint32_t l_best_total_score = 0;
                uint32_t l_best_min_max_score = 0;
                unsigned int l_best_last_index = 0;

                // Iterate on all level position information to compute the score of each available transition
                for(unsigned int l_info_index = 0; l_info_index < l_stack.get_level_nb_info(); ++l_info_index)
                {
                    print_single(1,"Info index = %i <=> Position = %i", l_info_index, l_stack.get_position_index(l_info_index));

                    // At the beginning all threads participates to ballot
                    unsigned int l_ballot_result = 0xFFFFFFFF;

                    // Each thread get its word in position info
                    uint32_t l_thread_available_variables = l_stack.get_position_info(l_info_index).get_word(threadIdx.x);

                    print_all(2,"Thread available variables = 0x%" PRIx32, l_thread_available_variables);

                    // Iterate on non null position info words determined by ballot between threads
                    do
                    {
                        // Sync between threads to determine who as some available variables
                        l_ballot_result = __ballot_sync(l_ballot_result, (int) l_thread_available_variables);

                        print_mask(3, l_ballot_result, "Thread available variables = 0x%" PRIx32, l_thread_available_variables);

                        // Ballot result cannot be NULL because we are by construction in a valid situation
                        assert(l_ballot_result);

                        // Determine first lane/thread having an available variable. Result is greater than 0 due to assert
                        unsigned l_elected_thread = __ffs((int)l_ballot_result) - 1;

                        print_single(3, "Elected thread : %i", l_elected_thread);

                        // Eliminate thread from next ballot
                        l_ballot_result &= ~(1u << l_elected_thread);

                        // Copy available variables because we will iterate on it
                        uint32_t l_current_available_variables = l_thread_available_variables;

                        // Share current available variables with all other threads so they can select the same variable
                        l_current_available_variables = __shfl_sync(0xFFFFFFFF, l_current_available_variables, (int)l_elected_thread);

                        // Iterate on available variables of elected thread
                        do
                        {
                            print_single(4, "Current available variables : 0x%" PRIx32, l_current_available_variables);

                            // Determine first available variable. Result  cannot be 0 due to ballot
                            unsigned l_bit_index = __ffs((int)l_current_available_variables) - 1;

                            print_single(4, "Bit index : %i", l_bit_index);

                            // Set variable bit to zero
                            uint32_t l_mask = ~(1u << l_bit_index);
                            l_current_available_variables &= l_mask;

                            // Compute piece index
                            uint32_t l_piece_index = CUDA_piece_position_info2::compute_piece_index(l_elected_thread, l_bit_index);

                            print_single(4, "Piece index : %i", l_piece_index);

                            // Piece orientation
                            uint32_t l_piece_orientation = CUDA_piece_position_info2::compute_orientation_index(l_elected_thread, l_bit_index);

                            print_single(4, "Piece orientation : %i", l_piece_orientation);

                            // Get position index corresponding to this info index
                            uint32_t l_position_index = l_stack.get_position_index(l_info_index);

                            bool l_invalid = false;

                            uint32_t l_info_bits_min = 0xFFFFFFFFu;
                            uint32_t l_info_bits_max = 0;
                            uint32_t l_info_bits_total = 0;

                            if(!threadIdx.x)
                            {
                                l_stack.set_piece_unavailable(l_piece_index);
                            }
                            __syncwarp(0xFFFFFFFF);
                            l_stack.clear_piece_info();
                            CUDA_glutton_max_stack::t_piece_infos & l_piece_infos = l_stack.get_thread_piece_info();

                            uint32_t l_mask_to_apply = l_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(l_bit_index)): 0xFFFFFFFFu;

                            // Each thread store the related info index corresponding to the orientation index
                            unsigned int l_related_thread_index = 0xFFFFFFFFu;

                            // Apply color constraint
                            print_single(4, "Apply color constraints");
                            for(unsigned int l_orientation_index = 0; l_orientation_index < 4; ++l_orientation_index)
                            {
                                uint32_t l_color_id = g_pieces[l_piece_index][(l_orientation_index + l_piece_orientation) % 4];
                                if(l_color_id)
                                {
                                    // Compute position index related to piece side
                                    uint32_t l_related_position_index = l_position_index + g_position_offset[l_orientation_index];

                                    // Check if position is free, if this not the case there is no corresponding index
                                    if(!l_stack.is_position_free(l_related_position_index))
                                    {
                                        print_single(5, "Position %i is not free:\n", l_related_position_index);
                                        continue;
                                    }

                                    // Compute corresponding info index
                                    uint32_t l_related_info_index = l_stack.get_info_index(l_related_position_index);
                                    print_single(5, "Info %i:\n", l_related_info_index);

                                    // Each thread store the related info index corresponding to the orientation index
                                    l_related_thread_index = threadIdx.x == l_orientation_index ? l_related_info_index : l_related_thread_index;

                                    uint32_t l_capability = l_stack.get_position_info(l_related_info_index).get_word(threadIdx.x);
                                    uint32_t l_constraint_capability = p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x);
                                    l_constraint_capability &= l_mask_to_apply;

                                    //print_all(5, "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\n", l_capability, l_constraint_capability);
                                    if((l_invalid = analyze_info(l_capability, l_constraint_capability, l_info_bits_min, l_info_bits_max, l_info_bits_total, l_piece_infos)))
                                    {
                                        break;
                                    }
                                    //print_all(5, "Min %3i Max %3i Total %i\n", l_info_bits_min, l_info_bits_max, l_info_bits_total);
                                    print_mask(5, __ballot_sync(0xFFFFFFFFu, l_capability | l_constraint_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nMin %3i\tMax %3i\tTotal %i\n", l_capability, l_constraint_capability, l_info_bits_min, l_info_bits_max, l_info_bits_total);
                                }
                            }
                            if(!l_invalid)
                            {
                                print_single(4, "Apply piece constraints before selected index");
                                for(unsigned int l_result_info_index = 0; l_result_info_index < l_info_index; ++l_result_info_index)
                                {
                                    if(__all_sync(0xFFFFFFFFu, l_result_info_index != l_related_thread_index))
                                    {
                                        print_single(5, "Info %i:\n", l_result_info_index);
                                        uint32_t l_capability = l_stack.get_position_info(l_result_info_index).get_word(threadIdx.x);
                                        if((l_invalid = analyze_info(l_capability, l_mask_to_apply, l_info_bits_min, l_info_bits_max, l_info_bits_total, l_piece_infos)))
                                        {
                                            break;
                                        }
                                        print_mask(5, __ballot_sync(0xFFFFFFFFu, l_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nMin %3i\tMax %3i\tTotal %i\n", l_capability, l_mask_to_apply, l_info_bits_min, l_info_bits_max, l_info_bits_total);
                                    }
                                }
                            }
                            if(!l_invalid)
                            {
                                print_single(4, "Apply piece constraints after selected index");
                                for(unsigned int l_result_info_index = l_info_index + 1; l_result_info_index < l_stack.get_level_nb_info(); ++l_result_info_index)
                                {
                                    if(__all_sync(0xFFFFFFFFu, l_result_info_index != l_related_thread_index))
                                    {
                                        print_single(5, "Info %i:\n", l_result_info_index);
                                        uint32_t l_capability = l_stack.get_position_info(l_result_info_index).get_word(threadIdx.x);
                                        if((l_invalid = analyze_info(l_capability, l_mask_to_apply, l_info_bits_min, l_info_bits_max, l_info_bits_total, l_piece_infos)))
                                        {
                                            break;
                                        }
                                        print_mask(5, __ballot_sync(0xFFFFFFFFu, l_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nMin %3i\tMax %3i\tTotal %i\n", l_capability, l_mask_to_apply, l_info_bits_min, l_info_bits_max, l_info_bits_total);
                                    }
                                }
                            }
                            // Manage pieces info
                            if(!l_invalid)
                            {
                                uint32_t l_piece_info_total_bit = 0;
                                uint32_t l_piece_info_min_bits = 0xFFFFFFFFu;
                                uint32_t l_piece_info_max_bits = 0;
                                for(unsigned int l_piece_info_index = 0; l_piece_info_index < 8; ++l_piece_info_index)
                                {
                                    CUDA_glutton_max_stack::t_piece_info l_piece_info = l_piece_infos[l_piece_info_index];
                                    if(__all_sync(0xFFFFFFFFu, l_piece_info))
                                    {
                                        unsigned int l_info_piece_index = 8 * threadIdx.x + l_piece_info_index;
                                        if(l_stack.is_piece_available(l_info_piece_index))
                                        {
                                            update_stats(l_piece_info, l_piece_info_min_bits, l_piece_info_max_bits, l_piece_info_total_bit);
                                            print_all(5, "Piece %i:\nMin %3i\tMax %3i\tTotal %i\n", l_info_piece_index, l_piece_info_min_bits, l_piece_info_max_bits, l_piece_info_total_bit);
                                        }
                                    }
                                    else
                                    {
                                        l_invalid = true;
                                        break;
                                    }
                                }
                                if(!l_invalid)
                                {
                                    l_info_bits_total += reduce_add_sync(l_piece_info_total_bit);
                                    l_piece_info_min_bits = reduce_min_sync(l_piece_info_min_bits);
                                    l_info_bits_min = l_piece_info_min_bits < l_info_bits_min ? l_piece_info_min_bits : l_info_bits_min;
                                    l_piece_info_max_bits = reduce_max_sync(l_piece_info_max_bits);
                                    l_info_bits_max = l_piece_info_max_bits > l_info_bits_max ? l_piece_info_max_bits : l_info_bits_max;
                                    print_single(4, "After reduction");
                                    print_single(4, "Min %3i\tMax %3i\tTotal %i\n", l_info_bits_min, l_info_bits_max, l_info_bits_total);
                                }
                            }
                            if(!l_invalid)
                            {
                                // compare with global stats
                                uint32_t l_min_max_score = (l_info_bits_max << 16u) + l_info_bits_min;
                                print_single(4, "Total %i\tMinMax %i\n", l_info_bits_total, l_min_max_score);
                                bool l_record_candidate = false;
                                if(l_info_bits_total > l_best_total_score || (l_info_bits_total == l_best_total_score && l_min_max_score > l_best_min_max_score))
                                {
                                    print_single(4, "New best score Total %i MinMax %i\n", l_info_bits_total, l_min_max_score);
                                    l_best_total_score = l_info_bits_total;
                                    l_best_min_max_score = l_min_max_score;
                                    // Clear previous candidate for best score
                                    for(unsigned int l_clear_info_index = l_best_start_index; l_clear_info_index <= l_best_last_index; ++l_clear_info_index)
                                    {
                                        // Clear previous candidate capability
                                        l_stack.get_best_candidate_info(l_clear_info_index).set_word(threadIdx.x, 0);
                                    }
                                    l_best_start_index = l_info_index;
                                    l_best_last_index = l_info_index;
                                    l_record_candidate = true;
                                }
                                else if(l_info_bits_total == l_best_total_score && l_min_max_score == l_best_min_max_score)
                                {
                                    print_single(4, "Same best score Total %i MinMax %i\n", l_info_bits_total, l_min_max_score);
                                    l_best_last_index = l_info_index;
                                    l_record_candidate = true;
                                }
                                if(l_record_candidate && !threadIdx.x)
                                {
                                    l_stack.get_best_candidate_info(l_info_index).set_bit(l_piece_index, static_cast<emp_types::t_orientation>(l_piece_orientation));
                                }
                                __syncwarp(0xFFFFFFFF);
                            }
                            if(!threadIdx.x)
                            {
                                l_stack.set_piece_available(l_piece_index);
                            }
                            __syncwarp(0xFFFFFFFF);
                        }  while(l_current_available_variables);

                    } while(l_ballot_result);
                }

                // If no best score found there is no interesting transition so go back
                if(!l_best_total_score)
                {
                    print_single(0, "No best score found, go up from one level");
                    print_device_info_position_index(0, l_stack);
                    l_best_start_index = l_stack.pop();
                    print_device_info_position_index(0, l_stack);
                    l_new_level = false;
                    continue;
                }
                // TO DELETE l_stack.unmark_best_candidates();
            }


            // At the beginning all threads participates to ballot
            unsigned int l_ballot_result = 0xFFFFFFFF;
            unsigned int l_best_candidate_index = l_best_start_index;
            uint32_t l_thread_best_candidates;

            print_single(0, "Iterate on best candidate from index %i", l_best_candidate_index);
            // Iterate on best candidates to prepare next level until we find a
            // candidate of reach the end of candidate info
            do
            {
                print_single(1,"Best Info index = %i <=> Position = %i", l_best_candidate_index, l_stack.get_position_index(l_best_candidate_index));

                // Each thread get its word in position info
                l_thread_best_candidates = l_stack.get_best_candidate_info(l_best_candidate_index).get_word(threadIdx.x);

                print_all(1,"Thread best candidates = 0x%" PRIx32, l_thread_best_candidates);

                // Sync between threads to determine who as some available variables
                l_ballot_result = __ballot_sync(l_ballot_result, (int) l_thread_best_candidates);

                print_mask(1, l_ballot_result, "Thread best candidates = 0x%" PRIx32, l_thread_best_candidates);

                // Ballot result cannot be NULL because we are by construction in a valid situation
                if(l_ballot_result)
                {
                    break;
                }
                ++l_best_candidate_index;

            } while(l_best_candidate_index < l_stack.get_level_nb_info());

            // No candidate found so we go up from one level
            if(l_best_candidate_index == l_stack.get_level_nb_info())
            {
                print_single(0, "No more best score, go up from one level");
                print_device_info_position_index(0, l_stack);
                l_best_start_index = l_stack.pop();
                print_device_info_position_index(0, l_stack);
                l_new_level = false;
                continue;
            }

            assert(l_ballot_result);

            // Determine first lane/thread having a candidate. Result is greater than 0 due to assert
            unsigned l_elected_thread = __ffs((int)l_ballot_result) - 1;

            print_single(0, "Elected thread : %i", l_elected_thread);

            // Share current best candidate with all other threads so they can select the same candidate
            l_thread_best_candidates = __shfl_sync(0xFFFFFFFF, l_thread_best_candidates, (int)l_elected_thread);

            // Determine first available candidate. Result  cannot be 0 due to ballot result
            unsigned l_bit_index = __ffs((int)l_thread_best_candidates) - 1;

            print_single(0, "Bit index : %i", l_bit_index);

            print_position_info(6, l_stack);

            // Set variable bit to zero in best candidate and current info
            if(threadIdx.x < 2)
            {
                CUDA_piece_position_info2 & l_position_info = threadIdx.x ? l_stack.get_best_candidate_info(l_best_candidate_index) : l_stack.get_position_info(l_best_candidate_index);
                l_position_info.clear_bit(l_elected_thread, l_bit_index);
            }
            __syncwarp(0xFFFFFFFF);
            print_single(0, "after clear\n");
            print_position_info(6, l_stack);

            // Compute piece index
            uint32_t l_piece_index = CUDA_piece_position_info2::compute_piece_index(l_elected_thread, l_bit_index);

            print_single(0, "Piece index : %i", l_piece_index);

            // Piece orientation
            uint32_t l_piece_orientation = CUDA_piece_position_info2::compute_orientation_index(l_elected_thread, l_bit_index);

            print_single(0, "Piece orientation : %i", l_piece_orientation);

            // Get position index corresponding to this info index
            uint32_t l_position_index = l_stack.get_position_index(l_best_candidate_index);

            {
                // Compute mask to apply which set piece bit to 0
                uint32_t l_mask_to_apply = l_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(l_bit_index)): 0xFFFFFFFFu;
                for (unsigned int l_result_info_index = 0; l_result_info_index < l_best_candidate_index; ++l_result_info_index)
                {
                    print_single(1, "Info %i -> %i:\n", l_result_info_index, l_result_info_index);
                    uint32_t l_capability = l_stack.get_position_info(l_result_info_index).get_word(threadIdx.x);
                    uint32_t l_constraint = l_mask_to_apply;
                    uint32_t l_result = l_capability & l_constraint;
                    print_mask(1, __ballot_sync(0xFFFFFFFFu, l_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability, l_mask_to_apply, l_result);
                    l_stack.get_next_level_position_info(l_result_info_index).set_word(threadIdx.x, l_result);
                }

                // Last position is not treated here because next level has 1 position less
                for (unsigned int l_result_info_index = l_best_candidate_index + 1; l_result_info_index < l_stack.get_level_nb_info() - 1; ++l_result_info_index)
                {
                    print_single(1, "Info %i -> %i:\n", l_result_info_index, l_result_info_index);
                    uint32_t l_capability = l_stack.get_position_info(l_result_info_index).get_word(threadIdx.x);
                    uint32_t l_constraint = l_mask_to_apply;
                    uint32_t l_result = l_capability & l_constraint;
                    print_mask(1, __ballot_sync(0xFFFFFFFFu, l_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability, l_mask_to_apply, l_result);
                    l_stack.get_next_level_position_info(l_result_info_index).set_word(threadIdx.x, l_result);
                }

                // Last position in next level it will be located at l_best_candidate_index
                print_single(0, "Info %i -> %i:\n", l_stack.get_level_nb_info() - 1, l_best_candidate_index);
                uint32_t l_capability = l_stack.get_position_info(l_stack.get_level_nb_info() - 1).get_word(threadIdx.x);
                uint32_t l_constraint = l_mask_to_apply;
                uint32_t l_result = l_capability & l_constraint;
                print_mask(1, __ballot_sync(0xFFFFFFFFu, l_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability, l_mask_to_apply, l_result);
                l_stack.get_next_level_position_info(l_best_candidate_index).set_word(threadIdx.x , l_result);

                print_device_info_position_index(0, l_stack);
                l_stack.push(l_best_candidate_index, l_position_index, l_piece_index, l_piece_orientation);
                print_device_info_position_index(0, l_stack);

                // Print relation index to position
                for(unsigned int l_warp_index = 0; l_warp_index <= l_stack.get_size() / 32; ++l_warp_index)
                {
                    unsigned int l_index = 32 * l_warp_index + threadIdx.x;
                    bool l_valid_index = l_index < l_stack.get_size() && l_stack.is_position_index_used(l_index);
                    print_mask(1, __ballot_sync(0xFFFFFFFFu, l_valid_index), "Index %i -> Position %i", l_index, l_valid_index ? l_stack.get_position_index(l_index) : 0xDEADCAFEu);
                }

                // Apply color constraint
                for(unsigned int l_orientation_index = 0; l_orientation_index < 4; ++l_orientation_index)
                {
                    uint32_t l_color_id = g_pieces[l_piece_index][(l_orientation_index + l_piece_orientation) % 4];
                    print_single(1, "Color Id %i", l_color_id);
                    if(l_color_id)
                    {
                        // Compute position index related to piece side
                        uint32_t l_related_position_index = l_position_index + g_position_offset[l_orientation_index];
                        print_single(1, "Related position index %i", l_related_position_index);

                        // Check if position is free, if this not the case there is no corresponding index
                        if(!l_stack.is_position_free(l_related_position_index))
                        {
                            print_single(1,"Position %i is not free", l_related_position_index);
                            continue;
                        }

                        // Compute corresponding info index
                        uint32_t l_related_info_index = l_stack.get_info_index(l_related_position_index);
                        print_single(1, "Related info index %i", l_related_info_index);

                        // If related index correspond to last position than result is stored in postition where we store the piece
                        uint32_t l_related_target_info_index = l_related_info_index < l_stack.get_level_nb_info() - 1 ? l_related_info_index : l_position_index;

                        print_single(1, "Color Info %i -> %i:\n", l_related_info_index, l_related_target_info_index);
                        print_mask(1, __ballot_sync(0xFFFFFFFFu, l_stack.get_position_info(l_related_info_index).get_word(threadIdx.x) | p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x)), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult 0x%08" PRIx32 "\n", l_stack.get_position_info(l_related_info_index).get_word(threadIdx.x), p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x),l_stack.get_position_info(l_related_info_index).get_word(threadIdx.x) & p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x));
                        l_stack.get_position_info(l_related_target_info_index).CUDA_and(l_stack.get_position_info(l_related_info_index), p_color_constraints.get_info(l_color_id - 1, l_orientation_index));
                    }
                }
            }

            // For latest level we do not search for best score at is zero in any case
            l_new_level = l_stack.get_level() < (l_stack.get_size() - 1);
        }

        print_single(0, "End with stack level %i", l_stack.get_level());
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
        CUDA_piece_position_info2::set_init_value(0);
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
                        if(l_color_id == l_iter_color)
                        {
                            l_color_constraints->get_info(l_color_index, static_cast<unsigned int>(l_color_orientation)).set_bit(l_piece_index, l_piece_orientation);
                        }
                    }
                }
                std::cout << "Color " << l_iter_color << emp_types::orientation2short_string(l_color_orientation) << ":" << std::endl;
                std::cout << l_color_constraints->get_info(l_color_index, static_cast<unsigned int>(l_color_orientation)) << std::endl;
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

        for(unsigned int l_position_index = 0; l_position_index < p_info.get_nb_pieces(); ++l_position_index)
        {
            std::cout << "Position " << l_position_index << "(" << p_info.get_x(l_position_index) << "," <<p_info.get_y(l_position_index) << "):" << std::endl;
            std::cout << l_initial_capability[l_position_index] << std::endl;
        }
        std::unique_ptr<CUDA_memory_managed_array<uint32_t>> l_cuda_array{new CUDA_memory_managed_array<uint32_t>(32)};
        for(unsigned int l_index = 0; l_index < 32 ; ++l_index)
        {
            (*l_cuda_array)[l_index] = 0;
        }

        emp_situation l_start_situation;

        unsigned int l_size = l_nb_pieces - l_start_situation.get_level();
        std::unique_ptr<CUDA_glutton_max_stack> l_stack{new CUDA_glutton_max_stack(l_size,l_nb_pieces)};
        for(unsigned int l_piece_index = 0; l_piece_index < l_nb_pieces; ++l_piece_index)
        {
            l_stack->set_piece_available(l_piece_index);
        }

        // Prepare stack with info of initial situation
        uint32_t l_info_index = 0;
        for(unsigned int l_position_index = 0; l_position_index < l_nb_pieces; ++l_position_index)
        {
            unsigned int l_x = p_info.get_x(l_position_index);
            unsigned int l_y = p_info.get_y(l_position_index);
            if(!l_start_situation.contains_piece(l_x, l_y))
            {
                l_stack->set_position_info_relation(l_info_index, l_position_index);
                l_stack->set_position_info(l_info_index, l_initial_capability[l_position_index]);
                ++l_info_index;
            }
            else
            {
                l_stack->set_piece_unavailable(l_start_situation.get_piece(l_x, l_y).first - 1);
            }
        }
        delete[] l_initial_capability;
        print_host_info_position_index(0, *l_stack);

        // Reset CUDA error status
        cudaGetLastError();
        std::cout << "Launch kernels" << std::endl;
        dim3 l_block_info(32, 1);
        dim3 l_grid_info(1);
        kernel<<<l_grid_info, l_block_info>>>(l_stack.get(), 1, *l_color_constraints);
        //test_kernel<<<l_grid_info, l_block_info>>>(l_stack.get(), 1, *l_cuda_array);
        cudaDeviceSynchronize();
        gpuErrChk(cudaGetLastError());

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
                unsigned int l_x = p_info.get_x(CUDA_glutton_max_stack::decode_position_index(l_played_info));
                unsigned int l_y = p_info.get_y(CUDA_glutton_max_stack::decode_position_index(l_played_info));
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
        for(unsigned int l_index = 0; l_index < l_size; ++l_index)
        {
            std::cout << l_stack->get_position_info(l_index) << std::endl;
            //l_stack->push();
        }

        std::cout << "CUDA array content" << std::endl;
        for(unsigned int l_index = 0; l_index < 32; ++l_index)
        {
            std::cout << "cuda_array[" << l_index << "] = " << (*l_cuda_array)[l_index] << std::endl;
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

        print_all(0, "managed_array[%i] = %i", threadIdx.x, p_array[threadIdx.x]);

        for(unsigned int l_piece_index = 0; l_piece_index < g_nb_pieces; ++l_piece_index)
        {
            print_single(1, "Piece[%i]={%i, %i, %i, %i}\n", l_piece_index + 1, g_pieces[l_piece_index][0], g_pieces[l_piece_index][1], g_pieces[l_piece_index][2], g_pieces[l_piece_index][3]);
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
            l_stack.push(0,0,0,0);
        }
        for(unsigned int l_index = 0; l_index < l_stack.get_size(); ++l_index)
        {
            l_stack.pop();
        }
    }

}
// EOF
