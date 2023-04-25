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

#include "my_cuda.h"
#include "CUDA_info.h"
#include "CUDA_utils.h"
#include "CUDA_color_constraints.h"
#include "CUDA_glutton_max_stack.h"
#include "CUDA_glutton_stack_XML_converter.h"
#include "emp_FSM_info.h"
#include "emp_piece_db.h"
#include "emp_situation.h"
#include "situation_string_formatter.h"
#include "quicky_exception.h"
#include <functional>
#ifdef ENABLE_CUDA_CODE
#include <nvfunctional>
#else // ENABLE_CUDA_CODE
#include <numeric>
#include <algorithm>
#endif // ENABLE_CUDA_CODE
#define LOG_EXECUTION
#define VERBOSITY_LEVEL 7

#include "CUDA_print.h"


/**
 * This file declare functions that will be implemented for
 * CUDA: performance. Corresponding implementation is in CUDA_glutton_max.cu
 * CPU: alternative implementation to debug algorithm. Corresponding implementation is in CUDA_glutton_max.cpp
 */
namespace edge_matching_puzzle
{

    /**
     * Store piece representation.
     * First dimension is piece index ( ie piece id -1 )
     * Second dimension is border orientation
     */
    extern __constant__ uint32_t g_pieces[256][4];

    /**
     * Return position offset for each orientation
     * NORTH : 0 EAST:1 SOUTH:2 WEST:3
     * Position offset depend on puzzle dimensions
     */
    extern __constant__ int g_position_offset[4];

    /**
     * Number of pieces remaining to set
     */
    extern __constant__ unsigned int g_nb_pieces;

    /**
     * Class implementing RAII idiom to manage piece availability
     * It is used when there are many way to exit from code andd that we need
     * to release piece when leaving
     */
    class unvailability_lock_gard
    {
      public:
        inline
        __device__
        unvailability_lock_gard(CUDA_glutton_max_stack & p_stack
                               ,uint32_t p_piece_index
                               )
                               :m_stack(p_stack)
                               ,m_piece_index(p_piece_index)
        {
#ifdef ENABLE_CUDA_CODE
            if(!threadIdx.x)
#endif // ENABLE_CUDA_CODE
            {
                m_stack.set_piece_unavailable(m_piece_index);
            }
#ifdef ENABLE_CUDA_CODE
            __syncwarp(0xFFFFFFFF);
#endif // ENABLE_CUDA_CODE
        }

        inline
        __device__
        ~unvailability_lock_gard()
        {
#ifdef ENABLE_CUDA_CODE
            if(!threadIdx.x)
#endif // ENABLE_CUDA_CODE
            {
                m_stack.set_piece_available(m_piece_index);
            }
#ifdef ENABLE_CUDA_CODE
            __syncwarp(0xFFFFFFFF);
#endif // ENABLE_CUDA_CODE
        }

      protected:

        inline
        __device__
        CUDA_glutton_max_stack & get_stack()
        {
            return m_stack;
        }
      private:
        CUDA_glutton_max_stack & m_stack;
        uint32_t m_piece_index;
    };

    class CUDA_glutton_max
    {

      public:

        inline
        CUDA_glutton_max(const emp_piece_db & p_piece_db
                        ,const emp_FSM_info & p_info
                        )
        :m_piece_db{p_piece_db}
        ,m_info(p_info)
        {

        }

        inline static
        void prepare_constants(const emp_piece_db & p_piece_db
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

#ifdef ENABLE_CUDA_CODE
            my_cuda::CUDA_info();

            // Fill constant variables
            cudaMemcpyToSymbol(g_pieces, l_pieces.data(), l_pieces.size() * sizeof(uint32_t ));
            cudaMemcpyToSymbol(g_position_offset, l_x_offset.data(), l_x_offset.size() * sizeof(int));
            cudaMemcpyToSymbol(g_nb_pieces, &l_nb_pieces, sizeof(unsigned int));
#else // ENABLE_CUDA_CODE
            for(unsigned int l_index = 0; l_index < 256 * 4; ++l_index)
            {
                g_pieces[l_index / 4][l_index % 4] = l_pieces[l_index];
            }
            for(unsigned int l_index = 0; l_index < 4; ++l_index)
            {
                g_position_offset[l_index] = l_x_offset[l_index];
            }
            g_nb_pieces = l_nb_pieces;

#endif // ENABLE_CUDA_CODE
        }

        inline static
        std::unique_ptr<CUDA_color_constraints>
        prepare_color_constraints(const emp_piece_db & p_piece_db
                                 ,const emp_FSM_info & p_info
                                 )
        {
            // Prepare color constraints
            CUDA_piece_position_info2::set_init_value(0);
            // We want to allocate an array able to contains all colors so with
            // size == color max Id + 1 because in some cases number of color
            // is less than color max id
            std::unique_ptr<CUDA_color_constraints> l_color_constraints{new CUDA_color_constraints(static_cast<unsigned int>(p_piece_db.get_border_color_id()))};
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
            return l_color_constraints;

        }

        inline static
        CUDA_piece_position_info2 *
        prepare_initial_capability(const emp_piece_db & p_piece_db
                                  ,const emp_FSM_info & p_info
                                  )
        {
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
            return l_initial_capability;
        }

        inline static
        std::unique_ptr<CUDA_glutton_max_stack>
        prepare_stack(const emp_piece_db & p_piece_db
                     ,const emp_FSM_info & p_info
                     ,emp_situation & p_start_situation
                     )
        {
            auto * l_initial_capability = prepare_initial_capability(p_piece_db, p_info);
            unsigned int l_nb_pieces = p_info.get_nb_pieces();
            unsigned int l_size = l_nb_pieces - p_start_situation.get_level();
            std::unique_ptr<CUDA_glutton_max_stack> l_stack{new CUDA_glutton_max_stack(l_size,l_nb_pieces)};
            for(unsigned int l_piece_index = 0; l_piece_index < l_nb_pieces; ++l_piece_index)
            {
                l_stack->set_piece_available(l_piece_index);
            }

            // Prepare stack with info of initial situation
            info_index_t l_info_index{0u};
            for(unsigned int l_position_index = 0; l_position_index < l_nb_pieces; ++l_position_index)
            {
                unsigned int l_x = p_info.get_x(l_position_index);
                unsigned int l_y = p_info.get_y(l_position_index);
                if(!p_start_situation.contains_piece(l_x, l_y))
                {
                    l_stack->set_position_info_relation(l_info_index, position_index_t(l_position_index));
                    l_stack->set_position_info(l_info_index, l_initial_capability[l_position_index]);
                    ++l_info_index;
                }
                else
                {
                    l_stack->set_piece_unavailable(p_start_situation.get_piece(l_x, l_y).first - 1);
                }
            }
            delete[] l_initial_capability;
            print_host_info_position_index(0, *l_stack);
            return l_stack;
        }

        /**
         * Print information relating info index and position index
         * @param p_indent_level indentation level
         * @param p_stack
         */
        inline static
        void
        print_host_info_position_index(unsigned int p_indent_level
                                      ,const CUDA_glutton_max_stack & p_stack
                                      )
        {
            std::cout << std::string(p_indent_level,' ') <<  "====== Position index <-> Info index ======" << std::endl;
            for(position_index_t l_index{0u}; l_index < p_stack.get_nb_pieces(); ++l_index)
            {
                std::cout << std::string(p_indent_level,' ') << "Position[" << l_index << "] -> Index " << p_stack.get_info_index(l_index) << std::endl;
            }
            for(info_index_t l_index{0u}; l_index < p_stack.get_size(); ++l_index)
            {
                std::cout << std::string(p_indent_level,' ') << (l_index < p_stack.get_level_nb_info() ? '*' : ' ') << " Index[" << l_index << "] -> Position " << p_stack.get_position_index(l_index) << std::endl;
            }
        }

        inline static
        void display_result(const CUDA_glutton_max_stack & p_stack
                           ,emp_situation & p_start_situation
                           ,const emp_FSM_info & p_info
                           )
        {
            if(p_stack.is_empty())
            {
                std::cout << "Empty stack" << std::endl;
            }
            else
            {
                unsigned int l_max_level = p_stack.get_level() - (unsigned int)p_stack.is_full();
                for(unsigned int l_level = 0; l_level <= l_max_level; ++l_level)
                {
                    CUDA_glutton_max_stack::played_info_t l_played_info = p_stack.get_played_info(l_level);
                    unsigned int l_x = p_info.get_x(static_cast<uint32_t>(CUDA_glutton_max_stack::decode_position_index(l_played_info)));
                    unsigned int l_y = p_info.get_y(static_cast<uint32_t>(CUDA_glutton_max_stack::decode_position_index(l_played_info)));
                    assert(!p_start_situation.contains_piece(l_x, l_y));
                    p_start_situation.set_piece(l_x, l_y
                                               ,emp_types::t_oriented_piece{static_cast<emp_types::t_piece_id >(1 + CUDA_glutton_max_stack::decode_piece_index(l_played_info))
                                               ,static_cast<emp_types::t_orientation>(CUDA_glutton_max_stack::decode_orientation_index(l_played_info))}
                                               );
                }
                std::cout << "Situation with stack played info:" << std::endl;
                std::cout << situation_string_formatter<emp_situation>::to_string(p_start_situation) << std::endl;
            }
            for(info_index_t l_index{0u}; l_index < p_stack.get_level_nb_info(); ++l_index)
            {
                std::cout << p_stack.get_position_info(l_index) << std::endl;
                //l_stack->push();
            }
        }

        inline static
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

        inline static
        __device__
#ifdef ENABLE_CUDA_CODE
       void count_result_nb_bits(uint32_t p_result_capability
#else // ENABLE_CUDA_CODE
       void count_result_nb_bits(pseudo_CUDA_thread_variable<uint32_t> p_result_capability
#endif // ENABLE_CUDA_CODE
                                ,uint32_t & p_min
                                ,uint32_t & p_max
                                ,uint32_t & p_total
                                )

       {
#ifdef ENABLE_CUDA_CODE
           uint32_t l_info_bits = my_cuda::reduce_add_sync(__popc(p_result_capability));
#else // ENABLE_CUDA_CODE
           uint32_t l_info_bits = 0;
           for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
            {
                l_info_bits += __builtin_popcount(p_result_capability[l_threadIdx_x]);
            }
#endif // ENABLE_CUDA_CODE
           update_stats(l_info_bits, p_min, p_max, p_total);
        }

        /**
         * Method to count the number of orientation possible for each piece at current position
         * @param p_result_capability capability for current position
         * @param p_piece_info variable storinng number of bits per piece
         */
        inline static
        __device__
#ifdef ENABLE_CUDA_CODE
        void analyze_info(uint32_t p_result_capability
                         ,CUDA_glutton_max_stack::t_piece_infos & p_piece_info
#else // ENABLE_CUDA_CODE
        void analyze_info(pseudo_CUDA_thread_variable<uint32_t> p_result_capability
                         ,pseudo_CUDA_thread_variable<CUDA_glutton_max_stack::t_piece_infos> & p_piece_info
#endif // ENABLE_CUDA_CODE
                         )
        {
#ifdef ENABLE_CUDA_CODE
            for(unsigned short & l_piece_index : p_piece_info)
#else // ENABLE_CUDA_CODE
            for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
            {
                for (unsigned short & l_piece_index : p_piece_info[l_threadIdx_x])
#endif // ENABLE_CUDA_CODE
                {
#ifdef ENABLE_CUDA_CODE
                    l_piece_index += static_cast<CUDA_glutton_max_stack::t_piece_info>(__popc(static_cast<int>(p_result_capability & 0xFu)));
                    p_result_capability = p_result_capability >> 4;
#else // ENABLE_CUDA_CODE
                    l_piece_index += static_cast<CUDA_glutton_max_stack::t_piece_info>(__builtin_popcount(static_cast<int>(p_result_capability[l_threadIdx_x] & 0xFu)));
                    p_result_capability[l_threadIdx_x] = p_result_capability[l_threadIdx_x] >> 4;
#endif // ENABLE_CUDA_CODE
                }
#ifndef ENABLE_CUDA_CODE
            }
#endif // ENABLE_CUDA_CODE
        }

        inline static
        __device__
        void print_position_info(unsigned int p_indent_level
                                ,const CUDA_glutton_max_stack & p_stack
                                ,const CUDA_piece_position_info2 & (CUDA_glutton_max_stack::*p_accessor)(info_index_t) const
                                )
        {
            for(info_index_t l_display_index{0u}; l_display_index < p_stack.get_level_nb_info(); ++l_display_index)
            {
#ifdef ENABLE_CUDA_CODE
                my_cuda::print_single(p_indent_level + 1, "Index = %" PRIu32 " <=> Position = %" PRIu32 "\n" ,static_cast<uint32_t>(l_display_index), static_cast<uint32_t>(p_stack.get_position_index(l_display_index)));
                uint32_t l_word = (p_stack.*p_accessor)(l_display_index).get_word(threadIdx.x);
                my_cuda::print_mask(p_indent_level + 2, __ballot_sync(0xFFFFFFFF, l_word), "Info = 0x%" PRIx32, l_word);
#else // ENABLE_CUDA_CODE
                my_cuda::print_single(p_indent_level + 1, {0, 1, 1}, "Index = %" PRIu32 " <=> Position = %" PRIu32 "\n" ,static_cast<uint32_t>(l_display_index), static_cast<uint32_t>(p_stack.get_position_index(l_display_index)));
                pseudo_CUDA_thread_variable<uint32_t> l_word{[&](dim3 threadIdx){ return (p_stack.*p_accessor)(l_display_index).get_word(threadIdx.x);}};
                uint32_t l_print_mask = __ballot_sync(0xFFFFFFFF, l_word);
                for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
                {
                    my_cuda::print_mask(p_indent_level + 2, l_print_mask, {l_threadIdx_x, 1, 1}, "Info = 0x%" PRIx32, l_word[l_threadIdx_x]);
                }
#endif // ENABLE_CUDA_CODE
            }
        }

        inline static
        __device__
        void print_position_info(unsigned int p_indent_level
                                ,const CUDA_glutton_max_stack & p_stack
                                )
        {
            my_cuda::print_single(p_indent_level, "Position info:");
            print_position_info(p_indent_level, p_stack, &CUDA_glutton_max_stack::get_position_info);
        }


        /**
         * Print information relating info index and position index
         * @param p_indent_level indentation level
         * @param p_stack
         */
        inline static
        __device__
        void
        print_device_info_position_index(unsigned int p_indent_level
                                        ,const CUDA_glutton_max_stack & p_stack
                                        )
        {
            my_cuda::print_single(p_indent_level, "====== Position index <-> Info index ======\n");
            for(unsigned int l_warp_index = 0u; l_warp_index <= (static_cast<uint32_t>(p_stack.get_nb_pieces()) / 32u); ++l_warp_index)
            {
#ifdef ENABLE_CUDA_CODE
                position_index_t l_thread_index{l_warp_index * 32u + threadIdx.x};
                my_cuda::print_mask(p_indent_level
                                   ,__ballot_sync(0xFFFFFFFF, l_thread_index < p_stack.get_nb_pieces())
                                   ,"Position[%" PRIu32 "] -> Index %" PRIu32
                                   ,static_cast<uint32_t>(l_thread_index)
                                   ,l_thread_index < p_stack.get_nb_pieces() ? static_cast<uint32_t>(p_stack.get_info_index(position_index_t(l_thread_index))) : 0xDEADCAFEu
                                   );
#else // ENABLE_CUDA_CODE
                pseudo_CUDA_thread_variable<position_index_t> l_thread_index{[=](dim3 threadIdx){return static_cast<position_index_t >(l_warp_index * 32u + threadIdx.x);}};
                uint32_t l_print_mask = __ballot_sync(0xFFFFFFFF, [&](dim3 threadIdx){return l_thread_index[threadIdx.x] < p_stack.get_nb_pieces();});
                for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
                {
                    my_cuda::print_mask(p_indent_level
                                       ,l_print_mask
                                       ,{l_threadIdx_x, 0, 0}
                                       ,"Position[%" PRIu32 "] -> Index %" PRIu32
                                       ,static_cast<uint32_t>(l_thread_index[l_threadIdx_x])
                                       ,l_thread_index[l_threadIdx_x] < p_stack.get_nb_pieces() ? static_cast<uint32_t>(p_stack.get_info_index(position_index_t(l_thread_index[l_threadIdx_x]))) : 0xDEADCAFEu
                                       );
                }
#endif // ENABLE_CUDA_CODE
            }
            for(unsigned int l_index = 0; l_index <= (p_stack.get_size() / 32); ++l_index)
            {
#ifdef ENABLE_CUDA_CODE
                unsigned int l_thread_index = 32 * l_index + threadIdx.x;
                my_cuda::print_mask(p_indent_level
                                   ,__ballot_sync(0xFFFFFFFF, l_thread_index < p_stack.get_size())
                                   ,"%c Index[%" PRIu32 "] -> Position %" PRIu32
                                   ,l_thread_index < p_stack.get_size() - p_stack.get_level() ? '*' : ' '
                                   ,l_thread_index
                                   ,l_thread_index < p_stack.get_size() ? static_cast<uint32_t>(p_stack.get_position_index(info_index_t(l_thread_index))) : 0xDEADCAFEu
                                   );
#else // ENABLE_CUDA_CODE
                pseudo_CUDA_thread_variable<unsigned int> l_thread_index{[=](dim3 threadIdx){ return 32 * l_index + threadIdx.x;}};
                uint32_t l_print_mask = __ballot_sync(0xFFFFFFFF, [&](dim3 threadIdx){ return l_thread_index[threadIdx.x] < p_stack.get_size();});
                for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
                {
                    my_cuda::print_mask(p_indent_level
                                       ,l_print_mask
                                       ,{l_threadIdx_x, 0, 0}
                                       ,"%c Index[%" PRIu32 "] -> Position %" PRIu32
                                       ,l_thread_index[l_threadIdx_x] < p_stack.get_size() - p_stack.get_level() ? '*' : ' '
                                       ,l_thread_index[l_threadIdx_x]
                                       ,l_thread_index[l_threadIdx_x] < p_stack.get_size() ? static_cast<uint32_t>(p_stack.get_position_index(info_index_t(l_thread_index[l_threadIdx_x]))) : 0xDEADCAFEu
                                       );
                }
#endif // ENABLE_CUDA_CODE
            }
        }

        inline static
        __device__
        bool is_position_invalid(
#ifdef ENABLE_CUDA_CODE
                                 uint32_t p_result
#else // ENABLE_CUDA_CODE
                                 const pseudo_CUDA_thread_variable<uint32_t> & p_result
#endif // ENABLE_CUDA_CODE
                                )
        {
            return (!__any_sync(0xFFFFFFFFu, p_result));
        };

        /**
         * Apply mask to position info in interval [p_start_index, p_limit_index[ and store the result in next level
         * @param p_start_index start index of interval to work on
         * @param p_limit_index end index of interval, this position info at this index is not modified
         * @param p_stack stack function is working on
         * @param p_mask_to_apply mask to apply to each position
         * @return true if some position is invalid
         */
        inline static
        __device__
        bool compute_next_level_position_info(info_index_t p_start_index
                                             ,info_index_t p_limit_index
                                             ,CUDA_glutton_max_stack & p_stack
#ifdef ENABLE_CUDA_CODE
                                             ,uint32_t p_mask_to_apply
#else // ENABLE_CUDA_CODE
                                             ,const pseudo_CUDA_thread_variable<uint32_t> & p_mask_to_apply
#endif // ENABLE_CUDA_CODE
        )
        {
            auto l_do_apply = [&](info_index_t p_index) -> bool
            {
                return true;
            };

            auto l_treat = [&](info_index_t p_result_info_index
#ifdef ENABLE_CUDA_CODE
                                       ,uint32_t p_capability
                                       ,uint32_t p_result
#else // ENABLE_CUDA_CODE
                                       ,const pseudo_CUDA_thread_variable<uint32_t> & p_capability
                                       ,const pseudo_CUDA_thread_variable<uint32_t> & p_result
#endif // ENABLE_CUDA_CODE
                                       )
            {
#ifdef ENABLE_CUDA_CODE
                p_stack.get_next_level_position_info(p_result_info_index).set_word(threadIdx.x, p_result);
#else // ENABLE_CUDA_CODE
                for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++ l_threadIdx_x)
                {
                    p_stack.get_next_level_position_info(p_result_info_index).set_word(l_threadIdx_x, p_result[l_threadIdx_x]);
                }
#endif // ENABLE_CUDA_CODE
            };

            return apply_simple_mask(p_start_index, p_limit_index, p_stack, p_mask_to_apply, l_do_apply, is_position_invalid, l_treat);
        }

        /**
         * Function applying simple mask on an index range
         * some index can be skipped thanks to p_do_appply
         * in case of invalid index return by p_is_position_invalid the loop
         * is interrupted and function return true
         * @param p_start_index
         * @param p_limit_index
         * @param p_stack
         * @param p_mask_to_apply
         * @param p_do_apply indicate if treatment should be apply to current index
         * @param p_is_position_invalid indicate if a position is valid or not
         * @param p_treat_simple_mask treat result of applied mask
         * @return true if an invalid position is obtained
         */
        inline static
        __device__
        bool apply_simple_mask(info_index_t p_start_index
                              ,info_index_t p_limit_index
                              ,CUDA_glutton_max_stack & p_stack
#ifdef ENABLE_CUDA_CODE
                              ,uint32_t p_mask_to_apply
                              ,nvstd::function<bool(info_index_t)> p_do_apply
                              ,nvstd::function<bool(uint32_t)> p_is_position_invalid
                              ,nvstd::function<void(info_index_t, uint32_t, uint32_t)> p_treat_simple_mask
#else
                              ,const pseudo_CUDA_thread_variable<uint32_t> & p_mask_to_apply
                              ,std::function<bool(info_index_t)> p_do_apply
                              ,std::function<bool(const pseudo_CUDA_thread_variable<uint32_t> &)> p_is_position_invalid
                              ,std::function<void(info_index_t
                              ,const pseudo_CUDA_thread_variable<uint32_t> &
                              ,const pseudo_CUDA_thread_variable<uint32_t> &
                                                 )
                                            > p_treat_simple_mask
#endif // ENABLE_CUDA_CODE
                              )
        {
            for(info_index_t l_result_info_index{p_start_index}; l_result_info_index < p_limit_index; ++l_result_info_index)
            {
                if(p_do_apply(l_result_info_index))
                {
#if VERBOSITY_LEVEL >= 6
                    my_cuda::print_single(5, "Info %i <=> Position %i :\n", static_cast<uint32_t>(l_result_info_index), static_cast<uint32_t>(p_stack.get_position_index(l_result_info_index)));
#endif // VERBOSITY_LEVEL >= 6
#ifdef ENABLE_CUDA_CODE
                    uint32_t l_capability = p_stack.get_position_info(l_result_info_index).get_word(threadIdx.x);
                    uint32_t l_result_capability = l_capability & p_mask_to_apply;
#if VERBOSITY_LEVEL >= 6
                    my_cuda::print_mask(1, __ballot_sync(0xFFFFFFFFu, l_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability, p_mask_to_apply, l_result_capability);
#endif // VERBOSITY_LEVEL >= 6
#else // ENABLE_CUDA_CODE
                    pseudo_CUDA_thread_variable<uint32_t> l_capability {[&](dim3 threadIdx){return p_stack.get_position_info(l_result_info_index).get_word(threadIdx.x);}};
                    pseudo_CUDA_thread_variable<uint32_t> l_result_capability =  l_capability & p_mask_to_apply;
#if VERBOSITY_LEVEL >= 6
                    uint32_t l_print_mask = __ballot_sync(0xFFFFFFFFu, l_capability);
                    for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++ l_threadIdx_x)
                    {
                        my_cuda::print_mask(5, l_print_mask, {l_threadIdx_x, 1, 1}, "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability[l_threadIdx_x], p_mask_to_apply[l_threadIdx_x], l_result_capability[l_threadIdx_x]);
                    }
#endif // VERBOSITY_LEVEL >= 6
#endif // ENABLE_CUDA_CODE
                    if(p_is_position_invalid(l_result_capability))
                    {
#if VERBOSITY_LEVEL >= 6
                        my_cuda::print_single(5, "INVALID\n");
#endif // VERBOSITY_LEVEL >= 6
                        return true;
                    }
                    p_treat_simple_mask(l_result_info_index, l_capability, l_result_capability);
                }
            }
            return false;
        }

        inline static
        __device__
        bool
        apply_color_constraints(uint32_t p_piece_index
                               ,uint32_t p_piece_orientation
                               ,position_index_t p_position_index
                               ,CUDA_glutton_max_stack & p_stack
                               ,const CUDA_color_constraints & p_color_constraints
#ifdef ENABLE_CUDA_CODE
                               ,uint32_t p_mask_to_apply
                               ,nvstd::function<bool(uint32_t)> p_is_position_invalid
                               ,nvstd::function<void(uint32_t, info_index_t, uint32_t, uint32_t)> p_treat_result
#else
                               ,const pseudo_CUDA_thread_variable<uint32_t> & p_mask_to_apply
                               ,std::function<bool(const pseudo_CUDA_thread_variable<uint32_t> &)> p_is_position_invalid
                               ,std::function<void(uint32_t, info_index_t, uint32_t, const pseudo_CUDA_thread_variable<uint32_t> &)> p_treat_result
#endif // ENABLE_CUDA_CODE
                               )
        {
            for(unsigned int l_orientation_index = 0; l_orientation_index < 4; ++l_orientation_index)
            {
                uint32_t l_color_id = g_pieces[p_piece_index][(l_orientation_index + p_piece_orientation) % 4];
#if VERBOSITY_LEVEL >= 6
                my_cuda::print_single(5, "Color Id %i", l_color_id);
#endif // VERBOSITY_LEVEL >= 6
                if(l_color_id)
                {
                    // Compute position index related to piece side
                    position_index_t l_related_position_index{static_cast<uint32_t>(p_position_index) + g_position_offset[l_orientation_index]};

                    // Check if position is free, if this not the case there is no corresponding index
                    if(!p_stack.is_position_free(l_related_position_index))
                    {
#if VERBOSITY_LEVEL >= 6
                        my_cuda::print_single(5, "Position %i is not free\n", static_cast<uint32_t>(l_related_position_index));
#endif // VERBOSITY_LEVEL >= 6
                        continue;
                    }

                    // Compute corresponding info index
                    info_index_t l_related_info_index = p_stack.get_info_index(l_related_position_index);
#if VERBOSITY_LEVEL >= 6
                    my_cuda::print_single(5, "Info %i <=> Position %i :\n", static_cast<uint32_t>(l_related_info_index), static_cast<uint32_t>(l_related_position_index));
#endif // VERBOSITY_LEVEL >= 6

#ifdef ENABLE_CUDA_CODE
                    uint32_t l_capability = p_stack.get_position_info(l_related_info_index).get_word(threadIdx.x);
                    uint32_t l_constraint_capability = p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x);
                    l_constraint_capability &= p_mask_to_apply;
                    uint32_t l_result_capability = l_capability & l_constraint_capability;
#else // ENABLE_CUDA_CODE
                    pseudo_CUDA_thread_variable<uint32_t> l_capability{[&](dim3 threadIdx) { return p_stack.get_position_info(l_related_info_index).get_word(threadIdx.x);}};
                    pseudo_CUDA_thread_variable<uint32_t> l_constraint_capability{[&](dim3 threadIdx) { return p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x);}};
                    l_constraint_capability &= p_mask_to_apply;
                    pseudo_CUDA_thread_variable<uint32_t> l_result_capability{l_capability & l_constraint_capability};
#endif // ENABLE_CUDA_CODE

#if VERBOSITY_LEVEL >= 6
#ifdef ENABLE_CUDA_CODE
                    my_cuda::print_mask(5, __ballot_sync(0xFFFFFFFFu, l_capability | l_constraint_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult 0x%08" PRIx32 "\n", l_capability, l_constraint_capability, l_result_capability);
#else // ENABLE_CUDA_CODE
                    uint32_t l_print_mask = 0;
                    dim3 threadIdx{0, 1, 1};
                    for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
                    {
                        l_print_mask |= ((l_capability[threadIdx.x] | l_constraint_capability[threadIdx.x]) != 0) << threadIdx.x;
                        my_cuda::print_mask(5, l_print_mask, threadIdx, "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult 0x%08" PRIx32 "\n", l_capability[threadIdx.x], l_constraint_capability[threadIdx.x], l_result_capability[threadIdx.x]);
                    }
#endif // ENABLE_CUDA_CODE
#endif // VERBOSITY_LEVEL >= 6

                    // Check validity after applying masks
                    if(p_is_position_invalid(l_result_capability))
                    {
#if VERBOSITY_LEVEL >= 6
                        my_cuda::print_single(5, "INVALID\n");
#endif // VERBOSITY_LEVEL >= 6
                        return true;
                    }

                    p_treat_result(l_orientation_index, l_related_info_index, l_color_id, l_result_capability);
                } // if color_id
            } // End of side for loop
            return false;
        }
        /**
         * Function allowing to iterate on all bit set in p_thread variable of
         * warp. Each thread in the warp provide an uin32_t variable, their
         * bit that are set will be listed in order and thread index, bit index
         * will be provided to p_func with a refernce on thread variable.
         * p_func can perfom a treatment and if needed modify thread variable to set bit to zero for example
         * @param p_thread_variable
         * @param p_func
         */
        inline static
        __device__
        void warp_iterate(
#ifdef ENABLE_CUDA_CODE
                           uint32_t & p_thread_variable
                          ,nvstd::function<void(uint32_t, uint32_t, nvstd::function<void()> &, nvstd::function<bool(uint32_t)> &, uint32_t &)> p_func
                          ,nvstd::function<void()> p_init
                          ,nvstd::function<bool(uint32_t)> p_is_position_invalid
#else // ENABLE_CUDA_CODE
                           pseudo_CUDA_thread_variable<uint32_t> & p_thread_variable
                          ,std::function<void(uint32_t, uint32_t, std::function<void()> &, std::function<bool(const pseudo_CUDA_thread_variable<uint32_t> &)> &, pseudo_CUDA_thread_variable<uint32_t> &)> p_func
                          ,std::function<void()> p_init
                          ,std::function<bool(const pseudo_CUDA_thread_variable<uint32_t> &)> p_is_position_invalid
#endif // ENABLE_CUDA_CODE
                          )
        {
#if VERBOSITY_LEVEL >= 5
#ifdef ENABLE_CUDA_CODE
            my_cuda::print_all(4,"Thread available variables = 0x%" PRIx32, p_thread_variable);
#else // ENABLE_CUDA_CODE
            for(dim3 threadIdx{0, 0, 0} ; threadIdx.x < 32 ; ++threadIdx.x)
            {
                my_cuda::print_all(4, threadIdx, "Thread available variables = 0x%" PRIx32, p_thread_variable[threadIdx.x]);
            }
#endif // ENABLE_CUDA_CODE
#endif // VERBOSITY_LEVEL >= 4

            // At the beginning all threads participates to ballot
            unsigned int l_ballot_result = 0xFFFFFFFF;

            // Iterate on non null position info words determined by ballot between threads
            do
            {
                // Sync between threads to determine who as some available variables
#ifdef ENABLE_CUDA_CODE
                l_ballot_result = __ballot_sync(l_ballot_result, (int) p_thread_variable);
#if VERBOSITY_LEVEL >= 6
                my_cuda::print_mask(5, l_ballot_result, "Thread available variables = 0x%" PRIx32, p_thread_variable);
#endif // VERBOSITY_LEVEL >= 6
#else // ENABLE_CUDA_CODE
                l_ballot_result = __ballot_sync(l_ballot_result, p_thread_variable);
#if VERBOSITY_LEVEL >= 6
                for(dim3 l_threadIdx{0, 1 , 1}; l_threadIdx.x < 32; ++l_threadIdx.x)
                {
                    my_cuda::print_mask(5, l_ballot_result, l_threadIdx, "Thread available variables = 0x%" PRIx32, p_thread_variable[l_threadIdx.x]);
                }
#endif // VERBOSITY_LEVEL >= 6
#endif // ENABLE_CUDA_CODE


                // Ballot result cannot be NULL because we are by construction in a valid situation
                assert(l_ballot_result);

                // Determine first lane/thread having an available variable. Result is greater than 0 due to assert
                unsigned l_elected_thread = __ffs((int)l_ballot_result) - 1;

#if VERBOSITY_LEVEL >= 4
                my_cuda::print_single(3, "Elected thread : %i", l_elected_thread);
#endif // VERBOSITY_LEVEL >= 4

                // Eliminate thread from next ballot
                l_ballot_result &= ~(1u << l_elected_thread);

                // Copy available variables because we will iterate on it
#ifdef ENABLE_CUDA_CODE
                // Share current available variables with all other threads so they can select the same variable
                uint32_t l_thread_variable = __shfl_sync(0xFFFFFFFF, p_thread_variable, (int)l_elected_thread);
#else // ENABLE_CUDA_CODE
                uint32_t l_thread_variable = p_thread_variable[l_elected_thread];
#endif // ENABLE_CUDA_CODE

                // Iterate on available variables of elected thread
                do
                {
#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Current available variables : 0x%" PRIx32, l_thread_variable);
#endif // VERBOSITY_LEVEL >= 5

                    // Determine first available variable. Result  cannot be 0 due to ballot
                    unsigned l_bit_index = __ffs((int)l_thread_variable) - 1;

#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Bit index : %i", l_bit_index);
#endif // VERBOSITY_LEVEL >= 5

                    // Set variable bit to zero
                    uint32_t l_mask = ~(1u << l_bit_index);
                    l_thread_variable &= l_mask;

                    p_func(l_elected_thread, l_bit_index, p_init, p_is_position_invalid, p_thread_variable);
                } while(l_thread_variable);
            } while(l_ballot_result);
        }

        inline static
        __device__
        void clear_invalid_bit(CUDA_glutton_max_stack & p_stack
                              ,info_index_t p_info_index
                              ,uint32_t p_elected_thread
                              ,uint32_t p_bit_index
                              )
        {
#ifdef ENABLE_CUDA_CODE
            if(threadIdx.x == p_elected_thread)
#endif // ENABLE_CUDA_CODE
            {
#if VERBOSITY_LEVEL >= 7
                my_cuda::print_single(6, "-> Info %" PRIu32 " Word %" PRIu32 " bit %" PRIu32, static_cast<uint32_t>(p_info_index), p_elected_thread, p_bit_index);
#endif // VERBOSITY_LEVEL >= 7
                p_stack.get_position_info(p_info_index).clear_bit(p_elected_thread, p_bit_index);
            }
        }

        /**
         * Method that will check every transition to determine if they lead to a valid situation or not
         * Transition leading to invalid situations will be removed
         * @param p_stack stack to work on
         * @param p_color_constraints color constraints
         * @return true if positions are still valid despite invalib bits removed
         */
        inline static
        __device__
        bool
        remove_invalid_transitions(CUDA_glutton_max_stack & p_stack
                                  ,const CUDA_color_constraints & p_color_constraints
                                  )
        {
#if VERBOSITY_LEVEL >= 3
            my_cuda::print_single(2,"==> Remove invalid transitions");
#endif // VERBOSITY_LEVEL >= 3
            bool l_invalid_found;
            do
            {
#if VERBOSITY_LEVEL >= 3
                my_cuda::print_single(2,"Start loop");
#endif // VERBOSITY_LEVEL >= 3
                l_invalid_found = false;
                // Iterate on all level position information to check each
                // available transition and remove the one that lead to invalid situation
                for (info_index_t l_info_index{0u};
                     l_info_index < p_stack.get_level_nb_info();
                     ++l_info_index
                    )
                {
#if VERBOSITY_LEVEL >= 4
                    my_cuda::print_single(3,"Info index = %i <=> Position = %i", static_cast<uint32_t>(l_info_index), static_cast<uint32_t>(p_stack.get_position_index(static_cast<info_index_t>(l_info_index))));
#endif // VERBOSITY_LEVEL >= 4

                    auto l_init_lambda = [&]()
                    {
                        p_stack.clear_piece_info();
                    };

                    auto l_lambda = [&](uint32_t p_elected_thread
                                       ,uint32_t p_bit_index
#ifdef ENABLE_CUDA_CODE
                                       ,nvstd::function<void()> p_init
                                       ,nvstd::function<bool(uint32_t)> p_is_position_invalid
                                       ,uint32_t & p_thread_variable
#else
                                       ,std::function<void()> p_init
                                       ,std::function<bool(const pseudo_CUDA_thread_variable<uint32_t> &)> p_is_position_invalid
                                       ,pseudo_CUDA_thread_variable<uint32_t> & p_thread_variable
#endif // ENABLE_CUDA_CODE
                                       )
                    {
                        // Compute piece index
                        uint32_t l_piece_index = CUDA_piece_position_info2::compute_piece_index(p_elected_thread, p_bit_index);

#if VERBOSITY_LEVEL >= 5
                        my_cuda::print_single(4, "Piece index : %i", l_piece_index);
#endif // VERBOSITY_LEVEL >= 5

                        // Piece orientation
                        uint32_t l_piece_orientation = CUDA_piece_position_info2::compute_orientation_index(p_elected_thread, p_bit_index);

#if VERBOSITY_LEVEL >= 5
                        my_cuda::print_single(4, "Piece orientation : %i", l_piece_orientation);
#endif // VERBOSITY_LEVEL >= 5

                        // Get position index corresponding to this info index
                        position_index_t l_position_index = p_stack.get_position_index(static_cast<info_index_t >(l_info_index));

                        unvailability_lock_gard l_lock{p_stack, l_piece_index};

                        l_init_lambda();

#ifdef ENABLE_CUDA_CODE
                        CUDA_glutton_max_stack::t_piece_infos & l_piece_infos = p_stack.get_thread_piece_info();
                        uint32_t l_mask_to_apply = p_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(p_bit_index)): 0xFFFFFFFFu;
#else // ENABLE_CUDA_CODE

                        pseudo_CUDA_thread_variable<uint32_t> l_mask_to_apply{[=](dim3 threadIdx){return p_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(p_bit_index)): 0xFFFFFFFFu;}};
                        pseudo_CUDA_thread_variable<CUDA_glutton_max_stack::t_piece_infos> l_piece_infos{[&](dim3 threadIdx){return p_stack.get_thread_piece_info(threadIdx.x);}};
#endif // ENABLE_CUDA_CODE

                        // Each thread store the related info index corresponding to the orientation index
#ifdef ENABLE_CUDA_CODE
                        unsigned int l_related_thread_index = 0xFFFFFFFFu;
#else // ENABLE_CUDA_CODE
                        pseudo_CUDA_thread_variable<unsigned int> l_related_thread_index = 0xFFFFFFFFu;
#endif // ENABLE_CUDA_CODE

                        auto l_lambda_treat_applied_color = [&](unsigned int p_orientation_index
                                                               ,info_index_t p_related_info_index
                                                               ,uint32_t p_color_id
#ifdef ENABLE_CUDA_CODE
                                                               ,uint32_t p_result_capability
#else // ENABLE_CUDA_CODE
                                                               ,const pseudo_CUDA_thread_variable<uint32_t> & p_result_capability
#endif // ENABLE_CUDA_CODE
                                                               )
                        {
                            // Each thread store the related info index corresponding to the orientation index
#ifdef ENABLE_CUDA_CODE
                            l_related_thread_index = threadIdx.x == p_orientation_index ? static_cast<uint32_t>(p_related_info_index) : l_related_thread_index;
#else // ENABLE_CUDA_CODE
                            l_related_thread_index = [&](dim3 threadIdx) { return threadIdx.x == p_orientation_index ? static_cast<uint32_t>(p_related_info_index): l_related_thread_index[threadIdx.x];};
#endif // ENABLE_CUDA_CODE
                            CUDA_glutton_max::analyze_info(p_result_capability, l_piece_infos);
                        };

                        // Apply color constraint
#if VERBOSITY_LEVEL >= 5
                        my_cuda::print_single(4, "Apply color constraints");
#endif // VERBOSITY_LEVEL >= 5
                        if(CUDA_glutton_max::apply_color_constraints(l_piece_index, l_piece_orientation, l_position_index, p_stack, p_color_constraints, l_mask_to_apply, CUDA_glutton_max::is_position_invalid, l_lambda_treat_applied_color))
                        {
                            l_invalid_found = true;
                            clear_invalid_bit(p_stack, l_info_index, p_elected_thread, p_bit_index);
                            return;
                        }

                        auto l_lamda_do_apply = [&](info_index_t p_result_info_index) -> bool
                        {
                            // Apply only on positions that have not received color constraints
#ifdef ENABLE_CUDA_CODE
                            return __all_sync(0xFFFFFFFFu, p_result_info_index != info_index_t(l_related_thread_index));
#else // ENABLE_CUDA_CODE
                            bool l_all = true;
                            for (unsigned int l_threadIdx_x = 0; l_all && l_threadIdx_x < 32;++l_threadIdx_x)
                            {
                                l_all = l_all && (p_result_info_index != info_index_t(l_related_thread_index[l_threadIdx_x]));
                            }
                            return l_all;
#endif // ENABLE_CUDA_CODE
                        };

                        auto l_lambda_treat_simple_mask = [&](info_index_t p_result_info_index
#ifdef ENABLE_CUDA_CODE
                                                             ,uint32_t p_capability
                                                             ,uint32_t p_result_capability
#else // ENABLE_CUDA_CODE
                                                             ,const pseudo_CUDA_thread_variable<uint32_t> & p_capability
                                                             ,const pseudo_CUDA_thread_variable<uint32_t> & p_result_capability
#endif // ENABLE_CUDA_CODE
                                                             )
                        {
                            CUDA_glutton_max::analyze_info(p_result_capability, l_piece_infos);
                        };

                        // This is reached only if no invalid position was detected in the previous loop
#if VERBOSITY_LEVEL >= 5
                        my_cuda::print_single(4, "Apply piece constraints before selected index");
#endif // VERBOSITY_LEVEL >= 5
                        if(CUDA_glutton_max::apply_simple_mask(static_cast<info_index_t>(0u), l_info_index, p_stack, l_mask_to_apply, l_lamda_do_apply, p_is_position_invalid, l_lambda_treat_simple_mask))
                        {
                            l_invalid_found = true;
                            clear_invalid_bit(p_stack, l_info_index, p_elected_thread, p_bit_index);
                            return ;
                        }

                        // This is reached only if no invalid position was detected in the previous loop
#if VERBOSITY_LEVEL >= 5
                        my_cuda::print_single(4, "Apply piece constraints after selected index");
#endif // VERBOSITY_LEVEL >= 5
                        if(CUDA_glutton_max::apply_simple_mask(l_info_index + static_cast<uint32_t>(1u), p_stack.get_level_nb_info(), p_stack, l_mask_to_apply, l_lamda_do_apply, p_is_position_invalid, l_lambda_treat_simple_mask))
                        {
                            l_invalid_found = true;
                            clear_invalid_bit(p_stack, l_info_index, p_elected_thread, p_bit_index);
                            return ;
                        }

                        // This is reached only if no invalid position was detected in the previous loop
                        // Manage pieces info
#if VERBOSITY_LEVEL >= 5
                        my_cuda::print_single(4, "Check pieces info");
#endif // VERBOSITY_LEVEL >= 5
                        for(unsigned int l_piece_info_index = 0; l_piece_info_index < 8; ++l_piece_info_index)
                        {
#ifdef ENABLE_CUDA_CODE
                            CUDA_glutton_max_stack::t_piece_info l_piece_info = l_piece_infos[l_piece_info_index];
#else // ENABLE_CUDA_CODE
                            pseudo_CUDA_thread_variable<CUDA_glutton_max_stack::t_piece_info> l_piece_info{[&](dim3 threadIdx){return l_piece_infos[threadIdx.x][l_piece_info_index];}};
#endif // ENABLE_CUDA_CODE
                            if(!__all_sync(0xFFFFFFFFu, l_piece_info))
                            {
                                l_invalid_found = true;
#if VERBOSITY_LEVEL >= 5
                                my_cuda::print_single(4, "INVALID PIECES:\n");
#endif // VERBOSITY_LEVEL >= 5
#if VERBOSITY_LEVEL >= 6
                                CUDA_glutton_max::debug_message_pieces(l_piece_info_index, l_piece_info);
#endif // VERBOSITY_LEVEL >= 6
                                clear_invalid_bit(p_stack, l_info_index, p_elected_thread, p_bit_index);
                                return;
                            }
                        }
                    };

                    // Each thread get its word in position info
#ifdef ENABLE_CUDA_CODE
                    uint32_t l_thread_available_variables = p_stack.get_position_info(info_index_t(l_info_index)).get_word(threadIdx.x);
#else // ENABLE_CUDA_CODE
                    pseudo_CUDA_thread_variable<uint32_t> l_thread_available_variables{[&](dim3 threadIdx){return p_stack.get_position_info(info_index_t(l_info_index)).get_word(threadIdx.x);}};
#endif // ENABLE_CUDA_CODE

                    CUDA_glutton_max::warp_iterate(l_thread_available_variables, l_lambda, l_init_lambda, CUDA_glutton_max::is_position_invalid);

                    if(!p_stack.is_position_valid(l_info_index))
                     {
#if VERBOSITY_LEVEL >= 3
                        my_cuda::print_single(2, "INFO %" PRIu32 " completely cleared !", static_cast<uint32_t>(l_info_index));
#endif // VERBOSITY_LEVEL >= 3
                        //exit(-1);
                        return false;
                     }
                }

            }
            while(l_invalid_found);
            return true;
        }

        //-------------------------------------------------------------------------
        inline static
        __device__
        uint32_t
        generate_best_info(info_index_t p_info_index
                          ,unsigned int p_elected_thread
                          ,unsigned int p_bit_index
                          )
        {
            assert(p_info_index < 256);
            assert(p_elected_thread < 32);
            assert(p_bit_index < 32);
            return (p_elected_thread << 16u) | (p_bit_index << 8u) | static_cast<uint32_t>(p_info_index);
        }

        //-------------------------------------------------------------------------
        inline static
#ifdef ENABLE_CUDA_CODE
        __device__
        void
        decode_best_info(uint32_t p_best_info
                        ,info_index_t & p_info_index
                        ,unsigned int & p_elected_thread
                        ,unsigned int & p_bit_index
                        )
        {
            p_info_index = static_cast<info_index_t >(p_best_info & 0xFFu);
            p_bit_index = (p_best_info >> 8u) & 0xFFu;
            assert(p_bit_index < 32);
            p_elected_thread = p_best_info >> 16u;
            assert(p_elected_thread < 32);
        }
#else // ENABLE_CUDA_CODE
        std::tuple<info_index_t, unsigned int, unsigned int>
        decode_best_info(uint32_t p_best_info)
        {
            info_index_t l_info_index = static_cast<info_index_t >(p_best_info & 0xFFu);
            unsigned int l_bit_index = (p_best_info >> 8u) & 0xFFu;
            assert(l_bit_index < 32);
            unsigned int l_elected_thread = p_best_info >> 16u;
            assert(l_elected_thread < 32);
            return {l_info_index, l_elected_thread, l_bit_index};
        }
#endif // ENABLE_CUDA_CODE

        /**
         * Method iterating on available transitions to determine the best candidate
         * @param p_stack
         * @param p_color_constraints reference on array containing colour constraints
         * @return information related to best transition found
         */
        inline static
        __device__
        uint32_t
        get_best_candidate(CUDA_glutton_max_stack & p_stack
                          ,const CUDA_color_constraints & p_color_constraints
                          )
        {
            uint32_t l_best_information = 0xFFFFFFFFu;
            uint32_t l_best_total_score = 0;
            uint32_t l_best_min_max_score = 0;

            // Iterate on all level position information to compute the score of each available transition
            for(info_index_t l_info_index{0u};
                l_info_index < p_stack.get_level_nb_info();
                ++l_info_index
               )
            {
#if VERBOSITY_LEVEL >= 4
                my_cuda::print_single(3,"Info index = %i <=> Position = %i", static_cast<uint32_t>(l_info_index), static_cast<uint32_t>(p_stack.get_position_index(static_cast<info_index_t>(l_info_index))));
#endif // VERBOSITY_LEVEL >= 4

                // Each thread get its word in position info
#ifdef ENABLE_CUDA_CODE
                uint32_t l_thread_available_variables = p_stack.get_position_info(info_index_t(l_info_index)).get_word(threadIdx.x);
#else // ENABLE_CUDA_CODE
                pseudo_CUDA_thread_variable<uint32_t> l_thread_available_variables{[&](dim3 threadIdx){return p_stack.get_position_info(info_index_t(l_info_index)).get_word(threadIdx.x);}};
#endif // ENABLE_CUDA_CODE

                uint32_t l_info_bits_min = 0xFFFFFFFFu;
                uint32_t l_info_bits_max = 0;
                uint32_t l_info_bits_total = 0;

                auto l_init_lambda = [&]()
                {
                    l_info_bits_min = 0xFFFFFFFFu;
                    l_info_bits_max = 0;
                    l_info_bits_total = 0;

                    p_stack.clear_piece_info();
                };

                auto l_lambda = [&](uint32_t p_elected_thread
                                   ,uint32_t p_bit_index
#ifdef ENABLE_CUDA_CODE
                                   ,nvstd::function<void()> p_init
                                   ,nvstd::function<bool(uint32_t)> p_is_position_invalid
                                   ,uint32_t & p_thread_variable
#else
                                   ,std::function<void()> p_init
                                   ,std::function<bool(const pseudo_CUDA_thread_variable<uint32_t> &)> p_is_position_invalid
                                   ,pseudo_CUDA_thread_variable<uint32_t> & p_thread_variable
#endif // ENABLE_CUDA_CODE
                                   )
                {
                    // Compute piece index
                    uint32_t l_piece_index = CUDA_piece_position_info2::compute_piece_index(p_elected_thread, p_bit_index);

#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Piece index : %i", l_piece_index);
#endif // VERBOSITY_LEVEL >= 4

                    // Piece orientation
                    uint32_t l_piece_orientation = CUDA_piece_position_info2::compute_orientation_index(p_elected_thread, p_bit_index);

#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Piece orientation : %i", l_piece_orientation);
#endif // VERBOSITY_LEVEL >= 4

                    // Get position index corresponding to this info index
                    position_index_t l_position_index = p_stack.get_position_index(static_cast<info_index_t >(l_info_index));

                    unvailability_lock_gard l_lock{p_stack, l_piece_index};

                    l_init_lambda();

#ifdef ENABLE_CUDA_CODE
                    CUDA_glutton_max_stack::t_piece_infos & l_piece_infos = p_stack.get_thread_piece_info();
                    uint32_t l_mask_to_apply = p_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(p_bit_index)): 0xFFFFFFFFu;
#else // ENABLE_CUDA_CODE
                    pseudo_CUDA_thread_variable<uint32_t> l_mask_to_apply{[=](dim3 threadIdx){return p_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(p_bit_index)): 0xFFFFFFFFu;}};
                    pseudo_CUDA_thread_variable<CUDA_glutton_max_stack::t_piece_infos> l_piece_infos{[&](dim3 threadIdx){return p_stack.get_thread_piece_info(threadIdx.x);}};
#endif // ENABLE_CUDA_CODE

                    // Each thread store the related info index corresponding to the orientation index
#ifdef ENABLE_CUDA_CODE
                    unsigned int l_related_thread_index = 0xFFFFFFFFu;
#else // ENABLE_CUDA_CODE
                    pseudo_CUDA_thread_variable<unsigned int> l_related_thread_index = 0xFFFFFFFFu;
#endif // ENABLE_CUDA_CODE

                    auto l_lambda_treat_applied_color = [&](unsigned int p_orientation_index
                                                           ,info_index_t p_related_info_index
                                                           ,uint32_t p_color_id
#ifdef ENABLE_CUDA_CODE
                                                           ,uint32_t p_result_capability
#else // ENABLE_CUDA_CODE
                                                           ,const pseudo_CUDA_thread_variable<uint32_t> p_result_capability
#endif // ENABLE_CUDA_CODE
                                                           )
                    {
                        // Each thread store the related info index corresponding to the orientation index
#ifdef ENABLE_CUDA_CODE
                        l_related_thread_index = threadIdx.x == p_orientation_index ? static_cast<uint32_t>(p_related_info_index) : l_related_thread_index;
#else // ENABLE_CUDA_CODE
                        l_related_thread_index = [&](dim3 threadIdx) { return threadIdx.x == p_orientation_index ? static_cast<uint32_t>(p_related_info_index): l_related_thread_index[threadIdx.x];};
#endif // ENABLE_CUDA_CODE
                        CUDA_glutton_max::analyze_info(p_result_capability, l_piece_infos);
                        CUDA_glutton_max::count_result_nb_bits(p_result_capability, l_info_bits_min, l_info_bits_max, l_info_bits_total);
#if VERBOSITY_LEVEL >= 6
                        CUDA_glutton_max::debug_message_info_bits(5, l_info_bits_min, l_info_bits_max, l_info_bits_total);
#endif // VERBOSITY_LEVEL >= 6
                    };

                    // Apply color constraint
#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Apply color constraints");
#endif // VERBOSITY_LEVEL >= 5
                    if(CUDA_glutton_max::apply_color_constraints(l_piece_index, l_piece_orientation, l_position_index, p_stack, p_color_constraints, l_mask_to_apply, CUDA_glutton_max::is_position_invalid, l_lambda_treat_applied_color))
                    {
                        my_cuda::print_single(1, "SHOULD NOT BE REACHED 1");
#ifndef ENABLE_CUDA_CODE
                        exit(-1);
#endif // ENABLE_CUDA_CODE
                        return;
                    }

                    auto l_lamda_do_apply = [&](info_index_t p_result_info_index) -> bool
                    {
#ifdef ENABLE_CUDA_CODE
                        return __all_sync(0xFFFFFFFFu, p_result_info_index != info_index_t(l_related_thread_index));
#else // ENABLE_CUDA_CODE
                        bool l_all = true;
                        for (unsigned int l_threadIdx_x = 0; l_all && l_threadIdx_x < 32;++l_threadIdx_x)
                        {
                            l_all = l_all && (p_result_info_index != info_index_t(l_related_thread_index[l_threadIdx_x]));
                        }
                        return l_all;
#endif // ENABLE_CUDA_CODE
                    };

                    auto l_lambda_treat_simple_mask = [&](info_index_t p_result_info_index
#ifdef ENABLE_CUDA_CODE
                                                         ,uint32_t p_capability
                                                         ,uint32_t p_result_capability
#else // ENABLE_CUDA_CODE
                                                         ,const pseudo_CUDA_thread_variable<uint32_t> & p_capability
                                                         ,const pseudo_CUDA_thread_variable<uint32_t> & p_result_capability
#endif // ENABLE_CUDA_CODE
                                                         )
                    {
                        CUDA_glutton_max::analyze_info(p_result_capability, l_piece_infos);
                        CUDA_glutton_max::count_result_nb_bits(p_result_capability, l_info_bits_min, l_info_bits_max, l_info_bits_total);
#if VERBOSITY_LEVEL >= 6
                        CUDA_glutton_max::debug_message_info_bits(5, l_info_bits_min, l_info_bits_max, l_info_bits_total);
#endif // VERBOSITY_LEVEL >= 6
                    };

                    // This is reached only if no invalid position was detected in the previous loop
#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Apply piece constraints before selected index");
#endif // VERBOSITY_LEVEL >= 5
                    if(CUDA_glutton_max::apply_simple_mask(static_cast<info_index_t>(0u), l_info_index, p_stack, l_mask_to_apply, l_lamda_do_apply, p_is_position_invalid, l_lambda_treat_simple_mask))
                    {
                        my_cuda::print_single(1, "SHOULD NOT BE REACHED 2");
#ifndef ENABLE_CUDA_CODE
                        exit(-1);
#endif // ENABLE_CUDA_CODE
                        return ;
                    }

                    // This is reached only if no invalid position was detected in the previous loop
#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Apply piece constraints after selected index");
#endif // VERBOSITY_LEVEL >= 5
                    if(CUDA_glutton_max::apply_simple_mask(l_info_index + static_cast<uint32_t>(1u), p_stack.get_level_nb_info(), p_stack, l_mask_to_apply, l_lamda_do_apply, p_is_position_invalid, l_lambda_treat_simple_mask))
                    {
                        my_cuda::print_single(1, "SHOULD NOT BE REACHED 3");
#ifndef ENABLE_CUDA_CODE
                        exit(-1);
#endif // ENABLE_CUDA_CODE
                        return ;
                    }

                    // This is reached only if no invalid position was detected in the previous loop
                    // Manage pieces info
#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Compute pieces info");
#endif // VERBOSITY_LEVEL >= 5

#ifdef ENABLE_CUDA_CODE
                    uint32_t l_piece_info_total_bit = 0;
                    uint32_t l_piece_info_min_bits = 0xFFFFFFFFu;
                    uint32_t l_piece_info_max_bits = 0;
#else // ENABLE_CUDA_CODE
                    pseudo_CUDA_thread_variable<uint32_t> l_piece_info_total_bit = 0;
                    pseudo_CUDA_thread_variable<uint32_t> l_piece_info_min_bits = 0xFFFFFFFFu;
                    pseudo_CUDA_thread_variable<uint32_t> l_piece_info_max_bits = 0;
#endif // ENABLE_CUDA_CODE
                    for(unsigned int l_piece_info_index = 0; l_piece_info_index < 8; ++l_piece_info_index)
                    {
#ifdef ENABLE_CUDA_CODE
                        CUDA_glutton_max_stack::t_piece_info l_piece_info = l_piece_infos[l_piece_info_index];
#else // ENABLE_CUDA_CODE
                        pseudo_CUDA_thread_variable<CUDA_glutton_max_stack::t_piece_info> l_piece_info{[&](dim3 threadIdx){return l_piece_infos[threadIdx.x][l_piece_info_index];}};
#endif // ENABLE_CUDA_CODE
                        if(__all_sync(0xFFFFFFFFu, l_piece_info))
                        {
#ifdef ENABLE_CUDA_CODE
                            unsigned int l_info_piece_index = 8 * threadIdx.x + l_piece_info_index;
#else // ENABLE_CUDA_CODE
                            pseudo_CUDA_thread_variable<unsigned int> l_info_piece_index{[=](dim3 threadIdx){return 8 * threadIdx.x + l_piece_info_index;}};
#endif // ENABLE_CUDA_CODE
#ifdef ENABLE_CUDA_CODE
                            if(p_stack.is_piece_available(l_info_piece_index))
#else // ENABLE_CUDA_CODE
                            for (unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32;++l_threadIdx_x)
                            {
                                if (p_stack.is_piece_available(l_info_piece_index[l_threadIdx_x]))
#endif // ENABLE_CUDA_CODE
                                {
#ifdef ENABLE_CUDA_CODE
                                    CUDA_glutton_max::update_stats(l_piece_info, l_piece_info_min_bits, l_piece_info_max_bits, l_piece_info_total_bit);
#else // ENABLE_CUDA_CODE
                                    CUDA_glutton_max::update_stats(l_piece_info[l_threadIdx_x], l_piece_info_min_bits[l_threadIdx_x], l_piece_info_max_bits[l_threadIdx_x], l_piece_info_total_bit[l_threadIdx_x]);
#endif // ENABLE_CUDA_CODE
#if VERBOSITY_LEVEL >= 6
#ifdef ENABLE_CUDA_CODE
                                    my_cuda::print_all(5, "Piece %i:\nMin %3i\tMax %3i\tTotal %i\n", l_info_piece_index, l_piece_info_min_bits, l_piece_info_max_bits, l_piece_info_total_bit);
#else // ENABLE_CUDA_CODE
                                    my_cuda::print_all(5, {l_threadIdx_x, 1, 1}, "Piece %i:\nMin %3i\tMax %3i\tTotal %i\n", l_info_piece_index[l_threadIdx_x], l_piece_info_min_bits[l_threadIdx_x], l_piece_info_max_bits[l_threadIdx_x], l_piece_info_total_bit[l_threadIdx_x]);
#endif // ENABLE_CUDA_CODE
#endif // VERBOSITY_LEVEL >= 6
                                }
#ifndef ENABLE_CUDA_CODE
                            }
#endif // ENABLE_CUDA_CODE
                        }
                        else
                        {
#if VERBOSITY_LEVEL >= 5
                            my_cuda::print_single(4, "INVALID PIECES:\n");
#endif // VERBOSITY_LEVEL >= 5
#if VERBOSITY_LEVEL >= 6
                            CUDA_glutton_max::debug_message_pieces(l_piece_info_index, l_piece_info);
#endif // VERBOSITY_LEVEL >= 6
                            my_cuda::print_single(1, "SHOULD NOT BE REACHED 4");
#ifndef ENABLE_CUDA_CODE
                            exit(-1);
#endif // ENABLE_CUDA_CODE
                            return;
                        }
                    }
                    // This is reached only if no invalid position was detected in the previous loop
                    l_info_bits_total += my_cuda::reduce_add_sync(l_piece_info_total_bit);
#ifdef ENABLE_CUDA_CODE
                    l_piece_info_min_bits = my_cuda::reduce_min_sync(l_piece_info_min_bits);
                    l_info_bits_min = l_piece_info_min_bits < l_info_bits_min ? l_piece_info_min_bits : l_info_bits_min;
#else // ENABLE_CUDA_CODE
                    my_cuda::reduce_min_sync(l_piece_info_min_bits);
                    l_info_bits_min = l_piece_info_min_bits[0] < l_info_bits_min ? l_piece_info_min_bits[0] : l_info_bits_min;
#endif // ENABLE_CUDA_CODE

#ifdef ENABLE_CUDA_CODE
                    l_piece_info_max_bits = my_cuda::reduce_max_sync(l_piece_info_max_bits);
                    l_info_bits_max = l_piece_info_max_bits > l_info_bits_max ? l_piece_info_max_bits : l_info_bits_max;
#else // ENABLE_CUDA_CODE
                    my_cuda::reduce_max_sync(l_piece_info_max_bits);
                    l_info_bits_max = l_piece_info_max_bits[0] > l_info_bits_max ? l_piece_info_max_bits[0] : l_info_bits_max;
#endif // ENABLE_CUDA_CODE
#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "After reduction");
                    my_cuda::print_single(4, "Min %3i\tMax %3i\tTotal %i\n", l_info_bits_min, l_info_bits_max, l_info_bits_total);
#endif // VERBOSITY_LEVEL >= 5

                    // compare with global stats
                    uint32_t l_min_max_score = (l_info_bits_max << 16u) + l_info_bits_min;
#if VERBOSITY_LEVEL >= 5
                    my_cuda::print_single(4, "Total %i\tMinMax %i\n", l_info_bits_total, l_min_max_score);
#endif // VERBOSITY_LEVEL >= 5
                    if(l_info_bits_total > l_best_total_score || (l_info_bits_total == l_best_total_score && l_min_max_score > l_best_min_max_score) || p_stack.get_level() == p_stack.get_size() - 1)
                    {
#if VERBOSITY_LEVEL >= 5
                        my_cuda::print_single(4, "New best score Total %i MinMax %i\n", l_info_bits_total, l_min_max_score);
#endif // VERBOSITY_LEVEL >= 5
                        l_best_total_score = l_info_bits_total;
                        l_best_min_max_score = l_min_max_score;
                        // Store transition characteristics
                        l_best_information = CUDA_glutton_max::generate_best_info(l_info_index, p_elected_thread, p_bit_index);
                    }
                };

                CUDA_glutton_max::warp_iterate(l_thread_available_variables, l_lambda, l_init_lambda, CUDA_glutton_max::is_position_invalid);
            } // Position iteration to compute best score
            return l_best_information;
        }

        inline static
        __device__
        void
        apply_best_candidate(uint32_t p_best_information
                            ,CUDA_glutton_max_stack & p_stack
                            ,const CUDA_color_constraints & p_color_constraints
                            )
        {
            // Apply best candidate
#if VERBOSITY_LEVEL >= 6
            print_position_info(5, p_stack);
#endif // VERBOSITY_LEVEL >= 6

            info_index_t l_info_index{0};
            unsigned int l_elected_thread;
            unsigned int l_bit_index;
#ifdef ENABLE_CUDA_CODE
            decode_best_info(p_best_information, l_info_index, l_elected_thread, l_bit_index);
#else // ENABLE_CUDA_CODE
            std::tie(l_info_index, l_elected_thread, l_bit_index) = decode_best_info(p_best_information);
#endif // ENABLE_CUDA_CODE

#if VERBOSITY_LEVEL >= 4
            my_cuda::print_single(3, "Info index %i Elected thread %i Bit index : %i", static_cast<uint32_t>(l_info_index), l_elected_thread, l_bit_index);
#endif // VERBOSITY_LEVEL >= 4

            // Set variable bit to zero in best candidate and current info
            CUDA_piece_position_info2 & l_position_info = p_stack.get_position_info(l_info_index);
            l_position_info.clear_bit(l_elected_thread, l_bit_index);

#if VERBOSITY_LEVEL >= 6
            my_cuda::print_single(5, "after clear\n");
            print_position_info(5, p_stack);
#endif // VERBOSITY_LEVEL >= 6

            // Compute piece index
            uint32_t l_piece_index = CUDA_piece_position_info2::compute_piece_index(l_elected_thread, l_bit_index);

#if VERBOSITY_LEVEL >= 5
            my_cuda::print_single(4, "Piece index : %i", l_piece_index);
#endif // VERBOSITY_LEVEL >= 5

            // Piece orientation
            uint32_t l_piece_orientation = CUDA_piece_position_info2::compute_orientation_index(l_elected_thread, l_bit_index);

#if VERBOSITY_LEVEL >= 5
            my_cuda::print_single(4, "Piece orientation : %i", l_piece_orientation);
#endif // VERBOSITY_LEVEL >= 5

            // Get position index corresponding to this info index
            position_index_t l_position_index = p_stack.get_position_index(l_info_index);

            // Compute mask to apply which set piece bit to 0
#ifdef ENABLE_CUDA_CODE
            uint32_t l_mask_to_apply = l_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(l_bit_index)): 0xFFFFFFFFu;
#else // ENABLE_CUDA_CODE
            pseudo_CUDA_thread_variable<uint32_t> l_mask_to_apply{[=](dim3 threadIdx){return l_elected_thread == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(l_bit_index)) : 0xFFFFFFFFu;}};
#endif // ENABLE_CUDA_CODE

            if (compute_next_level_position_info(info_index_t(0u), l_info_index, p_stack, l_mask_to_apply))
            {
                my_cuda::print_single(4, "SHOULD NOT BE REACHED 6");
#ifndef ENABLE_CUDA_CODE
                exit(-1);
#endif // ENABLE_CUDA_CODE
            }

            // Last position is not treated here because next level has 1 position less
            if (compute_next_level_position_info(l_info_index + 1, p_stack.get_level_nb_info() - 1, p_stack, l_mask_to_apply))
            {
                my_cuda::print_single(4, "SHOULD NOT BE REACHED 7");
#ifndef ENABLE_CUDA_CODE
                exit(-1);
#endif // ENABLE_CUDA_CODE
            }

            // No next level when we set latest piece
            if (l_info_index < (p_stack.get_level_nb_info() - 1) && p_stack.get_level() < (p_stack.get_size() - 1))
            {
                // Last position in next level it will be located at l_best_candidate_index
#if VERBOSITY_LEVEL >= 5
                my_cuda::print_single(4, "Info %i -> %i:\n", static_cast<uint32_t>(p_stack.get_level_nb_info()) - 1, static_cast<uint32_t>(l_info_index));
#endif // VERBOSITY_LEVEL >= 5
#ifdef ENABLE_CUDA_CODE
                uint32_t l_capability = p_stack.get_position_info(info_index_t(p_stack.get_level_nb_info() - 1)).get_word(threadIdx.x);
                uint32_t l_constraint = l_mask_to_apply;
                uint32_t l_result = l_capability & l_constraint;
#if VERBOSITY_LEVEL >= 6
                my_cuda::print_mask(5, __ballot_sync(0xFFFFFFFFu, l_capability), "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability, l_constraint, l_result);
#endif // VERBOSITY_LEVEL >= 6
                p_stack.get_next_level_position_info(l_info_index).set_word(threadIdx.x , l_result);
                if(__any_sync(0xFFFFFFFFu, l_result))
                {
                    my_cuda::print_single(2, "INVALID Best");
                    my_cuda::print_single(1, "SHOULD NOT BE REACHED 8");
                    return;
                }
#else // ENABLE_CUDA_CODE
                pseudo_CUDA_thread_variable<uint32_t> l_capability{[&](dim3 threadIdx){ return p_stack.get_position_info(info_index_t(p_stack.get_level_nb_info() - 1)).get_word(threadIdx.x);}};
                pseudo_CUDA_thread_variable<uint32_t> l_result = l_capability & l_mask_to_apply;
#if VERBOSITY_LEVEL >= 6
                uint32_t l_print_mask = __ballot_sync(0xFFFFFFFFu, l_capability);
#endif // VERBOSITY_LEVEL >= 6
                for (dim3 threadIdx{0, 1, 1}; threadIdx.x < 32; ++threadIdx.x)
                {
#if VERBOSITY_LEVEL >= 6
                    my_cuda::print_mask(5, l_print_mask, threadIdx, "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability[threadIdx.x], l_mask_to_apply[threadIdx.x], l_result[threadIdx.x]);
#endif // VERBOSITY_LEVEL >= 6
                    p_stack.get_next_level_position_info(l_info_index).set_word(threadIdx.x, l_result[threadIdx.x]);
                }
                if(!__any_sync(0xFFFFFFFFu, l_result))
                {
                    my_cuda::print_single(4, "INVALID Best");
                    my_cuda::print_single(4, "SHOULD NOT BE REACHED 8");
                    exit(-1);
                }
#endif // ENABLE_CUDA_CODE
            }

#if VERBOSITY_LEVEL >= 6
            print_device_info_position_index(0, p_stack);
#endif // VERBOSITY_LEVEL >= 6
            p_stack.push(l_info_index, l_position_index, l_piece_index, l_piece_orientation);
#if VERBOSITY_LEVEL >= 6
            print_device_info_position_index(0, p_stack);
#endif // VERBOSITY_LEVEL >= 6

            bool l_invalid = false;

            auto l_lambda_treat_applied_color = [&](unsigned int p_orientation_index
                                                   ,info_index_t p_related_info_index
                                                   ,uint32_t p_color_id
#ifdef ENABLE_CUDA_CODE
                                                   ,uint32_t p_result_capability
#else // ENABLE_CUDA_CODE
                                                   ,const pseudo_CUDA_thread_variable<uint32_t> p_result_capability
#endif // ENABLE_CUDA_CODE
                                                   )
            {
                // If related index correspond to last position of previous level ( we already did the push ) than result is stored in position where we store the piece
                info_index_t l_related_target_info_index = p_related_info_index < p_stack.get_level_nb_info() ? p_related_info_index : l_info_index;

#if VERBOSITY_LEVEL >= 5
                my_cuda::print_single(4, "Color Info %i -> %i:\n", static_cast<uint32_t>(p_related_info_index), static_cast<uint32_t>(l_related_target_info_index));
#endif // VERBOSITY_LEVEL >= 5

                p_stack.get_position_info(l_related_target_info_index).CUDA_and(p_stack.get_position_info(p_related_info_index), p_color_constraints.get_info(p_color_id - 1, p_orientation_index));
                if (!p_stack.is_position_valid(l_related_target_info_index))
                {
                    my_cuda::print_single(5, "INVALID Best color");
                    l_invalid = true;
                    my_cuda::print_single(5, "SHOULD NOT BE REACHED 9");
#ifndef ENABLE_CUDA_CODE
                    exit(-1);
#endif // ENABLE_CUDA_CODE
                }
            };

            // Apply color constraint
            l_invalid = apply_color_constraints(l_piece_index, l_piece_orientation, l_position_index, p_stack, p_color_constraints, 0xFFFFFFFFu, is_position_invalid, l_lambda_treat_applied_color);

            // Check if we exit from loop because everything was fine or due to invalid position
            if (l_invalid)
            {
                my_cuda::print_single(3, "SHOULD NOT BE REACHED 10");
#ifndef ENABLE_CUDA_CODE
                exit(-1);
#endif // ENABLE_CUDA_CODE
                // Restore stack to continue to iterate on best candidates
                print_device_info_position_index(0, p_stack);
                p_stack.pop();
                print_device_info_position_index(0, p_stack);
            }
        }

        inline static
        __device__
        void
        debug_message_pieces(uint32_t p_piece_info_index
#ifdef ENABLE_CUDA_CODE
                            ,CUDA_glutton_max_stack::t_piece_info p_piece_info
#else // ENABLE_CUDA_CODE
                            ,const pseudo_CUDA_thread_variable<CUDA_glutton_max_stack::t_piece_info> & p_piece_info
#endif // ENABLE_CUDA_CODE
                            )
        {
#ifdef ENABLE_CUDA_CODE
            my_cuda::print_mask(5, __ballot_sync(0xFFFFFFFFu, !p_piece_info), "Piece info[%" PRIu32 "] : %" PRIu32 "\n", 8 * threadIdx.x + p_piece_info_index, p_piece_info);
#else // ENABLE_CUDA_CODE
            uint32_t l_mask_print = __ballot_sync(0xFFFFFFFFu, [&](dim3 threadIdx){return !p_piece_info[threadIdx.x];});
            for (unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32;++l_threadIdx_x)
            {
                my_cuda::print_mask(5, l_mask_print, {l_threadIdx_x, 1, 1}, "Piece info[%" PRIu32 "] : %" PRIu32 "\n", 8 * l_threadIdx_x + p_piece_info_index, p_piece_info[l_threadIdx_x]);
            }
#endif // ENABLE_CUDA_CODE

        }

        inline static
        __device__
        void
        debug_message_info_bits(unsigned int p_level
                               ,uint32_t p_info_bits_min
                               ,uint32_t p_info_bits_max
                               ,uint32_t p_info_bits_total
                               )
        {
            my_cuda::print_single(p_level, "Min %3i\tMax %3i\tTotal %i\n", p_info_bits_min, p_info_bits_max, p_info_bits_total);
        }

        /**
         * CPU debug version of CUDA algorithm
         */
        void run();

      private:

        const emp_piece_db & m_piece_db;
        const emp_FSM_info & m_info;

    };


    __global__
    void kernel(CUDA_glutton_max_stack * p_stacks
               ,unsigned int p_nb_stack
               ,const CUDA_color_constraints & p_color_constraints
               )
    {
#ifdef ENABLE_CUDA_CODE
        assert(warpSize == blockDim.x);

        unsigned int l_stack_index = threadIdx.y + blockIdx.x * blockDim.y;
#else // ENABLE_CUDA_CODE
        unsigned int l_stack_index = 0;
        bool l_toc = false;
#endif // ENABLE_CUDA_CODE

        if(l_stack_index >= p_nb_stack)
        {
            return;
        }

        CUDA_glutton_max_stack & l_stack = p_stacks[l_stack_index];

        bool l_new_level = true;
#if VERBOSITY_LEVEL >= 1
        uint32_t l_step = 0xFFFFFFFFu;
#endif // VERBOSITY_LEVEL
        info_index_t l_pop_index = static_cast<info_index_t >(0);
        while(l_stack.get_level() < l_stack.get_size())
        {
#if VERBOSITY_LEVEL >= 1
            ++l_step;
            my_cuda::print_single(0,"Stack level = %i [%i]", l_stack.get_level(), l_step);
#ifndef ENABLE_CUDA_CODE
            if(!(l_step & 0x3FFu))
            {
                CUDA_glutton_stack_XML_converter l_dumper(l_toc ? "stack_toc.xml" : "stack_tic.xml");
                l_dumper.dump(l_stack);
                l_toc = !l_toc;
            }
#endif // ENABLE_CUDA_CODE
#endif // VERBOSITY_LEVEL >= 1

            if((!l_new_level) && (!l_stack.is_position_valid(l_pop_index)))
            {
#if VERBOSITY_LEVEL >= 1
                 my_cuda::print_single(0, "No more remaining bit in this index %i position %i, go up from one level", static_cast<uint32_t>(l_pop_index), static_cast<uint32_t>(l_stack.get_position_index(l_pop_index)));
#endif // VERBOSITY_LEVEL >= 1
#if VERBOSITY_LEVEL >= 2
                 CUDA_glutton_max::print_device_info_position_index(0, l_stack);
#endif // VERBOSITY_LEVEL >= 2
                 l_pop_index = l_stack.pop();
#if VERBOSITY_LEVEL >= 2
                 CUDA_glutton_max::print_device_info_position_index(0, l_stack);
#endif // VERBOSITY_LEVEL >= 2
                 l_new_level = false;
                 continue;
            }

#if VERBOSITY_LEVEL >= 2
            my_cuda::print_single(1,"Remove invalid bits");
#endif // VERBOSITY_LEVEL >= 2
            if(!CUDA_glutton_max::remove_invalid_transitions(l_stack, p_color_constraints))
            {
#if VERBOSITY_LEVEL >= 1
                my_cuda::print_single(0, "No best score found, go up from one level");
#endif // VERBOSITY_LEVEL >= 1
#if VERBOSITY_LEVEL >= 2
                CUDA_glutton_max::print_device_info_position_index(0, l_stack);
#endif // VERBOSITY_LEVEL >= 2
                l_pop_index = l_stack.pop();
#if VERBOSITY_LEVEL >= 2
                CUDA_glutton_max::print_device_info_position_index(0, l_stack);
#endif // VERBOSITY_LEVEL >= 2
                l_new_level = false;
                continue;
            }

#if VERBOSITY_LEVEL >= 2
            my_cuda::print_single(1,"Search for best score");
#endif // VERBOSITY_LEVEL >= 2
            uint32_t l_best_information = CUDA_glutton_max::get_best_candidate(l_stack, p_color_constraints);

            // If no best score found there is no interesting transition so go back
            if(0xFFFFFFFFu == l_best_information)
            {
#if VERBOSITY_LEVEL >= 1
                my_cuda::print_single(0, "No best score found, go up from one level");
#endif // VERBOSITY_LEVEL >= 1
                CUDA_glutton_max::print_device_info_position_index(0, l_stack);
                l_pop_index = l_stack.pop();
                CUDA_glutton_max::print_device_info_position_index(0, l_stack);
                l_new_level = false;
                my_cuda::print_single(1, "SHOULD NOT BE REACHED 5");
#ifndef ENABLE_CUDA_CODE
                exit(-1);
#endif // ENABLE_CUDA_CODE
                continue;
            }

#if VERBOSITY_LEVEL >= 2
            my_cuda::print_single(1, "Apply best candidate");
#endif // VERBOSITY_LEVEL >= 2
            CUDA_glutton_max::apply_best_candidate(l_best_information, l_stack, p_color_constraints);

            l_new_level = true;
#if VERBOSITY_LEVEL >= 3
            my_cuda::print_single(2, "after applying change\n");
            CUDA_glutton_max::print_position_info(2, l_stack);
#endif // VERBOSITY_LEVEL >= 3
        }

        my_cuda::print_single(0, "End with stack level %i", l_stack.get_level());
    }

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_H
// EOF
