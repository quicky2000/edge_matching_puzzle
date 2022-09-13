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
#include "CUDA_common.h"
#include "CUDA_color_constraints.h"
#include "CUDA_glutton_max_stack.h"
#include "emp_FSM_info.h"
#include "emp_piece_db.h"
#include "emp_situation.h"
#include "situation_string_formatter.h"
#include "quicky_exception.h"

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
            CUDA_info();

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

        /**
         * CPU debug version of CUDA algorithm
         */
        inline
        void run()
        {
            prepare_constants(m_piece_db, m_info);
            std::unique_ptr<CUDA_color_constraints> l_color_constraints = prepare_color_constraints(m_piece_db, m_info);
            emp_situation l_start_situation;
            std::unique_ptr<CUDA_glutton_max_stack> l_stack = prepare_stack(m_piece_db, m_info, l_start_situation);
            display_result(*l_stack, l_start_situation, m_info);
        }

      private:

        const emp_piece_db & m_piece_db;
        const emp_FSM_info & m_info;

    };


}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_H
// EOF
