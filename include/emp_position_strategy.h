/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2015  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
   The positions strategy objects contains precomputed information necessary
   to determine transitions related to one position
**/
#ifndef EMP_POSITION_STRATEGY_H
#define EMP_POSITION_STRATEGY_H

#include "emp_types.h"
#include "common.h"

namespace edge_matching_puzzle
{
    class emp_position_strategy
    {
      public:
        // neighbour information : store all what is necessary to access to neighbour
        // info : index in strategy + bitmask to extract the colour generating the constraint
        typedef std::pair<emp_position_strategy *,emp_types::t_binary_piece> t_neighbour_access;

        inline
        emp_position_strategy(const emp_types::t_kind & p_kind
                             ,const emp_types::bitfield & p_previous_available_pieces
                             );

        /**
         * To know which kind of piece is stored in this position
        **/
        [[nodiscard]]
        inline
        const emp_types::t_kind & get_kind()const;

        /**
         * Return the binary representation of piece stored in this position
         **/
        [[nodiscard]]
        inline
        const emp_types::t_binary_piece & get_piece_info()const;

        /**
         * Assign a piece to this position by storing piece binary representation
         **/
        inline
        void set_piece_info(const emp_types::t_binary_piece & p_piece_info);

        /**
         * Define neighbour access info for one orientation
         **/
        inline
        void set_neighbour_access(const emp_types::t_orientation & p_orientation
                                 ,const t_neighbour_access & p_access_info
                                 );

        /**
         * Compute binary constraint for this position according to neighbour access infos
         **/
        [[nodiscard]]
        inline
        emp_types::t_binary_piece compute_constraint()const;

        /**
         * Compute real available transitions by taking in account theorical transitions and available pieces
         **/
        inline
        void compute_available_transitions(const emp_types::bitfield & p_theoric_transitions);

        /**
         * Return available pieces
         **/
        [[nodiscard]]
        inline
        const emp_types::bitfield & get_available_pieces()const;

        /**
         * Select piece and update info to make it no more usable
         **/
        inline
        void select_piece(const unsigned int & p_transition_id
#ifdef HANDLE_IDENTICAL_PIECES
                         ,const emp_types::bitfield & p_identical_pieces
#endif // HANDLE_IDENTICAL_PIECES
                         );

        /**
         * to get next available transition : Oriented piece kind id
         **/
        [[nodiscard]]
        inline
        unsigned int get_next_transition()const;

      private:

        const emp_types::t_kind m_position_kind;

        emp_types::t_binary_piece m_piece_info;

        t_neighbour_access m_neighbour_access[4];

        emp_types::bitfield m_available_transitions;

        const emp_types::bitfield & m_previous_available_pieces;

        emp_types::bitfield m_available_pieces;
    };

    //----------------------------------------------------------------------------
    emp_position_strategy::emp_position_strategy(const emp_types::t_kind & p_kind
                                                ,const emp_types::bitfield & p_previous_available_pieces
                                                )
    :m_position_kind(p_kind)
    ,m_piece_info(0)
    ,m_available_transitions(p_previous_available_pieces.bitsize())
    ,m_previous_available_pieces(p_previous_available_pieces)
    ,m_available_pieces(p_previous_available_pieces.bitsize())
    {
        std::transform(&m_neighbour_access[0], &m_neighbour_access[4], &m_neighbour_access[0], [](t_neighbour_access){return std::make_pair(nullptr,0);});
    }

    //--------------------------------------------------------------------------
    const emp_types::t_kind & emp_position_strategy::get_kind()const
    {
        return m_position_kind;
    }

    //--------------------------------------------------------------------------
    const emp_types::t_binary_piece & emp_position_strategy::get_piece_info()const
    {
        return m_piece_info;
    }

    //--------------------------------------------------------------------------
    void emp_position_strategy::set_piece_info(const emp_types::t_binary_piece & p_piece_info)
    {
        m_piece_info = p_piece_info;
    }

    //--------------------------------------------------------------------------
    void emp_position_strategy::set_neighbour_access(const emp_types::t_orientation & p_orientation
                                                    ,const t_neighbour_access & p_access_info
                                                    )
    {
        auto l_orientation = static_cast<unsigned int>(p_orientation);
        assert(l_orientation < 4);
        m_neighbour_access[l_orientation] = p_access_info;
    }

    //--------------------------------------------------------------------------
    const emp_types::bitfield & emp_position_strategy::get_available_pieces()const
    {
        return m_available_pieces;
    }

    //--------------------------------------------------------------------------
    emp_types::t_binary_piece emp_position_strategy::compute_constraint()const
    {
        emp_types::t_binary_piece l_result = 0;
        for(const auto & l_neighbour_access : m_neighbour_access)
        {
            l_result |= l_neighbour_access.first->get_piece_info() & l_neighbour_access.second;
        }
        return l_result;
    }

    //----------------------------------------------------------------------------
    void emp_position_strategy::compute_available_transitions(const emp_types::bitfield & p_theoric_transitions)
    {
        m_available_transitions.apply_and(p_theoric_transitions,m_previous_available_pieces);
    }

    //----------------------------------------------------------------------------
    void emp_position_strategy::select_piece(const unsigned int & p_transition_id
#ifdef HANDLE_IDENTICAL_PIECES
                                            ,const emp_types::bitfield & p_identical_pieces
#endif // HANDLE_IDENTICAL_PIECES
                                            )
    {
        m_available_pieces = m_previous_available_pieces;
        m_available_pieces.set(0,4,p_transition_id & ~((emp_types::t_piece_id)0x3));
#ifndef HANDLE_IDENTICAL_PIECES
        m_available_transitions.set(0,1,p_transition_id);
#else
        m_available_transitions.apply_and(m_available_transitions,p_identical_pieces);
#endif
    }

    //----------------------------------------------------------------------------
    unsigned int emp_position_strategy::get_next_transition()const
    {
        return m_available_transitions.ffs();
    }
}
#endif // EMP_POSITION_STRATEGY_H
//EOF
