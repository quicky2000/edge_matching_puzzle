/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2014  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef EMP_FSM_INFO_H
#define EMP_FSM_INFO_H

#include "emp_types.h"
#include <cinttypes>
#include <array>

namespace edge_matching_puzzle
{
    /**
     * Class storing generic information about puzzle
     */
    class emp_FSM_info
    {
      public:
        /**
         * Constructor
         * @param p_width Puzzle width
         * @param p_height
         * @param p_piece_id_size
         * @param p_dumped_piece_id_size
         */
        inline
        emp_FSM_info(const uint32_t & p_width
                    ,const uint32_t & p_height
                    ,const unsigned int & p_piece_id_size
                    ,const unsigned int & p_dumped_piece_id_size
                    );

        /**
         * Return puzzle width
         * @return width
         */
        inline
        const uint32_t & get_width() const;

        /**
         * Return puzzle height
         * @return height
         */
        inline
        const uint32_t & get_height() const;

        /**
         * Accessor to number of bits needed to code pieces ids from to 0 to nb pieces - 1
         * @return number of bits needed to code pieces ids from to 0 to nb pieces - 1
         */
        inline
        const unsigned int & get_piece_id_size() const;

        /**
         * Accessor to number of bits needed to code pieces ids from to 1 to nb pieces
         * @return number of bits needed to code pieces ids from to 1 to nb pieces
         */
        inline
        const unsigned int & get_dumped_piece_id_size() const;

        /**
         * Indicate if position defined by parameters is corner/border/center
         * @param p_x column index
         * @param p_y row index
         * @return kind of position: corner/border/center
         */
        inline
        emp_types::t_kind get_position_kind(const unsigned int & p_x
                                           ,const unsigned int & p_y
                                           ) const;

        /**
           Compute index related to position X,Y
           @param X position
           @param Y position
           @return index related to position
        */
        inline
        unsigned int get_position_index(const unsigned int & p_x
                                       ,const unsigned int & p_y
                                       ) const;

        /**
         * Compute X coordinate from position index
         * @param p_position_index
         * @return X coordinate
         */
        inline
        unsigned int get_x(unsigned int p_position_index) const;

        /**
         * Compute Y coordinate from position index
         * @param p_position_index
         * @return Y coordinate
         */
        inline
        unsigned int get_y(unsigned int p_position_index) const;

        /**
         * Return number of border positions
         * @return number of border positions
         */
        inline
        unsigned int get_nb_borders() const;

        /**
         * Return number of center positions
         * @return number of center positions
         */
        inline
        unsigned int get_nb_centers() const;

        /**
         * Return number of pieces
         * @return number of pieces
         */
        inline
        unsigned int get_nb_pieces() const;

        /**
         * Compute corner borders orientation according to position index
         * @return pair of orientations indicating orientation of corner borders
         */
        inline
        std::pair<emp_types::t_orientation, emp_types::t_orientation> get_corner_orientation(unsigned int p_position_index) const;

        /**
         * Compute border orientation according to position index
         * @return pair of orientations indicating orientation of corner borders
         */
        inline
        emp_types::t_orientation get_border_orientation(unsigned int p_position_index) const;

      private:

        /**
         * Puzzle width
         */
        const uint32_t m_width;

        /**
         * Puzzle heigth
         */
        const uint32_t m_height;

        /**
         * Number of bits needed to code pieces ids from o to nb pieces - 1
         */
        const unsigned int m_piece_id_size;

        /**
         * Number of bits needed to code pieces ids from 1 to nb  pieces
         */
        const unsigned int m_dumped_piece_id_size;
    };


    //----------------------------------------------------------------------------
    emp_FSM_info::emp_FSM_info(const uint32_t & p_width
                              ,const uint32_t & p_height
                              ,const unsigned int & p_piece_id_size
                              ,const unsigned int & p_dumped_piece_id_size
                              )
    :m_width(p_width)
    ,m_height(p_height)
    ,m_piece_id_size(p_piece_id_size)
    ,m_dumped_piece_id_size(p_dumped_piece_id_size)
    {
    }

    //----------------------------------------------------------------------------
    const uint32_t & emp_FSM_info::get_width() const
    {
        return m_width;
    }
  
    //----------------------------------------------------------------------------
    const uint32_t & emp_FSM_info::get_height() const
    {
        return m_height;
    }

    //----------------------------------------------------------------------------
    const unsigned int & emp_FSM_info::get_piece_id_size() const
    {
        return m_piece_id_size;
    }

    //----------------------------------------------------------------------------
    const unsigned int & emp_FSM_info::get_dumped_piece_id_size() const
    {
        return m_dumped_piece_id_size;
    }

    //-------------------------------------------------------------------------
    emp_types::t_kind
    emp_FSM_info::get_position_kind(const unsigned int & p_x
                                   ,const unsigned int & p_y
                                   ) const
    {
        assert(p_x < m_width);
        assert(p_y < m_height);
        emp_types::t_kind l_type = emp_types::t_kind::CENTER;
        if(!p_x || !p_y || m_width - 1 == p_x || p_y == m_height - 1)
        {
            l_type = emp_types::t_kind::BORDER;
            if((!p_x && !p_y) ||
               (!p_x && p_y == m_height - 1) ||
               (!p_y && p_x == m_width - 1) ||
               (p_y == m_height - 1 && p_x == m_width - 1)
              )
            {
                l_type = emp_types::t_kind::CORNER;
            }
        }
        return l_type;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_FSM_info::get_position_index(const unsigned int & p_x
                                    ,const unsigned int & p_y
                                    ) const
    {
        assert(p_x < m_width);
        assert(p_y < m_height);
        return m_width * p_y + p_x;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_FSM_info::get_nb_borders() const
    {
        return 2 * ((m_width - 2) + (m_height - 2));
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_FSM_info::get_nb_centers() const
    {
        return (m_width - 2) * (m_height - 2);
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_FSM_info::get_nb_pieces() const
    {
        return m_width * m_height;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_FSM_info::get_x(unsigned int p_position_index) const
    {
        assert(p_position_index < get_nb_pieces());
        return p_position_index % m_width;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_FSM_info::get_y(unsigned int p_position_index) const
    {
        assert(p_position_index < get_nb_pieces());
        return p_position_index / m_width;
    }

    //-------------------------------------------------------------------------
    std::pair<emp_types::t_orientation, emp_types::t_orientation>
    emp_FSM_info::get_corner_orientation(unsigned int p_position_index) const
    {
        std::array<unsigned int, 4> l_corner_positions{get_position_index(0, 0)
                                                      ,get_position_index(m_width - 1, 0)
                                                      ,get_position_index(m_width - 1, m_height - 1)
                                                      ,get_position_index(0 , m_height - 1)
                                                      };
        unsigned int l_index = 0;
        for(auto l_iter: l_corner_positions)
        {
            if(p_position_index == l_iter)
            {
                emp_types::t_orientation l_border1{static_cast<emp_types::t_orientation>(l_index)};
                emp_types::t_orientation l_border2{emp_types::get_previous_orientation(l_border1)};
                return {l_border1, l_border2};
            }
            ++l_index;
        }
        throw quicky_exception::quicky_logic_exception("Position (" + std::to_string(get_x(p_position_index)) + "," + std::to_string(get_y(p_position_index)) + ") is not a corner", __LINE__, __FILE__);
    }

    //-------------------------------------------------------------------------
    emp_types::t_orientation
    emp_FSM_info::get_border_orientation(unsigned int p_position_index) const
    {
        unsigned int l_x = get_x(p_position_index);
        unsigned int l_y = get_y(p_position_index);

        if(!l_y && l_x && (l_x != m_width - 1))
        {
            return emp_types::t_orientation::NORTH;
        }
        if((l_x == m_width - 1) && l_y && (l_y < m_height - 1))
        {
            return emp_types::t_orientation::EAST;
        }
        if((l_y == m_height - 1) && l_x && (l_x != m_width - 1))
        {
            return emp_types::t_orientation::SOUTH;
        }
        if(!l_x && l_y && (l_y < m_height - 1))
        {
            return emp_types::t_orientation::WEST;
        }
        throw quicky_exception::quicky_logic_exception("Position (" + std::to_string(get_x(p_position_index)) + "," + std::to_string(get_y(p_position_index)) + ") is not a border", __LINE__, __FILE__);
    }

}
#endif // EMP_FSM_INFO_H
//EOF

