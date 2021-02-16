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

}
#endif // EMP_FSM_INFO_H
//EOF

