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
}
#endif // EMP_FSM_INFO_H
//EOF

