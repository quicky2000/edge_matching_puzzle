/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
      Copyright (C) 2017  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef SIMPLEX_VARIABLE_H
#define SIMPLEX_VARIABLE_H

#include "emp_types.h"
#include <iostream>

namespace edge_matching_puzzle
{
  /**
     Class storing edge matching puzzle information related to a variable like
     position piece id orientation
  */
  class simplex_variable
  {
    friend std::ostream & operator<<(std::ostream & p_stream, const simplex_variable & p_variable);
  public:
    /**
       Constructor
       @param X position
       @param Y position
    */
    inline simplex_variable(const unsigned int & p_id,
			    const unsigned int & p_x,
			    const unsigned int & p_y,
			    const emp_types::t_piece_id & p_piece_id,
			    const emp_types::t_orientation p_orientation
			    );

    inline const unsigned int & get_id(void)const;
    inline const unsigned int & get_x(void)const;
    inline const unsigned int & get_y(void)const;
    inline const emp_types::t_piece_id & get_piece_id(void)const;
    inline const emp_types::t_orientation & get_orientation(void)const;
  private:
    unsigned int m_id;
    unsigned int m_x;
    unsigned int m_y;
    emp_types::t_piece_id m_piece_id;
    emp_types::t_orientation m_orientation;
  };

  //----------------------------------------------------------------------------
  simplex_variable::simplex_variable(const unsigned int & p_id,
				     const unsigned int & p_x,
				     const unsigned int & p_y,
				     const emp_types::t_piece_id & p_piece_id,
				     const emp_types::t_orientation p_orientation
				     ):
    m_id(p_id),
    m_x(p_x),
    m_y(p_y),
    m_piece_id(p_piece_id),
    m_orientation(p_orientation)
    {
      assert(m_piece_id);
    }

  //----------------------------------------------------------------------------
  const unsigned int & simplex_variable::get_id(void)const
  {
    return m_id;
  }

  //----------------------------------------------------------------------------
  const unsigned int & simplex_variable::get_x(void)const
  {
    return m_x;
  }

  //----------------------------------------------------------------------------
  const unsigned int & simplex_variable::get_y(void)const
  {
    return m_y;
  }

  //----------------------------------------------------------------------------
  const emp_types::t_piece_id & simplex_variable::get_piece_id(void)const
  {
    return m_piece_id;
  }

  //----------------------------------------------------------------------------
  const emp_types::t_orientation & simplex_variable::get_orientation(void)const
  {
    return m_orientation;
  }

  //----------------------------------------------------------------------------
  inline std::ostream & operator<<(std::ostream & p_stream, const simplex_variable & p_variable)
  {
    p_stream << p_variable.m_piece_id;
    p_stream << emp_types::orientation2short_string(p_variable.m_orientation);
    p_stream << "(" << p_variable.m_x << "," << p_variable.m_y << ")";
    return p_stream;
  }
}
#endif // SIMPLEX_VARIABLE_H
//EOF
