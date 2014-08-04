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

#ifndef EMP_LINK_H
#define EMP_LINK_H

#include "emp_types.h"

namespace edge_matching_puzzle
{
  class emp_link
  {
    friend std::ostream & operator<<(std::ostream & p_stream,const emp_link & p_link);
  public:
    inline emp_link(const emp_types::t_piece_id & p_piece1,
                    const emp_types::t_orientation & p_orient1,
                    const emp_types::t_piece_id & p_piece2,
                    const emp_types::t_orientation & p_orient2
                    );
    
  private:
    const emp_types::t_piece_id m_piece1;
    const emp_types::t_orientation m_orient1;
    const emp_types::t_piece_id m_piece2;
    const emp_types::t_orientation m_orient2;
  };

  //----------------------------------------------------------------------------
  emp_link::emp_link(const emp_types::t_piece_id & p_piece1,
                     const emp_types::t_orientation & p_orient1,
                     const emp_types::t_piece_id & p_piece2,
                     const emp_types::t_orientation & p_orient2
                     ):
    m_piece1(p_piece1),
    m_orient1(p_orient1),
    m_piece2(p_piece2),
    m_orient2(p_orient2)
    {
    }
  //----------------------------------------------------------------------------
  inline std::ostream & operator<<(std::ostream & p_stream,const emp_link & p_link)
    {
      p_stream << "(" << p_link.m_piece1 << ", " ;
      p_stream << emp_types::orientation2string(p_link.m_orient1) << ", " ;
      p_stream << p_link.m_piece2 << ", " ;
      p_stream << emp_types::orientation2string(p_link.m_orient2) << ")" ;
      return p_stream;
    }
}
#endif //EMP_LINK_H
//EOF
