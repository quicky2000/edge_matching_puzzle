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
#ifndef EMP_PIECE_CONSTRAINT_H
#define EMP_PIECE_CONSTRAINT_H

#include "emp_constraint.h"
#include <set>

namespace edge_matching_puzzle
{
  class emp_piece_constraint
  {
    friend std::ostream & operator<<(std::ostream & p_stream,const emp_piece_constraint & p_constraint);
  public:
    inline emp_piece_constraint(void);
    inline emp_piece_constraint(const std::set<emp_constraint> & p_constraint);
    inline bool operator <(const emp_piece_constraint & p_piece_constraint)const;
  private:
    const unsigned int m_size;
    std::set<emp_constraint> m_constraints;
  };
  //----------------------------------------------------------------------------
  emp_piece_constraint::emp_piece_constraint(void):
    //    m_constraints({NULL,NULL,NULL,NULL}),
    m_size(0)
    {
    }

  //----------------------------------------------------------------------------
  inline std::ostream & operator<<(std::ostream & p_stream,const emp_piece_constraint & p_constraint)
    {
      p_stream << "[" ;
      unsigned int l_index = 0;
      for(auto l_iter : p_constraint.m_constraints)
        {
          if(l_index) p_stream << " ";
          p_stream << l_iter;
          ++l_index;
        }
      p_stream << "]" ;
      return p_stream;
    }

  //----------------------------------------------------------------------------
  emp_piece_constraint::emp_piece_constraint(const std::set<emp_constraint> & p_constraint):
    m_size(p_constraint.size()),
    m_constraints(p_constraint)
    {
      
    }
  //----------------------------------------------------------------------------
  bool emp_piece_constraint::operator <(const emp_piece_constraint & p_piece_constraint)const
  {
    if(m_size != p_piece_constraint.m_size) return m_size < p_piece_constraint.m_size;
    return m_constraints < p_piece_constraint.m_constraints;
  }
 
}
#endif // EMP_CONSTRAINT_H
//EOF
