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
#ifndef EMP_CONSTRAINT_H
#define EMP_CONSTRAINT_H

#include "emp_types.h"

namespace edge_matching_puzzle
{
  class emp_constraint
  {
    friend std::ostream & operator<<(std::ostream & p_stream,const emp_constraint & p_constraint);
  public:
    inline emp_constraint(const emp_types::t_color_id & p_color,
			  const emp_types::t_orientation & p_orientation);
    inline const emp_types::t_color_id & get_color(void)const;
    inline const emp_types::t_orientation & get_orientation(void)const;
    inline bool operator <(const emp_constraint & p_constraint)const;
    inline bool operator !=(const emp_constraint & p_constraint)const;
  private:
    const emp_types::t_color_id m_color;
    const emp_types::t_orientation  m_orientation;
  };

  //----------------------------------------------------------------------------
  inline std::ostream & operator<<(std::ostream & p_stream,const emp_constraint & p_constraint)
    {
      p_stream << "(" << emp_types::orientation2short_string(p_constraint.m_orientation) << " => " << p_constraint.m_color << ")";
      return p_stream;
    }

  //----------------------------------------------------------------------------
  emp_constraint::emp_constraint(const emp_types::t_color_id & p_color,
				 const emp_types::t_orientation & p_orientation):
    m_color(p_color),
    m_orientation(p_orientation)
    {
    }
    //----------------------------------------------------------------------------
    const emp_types::t_color_id & emp_constraint::get_color(void)const
      {
	return m_color;
      }
    //----------------------------------------------------------------------------
    const emp_types::t_orientation & emp_constraint::get_orientation(void)const
      {
	return m_orientation;
      }

    //----------------------------------------------------------------------------
    bool emp_constraint::operator <(const emp_constraint & p_constraint)const
    {
      if(m_color != p_constraint.m_color) return m_color < p_constraint.m_color;
      return m_orientation < p_constraint.m_orientation;
    }

    //----------------------------------------------------------------------------
    bool emp_constraint::operator !=(const emp_constraint & p_constraint)const
    {
      if(m_color != p_constraint.m_color) return m_color != p_constraint.m_color;
      return m_orientation != p_constraint.m_orientation;
    }
}
#endif // EMP_CONSTRAINT_H
//EOF
