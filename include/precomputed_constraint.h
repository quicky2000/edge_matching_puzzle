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
#ifndef PRECOMPUTED_CONSTRAINT_H
#define PRECOMPUTED_CONSTRAINT_H

#include "emp_types.h"

namespace edge_matching_puzzle
{
  /**
     All information need to compute to perform a constraint request
   **/
  class precomputed_constraint
  {
  public:
    inline precomputed_constraint(const unsigned int & p_x,
                                  const unsigned int & p_y,
                                  const emp_types::t_orientation & p_color_orient,
                                  const emp_types::t_orientation & p_side_orient);
    inline precomputed_constraint(void);
    inline const unsigned int & get_x(void)const;
    inline const unsigned int & get_y(void)const;
    inline const emp_types::t_orientation & get_color_orient(void)const;
    inline const emp_types::t_orientation & get_side_orient(void)const;
    inline precomputed_constraint & operator=(const edge_matching_puzzle::precomputed_constraint & p_constraint);
  private:
    unsigned int m_x;
    unsigned int m_y;
    emp_types::t_orientation m_color_orient;
    emp_types::t_orientation m_side_orient;
  };

  //----------------------------------------------------------------------------
  precomputed_constraint::precomputed_constraint(const unsigned int & p_x,
                                                 const unsigned int & p_y,
                                                 const emp_types::t_orientation & p_color_orient,
                                                 const emp_types::t_orientation & p_side_orient):
    m_x(p_x),
    m_y(p_y),
    m_color_orient(p_color_orient),
    m_side_orient(p_side_orient)
      {
      }
#ifdef PRECOMPUTED_CONSTRAINT_ARRAY
  //----------------------------------------------------------------------------
    precomputed_constraint::precomputed_constraint(void):
      m_x(0),
      m_y(0),
      m_color_orient(emp_types::t_orientation::NORTH),
      m_side_orient(emp_types::t_orientation::NORTH)
	{
	}
#endif

      //----------------------------------------------------------------------------
      precomputed_constraint & precomputed_constraint::operator=(const edge_matching_puzzle::precomputed_constraint & p_constraint)
	{
	  m_x = p_constraint.m_x;
	  m_y = p_constraint.m_y;
	  m_color_orient = p_constraint.m_color_orient;
	  m_side_orient = p_constraint.m_side_orient;
	  return *this;
	}
  //----------------------------------------------------------------------------
  const unsigned int & precomputed_constraint::get_x(void)const
    {
      return m_x;
    }
  //----------------------------------------------------------------------------
  const unsigned int & precomputed_constraint::get_y(void)const
    {
      return m_y;
    }
  //----------------------------------------------------------------------------
  const emp_types::t_orientation & precomputed_constraint::get_color_orient(void)const
    {
      return m_color_orient;
    }
  //----------------------------------------------------------------------------
  const emp_types::t_orientation & precomputed_constraint::get_side_orient(void)const
    {
      return m_side_orient;
    }
}
#endif // PRECOMPUTED_CONSTRAINT_H
//EOF
