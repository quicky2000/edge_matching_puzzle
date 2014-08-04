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

namespace edge_matching_puzzle
{
  class emp_FSM_info
    {
    public:
      inline emp_FSM_info(const unsigned int & p_width,
                          const unsigned int & p_height);
      inline const unsigned int & get_width(void)const;
      inline const unsigned int & get_height(void)const;
    private:
      const unsigned int m_width;
      const unsigned int m_height;
    };

  //----------------------------------------------------------------------------
  emp_FSM_info::emp_FSM_info(const unsigned int & p_width,
                             const unsigned int & p_height):
    m_width(p_width),
    m_height(p_height)
    {
    }

  //----------------------------------------------------------------------------
  const unsigned int & emp_FSM_info::get_width(void)const
    {
      return m_width;
    }
  
  //----------------------------------------------------------------------------
  const unsigned int & emp_FSM_info::get_height(void)const
    {
      return m_height;
    }

}
#endif // EMP_FSM_INFO_H
//EOF

