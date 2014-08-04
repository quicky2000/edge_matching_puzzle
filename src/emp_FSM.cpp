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
#include "emp_FSM.h"
#include "emp_FSM_motor.h"
#include "emp_FSM_situation_analyzer.h"

namespace edge_matching_puzzle
{
  //-----------------------------------------------------------------------------
  emp_FSM::emp_FSM(const emp_FSM_info & p_info,
		   const emp_piece_db & p_piece_db):
    FSM<emp_FSM_situation,emp_FSM_transition>("emp_FSM",*(new emp_FSM_motor()), *(new emp_FSM_situation_analyzer(p_info,p_piece_db)))
  {
    emp_FSM_situation * l_initial_situation = new emp_FSM_situation();
    l_initial_situation->set_context(*(new emp_FSM_context()));

    const emp_piece_corner & l_corner = p_piece_db.get_corner(0);
    for(unsigned int l_index = (unsigned int)emp_types::t_orientation::NORTH;
        l_index <= (unsigned int)emp_types::t_orientation::WEST;
        ++l_index)
      {
        if(!l_corner.get_color(emp_types::t_orientation::NORTH,(emp_types::t_orientation)l_index) && !l_corner.get_color(emp_types::t_orientation::WEST,(emp_types::t_orientation)l_index))
          {
            l_initial_situation->set_piece(0,0,emp_types::t_oriented_piece(l_corner.get_id(),(emp_types::t_orientation)l_index));
          }
      }

    set_situation(*l_initial_situation);
  }

  //-----------------------------------------------------------------------------
  void emp_FSM::configure(void)
  {
  }
  
  //-----------------------------------------------------------------------------
  const std::string & emp_FSM::get_class_name(void)const
  {
    return m_class_name;
  }
  const std::string emp_FSM::m_class_name = "emp_FSM";
}
//EOF
