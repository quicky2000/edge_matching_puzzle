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
#ifndef EMP_FSM_UI_H
#define EMP_FSM_UI_H

#include "FSM_UI.h"
#include "emp_gui.h"
#include <string>
#include <unistd.h>

namespace edge_matching_puzzle
{
  class emp_FSM_UI: public FSM_base::FSM_UI<emp_FSM_situation>
    {
    public:
      inline emp_FSM_UI(emp_gui & p_gui);
      
      // Methods to implement
      inline const std::string & get_class_name()const;
    private:
      // Methods to implement
      inline void display_specific_situation(const emp_FSM_situation & p_situation);
    // End of method to implement
      emp_gui * m_gui;
      static const std::string m_class_name;
    };

  //----------------------------------------------------------------------------
  emp_FSM_UI::emp_FSM_UI(emp_gui & p_gui):
    m_gui(&p_gui)
    {
    }

  //----------------------------------------------------------------------------
  void emp_FSM_UI::display_specific_situation(const emp_FSM_situation & p_situation)
    {
      if(!p_situation.is_final()) return;
      m_gui->display(p_situation);
      m_gui->refresh();
      sleep(3);
    }

  //----------------------------------------------------------------------------
  const std::string & emp_FSM_UI::get_class_name()const
    {
      return m_class_name;
    }
 
}

#endif // EMP_FSM_UI_H
