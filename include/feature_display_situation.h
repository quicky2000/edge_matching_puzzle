/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
      Copyright (C) 2016  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef FEATURE_DISPLAY_SITUATION_H
#define FEATURE_DISPLAY_SITUATION_H

#include "emp_gui.h"
#include "feature_if.h"

namespace edge_matching_puzzle
{
  class feature_display_situation :public feature_if
  {
  public:
    inline feature_display_situation(const std::string & p_situation,
				    const emp_FSM_info & p_info,
				    emp_gui & p_gui);
    // Virtual methods inherited from feature_if
    inline void run(void);
    // End of virtual methods inherited from feature_if    
  private:
    emp_FSM_situation m_situation;
    emp_gui & m_gui;
    const emp_FSM_info & m_info;
  };
 
  //----------------------------------------------------------------------------
  feature_display_situation::feature_display_situation(const std::string & p_situation,
						     const emp_FSM_info & p_info,
						     emp_gui & p_gui):
    m_gui(p_gui),
    m_info(p_info)
      {
	m_situation.set_context(*(new emp_FSM_context(m_info.get_width() * m_info.get_height())));
	m_situation.set(p_situation);
      }
    
    //----------------------------------------------------------------------------
    void feature_display_situation::run(void)
    {
      m_gui.display(m_situation);
      m_gui.refresh();
      sleep(10);
    }
}
#endif // FEATURE_DISPLAY_SITUATION_H
//EOF
