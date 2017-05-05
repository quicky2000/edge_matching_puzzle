/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
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
#ifndef FEATURE_DISPLAY_DUMP_H
#define FEATURE_DISPLAY_DUMP_H

#include "emp_gui.h"
#include "feature_if.h"
#include "emp_situation_binary_reader.h"

namespace edge_matching_puzzle
{
  class feature_display_dump :public feature_if
  {
  public:
    inline feature_display_dump(const std::string & p_file_name,
				const emp_FSM_info & p_info,
				emp_gui & p_gui);
      // Virtual methods inherited from feature_if
    inline void run(void);
    // End of virtual methods inherited from feature_if    
  private:
    emp_situation_binary_reader m_reader;
    emp_gui & m_gui;
    const emp_FSM_info & m_info;
  };
 
  //----------------------------------------------------------------------------
  feature_display_dump::feature_display_dump(const std::string & p_file_name,
					     const emp_FSM_info & p_info,
					     emp_gui & p_gui):
    m_reader(p_file_name,p_info),
    m_gui(p_gui),
    m_info(p_info)
  {
  }

  //----------------------------------------------------------------------------
  void feature_display_dump::run(void)
  {
    const uint64_t & l_nb_recorded = m_reader.get_nb_recorded();
    emp_FSM_situation l_situation;
    l_situation.set_context(*(new emp_FSM_context(m_info.get_width() * m_info.get_height())));
    uint64_t l_situation_number;
    for(uint64_t i = 0 ; i < l_nb_recorded ; ++i)
      {
	m_reader.read(i,l_situation,l_situation_number);
	m_gui.display(l_situation);
	m_gui.refresh();
	std::cout << "Solution " << i << std::endl;
	std::cout << " ==> \"" << l_situation.to_string() << "\"" << std::endl;
	sleep(1);
      }
  }
 }
#endif // FEATURE_DISPLAY_DUMP_H
//EOF
