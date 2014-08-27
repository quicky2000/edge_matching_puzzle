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
#include "emp_situation_binary_dumper.h"
#include "algorithm_deep_raw.h"
#include "emp_gui.h"
#include <string>
#include <unistd.h>

namespace edge_matching_puzzle
{
  class emp_FSM_UI: public FSM_base::FSM_UI<emp_FSM_situation>
    {
    public:
      inline emp_FSM_UI(emp_gui & p_gui,
                        emp_situation_binary_dumper & p_dumper);
      inline ~emp_FSM_UI(void);
      inline void set_algo(const FSM_framework::algorithm_deep_raw & p_algo);
      // Methods to implement
      inline const std::string & get_class_name()const;
    private:
      // Methods to implement
      inline void display_specific_situation(const emp_FSM_situation & p_situation);
    // End of method to implement
      emp_gui * m_gui;
      static const std::string m_class_name;
      emp_situation_binary_dumper & m_dumper;
      const FSM_framework::algorithm_deep_raw * m_algo;
    };

  //----------------------------------------------------------------------------
  emp_FSM_UI::emp_FSM_UI(emp_gui & p_gui,
                         emp_situation_binary_dumper & p_dumper):
    m_gui(&p_gui),
    m_dumper(p_dumper),
    m_algo(NULL)
    {
    }

  //----------------------------------------------------------------------------
    void emp_FSM_UI::set_algo(const FSM_framework::algorithm_deep_raw & p_algo)
    {
      m_algo = &p_algo;
    }
    //----------------------------------------------------------------------------
    emp_FSM_UI::~emp_FSM_UI(void)
      {
        m_dumper.dump(m_algo->get_total_situations());
      }

  //----------------------------------------------------------------------------
  void emp_FSM_UI::display_specific_situation(const emp_FSM_situation & p_situation)
    {
      if(!p_situation.is_final()) return;
#if 0
      std::cout << "Final situation : \"" << p_situation.get_string_id() << "\"" << std::endl ; 
      std::cout << "Situations explored : " << m_algo->get_total_situations() << std::endl ;
#endif
      m_dumper.dump(p_situation);
      m_dumper.dump(m_algo->get_total_situations());
      return;
      m_gui->display(p_situation);
      m_gui->refresh();
      //      sleep(3);
    }

  //----------------------------------------------------------------------------
  const std::string & emp_FSM_UI::get_class_name()const
    {
      return m_class_name;
    }
 
}

#endif // EMP_FSM_UI_H
