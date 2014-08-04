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
#ifndef EMP_FSM_MOTOR_H
#define EMP_FSM_MOTOR_H

#include "FSM_motor.h"
#include "emp_FSM_situation.h"
#include "emp_FSM_transition.h"

namespace edge_matching_puzzle
{
  class emp_FSM_motor:public FSM_base::FSM_motor<emp_FSM_situation,emp_FSM_transition>
    {
    public:
      inline emp_FSM_motor(void);
      // Methods inherited from FSM_motor
      inline const std::string & get_class_name(void)const;
      inline emp_FSM_situation & run(const emp_FSM_situation & p_situation,
                                     const emp_FSM_transition & p_transition);
    private:
      static const std::string m_class_name;
    };
  //----------------------------------------------------------------------------
  emp_FSM_motor::emp_FSM_motor(void)
    {
    }

  //----------------------------------------------------------------------------
  const std::string & emp_FSM_motor::get_class_name(void)const
    {
      return m_class_name;
    }
  //----------------------------------------------------------------------------
  emp_FSM_situation & emp_FSM_motor::run(const emp_FSM_situation & p_situation,
                                         const emp_FSM_transition & p_transition)
    {
      emp_FSM_situation *l_result = new emp_FSM_situation(p_situation);
      l_result->set_piece(p_transition.get_x(),p_transition.get_y(),p_transition.get_piece());
      return *l_result;
    }
}
#endif // EMP_FSM_MOTOR_H
//EOF

