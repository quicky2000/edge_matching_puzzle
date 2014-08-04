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
#ifndef EMP_FSM_H
#define EMP_FSM_H
#include "FSM.h"
#include "emp_FSM_situation.h"
#include "emp_FSM_transition.h"

namespace edge_matching_puzzle
{
  class emp_piece_db;
  class emp_FSM_motor;
  class emp_FSM_situation_analyzer;

  class emp_FSM:public FSM_base::FSM<emp_FSM_situation,emp_FSM_transition>
  {
  public:
    emp_FSM(const emp_FSM_info & p_info,
	    const emp_piece_db & p_piece_db);
    // Methods inherited from FSM
    void configure(void);
    const std::string & get_class_name(void)const;

  private:
    static const std::string m_class_name;
  };
  
}
#endif // EMP_FSM_H
//EOF
