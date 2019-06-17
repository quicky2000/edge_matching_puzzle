/*    This file is part of edge_matching_puzzle
      Copyright (C) 2019  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#include <emp_se_step_info.h>

#include "emp_se_step_info.h"
#include "emp_FSM_info.h"

namespace edge_matching_puzzle
{
    //-------------------------------------------------------------------------
    emp_se_step_info::emp_se_step_info(emp_types::t_kind p_kind
                                      ,unsigned int p_nb_variables
                                      ,unsigned int p_x
                                      ,unsigned int p_y
                                      )
    :m_position_kind(p_kind)
    ,m_available_variables(p_nb_variables, true)
    ,m_variable_index{0}
    ,m_check_piece_index{0}
    ,m_x{p_x}
    ,m_y{p_y}
    {

    }

}
// EOF
