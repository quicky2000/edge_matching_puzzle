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

namespace edge_matching_puzzle
{
    //-------------------------------------------------------------------------
    emp_se_step_info::emp_se_step_info(emp_types::t_kind p_kind
                                      ,unsigned int p_nb_variables
                                      )
    :m_position_kind(p_kind)
    ,m_available_variables(p_nb_variables, true)
    ,m_variable_index{0}
    {

    }

    //-------------------------------------------------------------------------
    bool
    emp_se_step_info::get_next_variable(unsigned int & p_variable_index) const
    {
        p_variable_index = (unsigned int)m_available_variables.ffs(); 
        // Remove one because 0 mean no variable available, n mean variable
        // n-1 available
        return p_variable_index-- != 0;
    }

    //-------------------------------------------------------------------------
    void
    emp_se_step_info::select_variable(unsigned int p_variable_index
                                     ,emp_se_step_info & p_previous_step
                                     ,const emp_types::bitfield & p_mask
                                     )
    {
        p_previous_step.m_available_variables.set(0, 1, p_variable_index);
        p_previous_step.m_variable_index = p_variable_index;
        m_available_variables.apply_and(p_previous_step.m_available_variables, p_mask);
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_se_step_info::get_variable_index() const
    {
        return m_variable_index;
    }
}
// EOF
