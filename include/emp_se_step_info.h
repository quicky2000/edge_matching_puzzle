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

#ifndef _EMP_SE_STEP_INFO_H_
#define _EMP_SE_STEP_INFO_H_

#include "emp_types.h"

namespace edge_matching_puzzle
{
    /**
     * Class storing information related to each step of system equation feature
     */
    class emp_se_step_info
    {
      public:
        /**
         * Constructor
         */
         emp_se_step_info(emp_types::t_kind p_kind
                         ,unsigned int p_nb_variables
                         );

         void select_variable(unsigned int p_variable_index
                             ,emp_se_step_info & p_previous_step
                             ,const emp_types::bitfield & p_mask
                             );

         bool get_next_variable(unsigned int & p_variable_index) const;

         unsigned int get_variable_index() const;

      private:

        /**
         * Position kind for this step
         */
        emp_types::t_kind m_position_kind;

        /**
         *
         */
        emp_types::bitfield m_available_variables;

        /**
         * Variable associated to this step
         */
        unsigned int m_variable_index;
    };
}
#endif // _EMP_SE_STEP_INFO_H_