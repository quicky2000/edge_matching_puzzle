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
                        ,unsigned int p_x
                        ,unsigned int p_y
                        );

        void select_variable(unsigned int p_variable_index
                            ,emp_se_step_info & p_previous_step
                            ,const emp_types::bitfield & p_mask
                            );

        bool get_next_variable(unsigned int & p_variable_index) const;

        unsigned int get_variable_index() const;

        /**
         * Method checking if applying a mask on variable bitfield result on
         * a bitfield with only 0 bits
         * @param p_mask mask to apply
         * @return false if result has only 0 bits
         */
        bool check_mask(const emp_types::bitfield & p_mask, unsigned int p_variable_index);

        void set_check_piece_index(unsigned int p_check_piece_index);

        unsigned int get_check_piece_index() const;

        unsigned int get_x() const;

        unsigned int get_y() const;

        emp_types::t_kind get_kind() const;

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

        /**
         * Check piece index
         */
        unsigned int m_check_piece_index;

        /**
         * X position
         */
        unsigned int m_x;

        /**
         * X position
         */
        unsigned int m_y;
    };
}
#endif // _EMP_SE_STEP_INFO_H_
