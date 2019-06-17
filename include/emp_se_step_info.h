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

        inline
        void select_variable(unsigned int p_variable_index
                            ,emp_se_step_info & p_previous_step
                            ,const emp_types::bitfield & p_mask
                            );

        inline
        bool get_next_variable(unsigned int & p_variable_index) const;

        inline
        unsigned int get_variable_index() const;

        /**
         * Method checking if applying a mask on variable bitfield result on
         * a bitfield with only 0 bits
         * @param p_mask mask to apply
         * @return false if result has only 0 bits
         */
        inline
        bool check_mask(const emp_types::bitfield & p_mask, unsigned int p_variable_index);

        inline
        void set_check_piece_index(unsigned int p_check_piece_index);

        inline
        unsigned int get_check_piece_index() const;

        inline
        unsigned int get_x() const;

        inline
        unsigned int get_y() const;

        inline
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


    //-------------------------------------------------------------------------
    bool
    emp_se_step_info::get_next_variable(unsigned int & p_variable_index) const
    {
        p_variable_index = (unsigned int)m_available_variables.ffs(p_variable_index);
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
        unsigned int l_min_index = m_variable_index < p_variable_index ? m_variable_index : p_variable_index;
        m_available_variables.apply_and(p_previous_step.m_available_variables, p_mask, l_min_index);
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_se_step_info::get_variable_index() const
    {
        return m_variable_index;
    }

    //-------------------------------------------------------------------------
    bool
    emp_se_step_info::check_mask(const emp_types::bitfield & p_mask, unsigned int p_variable_index)
    {
        return m_available_variables.r_and_not_null(p_mask, p_variable_index);
    }

    //-------------------------------------------------------------------------
    void
    emp_se_step_info::set_check_piece_index(unsigned int p_check_piece_index)
    {
        m_check_piece_index = p_check_piece_index;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_se_step_info::get_check_piece_index() const
    {
        return m_check_piece_index;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_se_step_info::get_x() const
    {
        return m_x;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_se_step_info::get_y() const
    {
        return m_y;
    }

    //-------------------------------------------------------------------------
    emp_types::t_kind
    emp_se_step_info::get_kind() const
    {
        return m_position_kind;
    }

}
#endif // _EMP_SE_STEP_INFO_H_
