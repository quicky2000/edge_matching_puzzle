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

#ifndef _EMP_SYSTEM_EQUATION_H_
#define _EMP_SYSTEM_EQUATION_H_

#include "feature_if.h"
#include "emp_piece_db.h"
#include "emp_FSM_info.h"
#include "emp_gui.h"
#include "emp_FSM_situation.h"
#include "emp_variable_generator.h"
#include "emp_types.h"
#include "emp_se_step_info.h"
#include <string>
#include <vector>

namespace edge_matching_puzzle
{
    class emp_se_step_info;

    class feature_system_equation: public feature_if
    {
      public:
        feature_system_equation(const emp_piece_db & p_db
                               ,const emp_FSM_info & p_info
                               ,const std::string & p_initial_situation
                               ,emp_gui & p_gui
                               );

        // Method inherited from feature if
        void run() override;

        // End of method inherited from feature if

        ~feature_system_equation() override = default;

      private:

        emp_FSM_situation extract_situation(const std::vector<emp_se_step_info> & p_stack
                                           ,unsigned int p_step
                                           );
        /**
         * Contains current situation
         */
        emp_FSM_situation m_situation;

        /**
         * Generate variables of equation system representing the puzzle
         */
        emp_variable_generator m_variable_generator;


        std::vector<emp_types::bitfield> m_pieces_and_masks;

        /**
         * Graphical interface for situation display
         */
        emp_gui & m_gui;

        /**
         * Puzzle information
         */
        const emp_FSM_info & m_info;
    };
}
#endif // _EMP_SYSTEM_EQUATION_H_
// EOF
