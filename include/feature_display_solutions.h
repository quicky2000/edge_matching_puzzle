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

#ifndef FEATURE_DISPLAY_SOLUTIONS
#define FEATURE_DISPLAY_SOLUTIONS

#include "emp_gui.h"
#include "algo_based_feature.h"
#include "algorithm_deep_raw.h"
#include <string>
#include <unistd.h>

namespace edge_matching_puzzle
{
  class feature_display_solutions: public algo_based_feature<FSM_framework::algorithm_deep_raw>
  {
  public:
    inline feature_display_solutions(const emp_piece_db & p_db,
                               const emp_FSM_info & p_info,
                               emp_gui & p_gui);
    // Methods to implement inherited from algo_based_feature
    inline const std::string& get_class_name(void) const;
    inline void print_status(void){}
    // End of Methods to implement inherited from algo_based_feature
  private:
    // Methods to implement inherited from algo_based_feature
    inline void display_specific_situation(const emp_FSM_situation & p_situation);
    // End of method to implement inherited from algo_based_feature
    static const std::string m_class_name;
  };

  //----------------------------------------------------------------------------
  feature_display_solutions::feature_display_solutions(const emp_piece_db & p_db,
                                           const emp_FSM_info & p_info,
                                           emp_gui & p_gui):
    algo_based_feature<FSM_framework::algorithm_deep_raw>(p_db,p_info,p_gui)
    {
    }

    //----------------------------------------------------------------------------
    void feature_display_solutions::display_specific_situation(const emp_FSM_situation & p_situation)
    {
      if(!p_situation.is_final()) return;
      std::cout << "Final situation : \"" << p_situation.get_string_id() << "\"" << std::endl ; 
      std::cout << "Situations explored : " << get_algo().get_total_situations() << std::endl ;
      get_gui().display(p_situation);
      get_gui().refresh();
    }

    //----------------------------------------------------------------------------
    const std::string & feature_display_solutions::get_class_name(void) const
      {
        return m_class_name;
      }
 }

#endif // FEATURE_DISPLAY_SOLUTIONS
//EOF
