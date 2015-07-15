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

#ifndef FEATURE_DUMP_SOLUTIONS
#define FEATURE_DUMP_SOLUTIONS

#include "emp_gui.h"
#include "algo_based_feature.h"
#include "algorithm_deep_raw.h"
#include "emp_situation_binary_dumper.h"
#include <string>
#include <unistd.h>

namespace edge_matching_puzzle
{
  class feature_dump_solutions: public algo_based_feature<FSM_framework::algorithm_deep_raw>
  {
  public:
    inline feature_dump_solutions(const emp_piece_db & p_db,
                                  const emp_FSM_info & p_info,
                                  emp_gui & p_gui,
                                  const std::string & p_file_name);

    inline ~feature_dump_solutions(void);
    // Methods to implement inherited from algo_based_feature
    inline const std::string& get_class_name(void) const;
    inline void print_status(void){}
    // End of Methods to implement inherited from algo_based_feature
  private:
    // Methods to implement inherited from algo_based_feature
    inline void display_specific_situation(const emp_FSM_situation & p_situation);
    // End of method to implement inherited from algo_based_feature
    emp_situation_binary_dumper m_dumper;

    inline static const std::string compute_file_name(const emp_FSM_info & p_info);

    static const std::string m_class_name;
  };

  //----------------------------------------------------------------------------
  feature_dump_solutions::feature_dump_solutions(const emp_piece_db & p_db,
                                                 const emp_FSM_info & p_info,
                                                 emp_gui & p_gui,
                                                 const std::string & p_file_name):
    algo_based_feature<FSM_framework::algorithm_deep_raw>(p_db,p_info,p_gui),
    m_dumper("" != p_file_name ? p_file_name : compute_file_name(p_info),p_info)
    {
    }

  //----------------------------------------------------------------------------
    const std::string feature_dump_solutions::compute_file_name(const emp_FSM_info & p_info)
      {
        std::string l_file_name;
        std::stringstream l_stream_width;
        l_stream_width << p_info.get_width();
        std::stringstream l_stream_height;
        l_stream_height << p_info.get_height();
        l_file_name = l_stream_width.str() + "_" + l_stream_height.str() + "_results.bin";
        return l_file_name;
      }

    //----------------------------------------------------------------------------
    void feature_dump_solutions::display_specific_situation(const emp_FSM_situation & p_situation)
    {
      if(!p_situation.is_final()) return;
      m_dumper.dump(p_situation);
      m_dumper.dump(get_algo().get_total_situations());
    }

    //----------------------------------------------------------------------------
    const std::string & feature_dump_solutions::get_class_name(void) const
      {
        return m_class_name;
      }
    //----------------------------------------------------------------------------
    feature_dump_solutions::~feature_dump_solutions(void)
      {
        m_dumper.dump(get_algo().get_total_situations());
      }
 }

#endif // FEATURE_DUMP_SOLUTIONS
//EOF
