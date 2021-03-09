/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2020  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_FACTORY_FEATURE_H
#define EDGE_MATCHING_PUZZLE_FACTORY_FEATURE_H

#include "feature_border_exploration.h"
#include "feature_simplex.h"
#include "feature_display_all.h"
#include "feature_display_max.h"
#include "feature_display_solutions.h"
#include "feature_dump_solutions.h"
#include "feature_compute_stats.h"
#include "feature_dump_summary.h"
#include "feature_display_dump.h"
#include "feature_display_dump.h"
#include "feature_display_situation.h"
#include "feature_system_equation.h"
#include "feature_CUDA_backtracker.h"
#include "feature_CUDA_glutton_max.h"
#include "feature_situation_profile.h"
#include "feature_profile.h"
#include "emp_strategy.h"
#include <string>

namespace edge_matching_puzzle
{
    class factory_feature
    {
      public:

        inline static
        feature_if & create_feature(const std::string & p_feature_name
                                   ,emp_piece_db & p_piece_db
                                   ,emp_FSM_info & p_info
                                   ,emp_gui & p_gui
                                   ,const std::string & p_initial_situation
                                   ,const std::string & p_dump_file_name
                                   ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                   ,const std::string & p_hint
                                   );

      private:

    };

    //-------------------------------------------------------------------------
    feature_if & factory_feature::create_feature(const std::string & p_feature_name
                                                ,emp_piece_db & p_piece_db
                                                ,emp_FSM_info & p_info
                                                ,emp_gui & p_gui
                                                ,const std::string & p_initial_situation
                                                ,const std::string & p_dump_file_name
                                                ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                                ,const std::string & p_hint
                                                )
    {
        feature_if * l_feature;
        if("display_all" == p_feature_name)
        {
            l_feature = new feature_display_all(p_piece_db, p_info, p_gui);
        }
        else if("display_max" == p_feature_name)
        {
            l_feature = new feature_display_max(p_piece_db, p_info, p_gui);
        }
        else if("display_solutions" == p_feature_name)
        {
            l_feature = new feature_display_solutions(p_piece_db, p_info, p_gui);
        }
        else if("dump_solutions" == p_feature_name)
        {
            l_feature = new feature_dump_solutions(p_piece_db, p_info, p_gui, p_dump_file_name);
        }
        else if("dump_summary" == p_feature_name)
        {
            l_feature = new feature_dump_summary(p_dump_file_name, p_info);
        }
        else if("display_dump" == p_feature_name)
        {
            l_feature = new feature_display_dump(p_dump_file_name, p_info, p_gui);
        }
        else if("display_situation" == p_feature_name)
        {
            l_feature = new feature_display_situation(p_initial_situation, p_info, p_gui);
        }
        else if("compute_stats" == p_feature_name)
        {
            l_feature = new feature_compute_stats(p_piece_db, p_info, p_gui);
        }
        else if("border_exploration" == p_feature_name)
        {
            l_feature = new feature_border_exploration(p_piece_db
                                                      ,p_info
                                                      ,p_initial_situation
                                                      );
        }
        else if("simplex" == p_feature_name)
        {
            l_feature = new feature_simplex(p_piece_db
                                           ,p_strategy_generator
                                           ,p_info
                                           ,p_initial_situation
                                           ,p_gui
                                           );
        }
        else if("system_equation" == p_feature_name)
        {
            l_feature = new feature_system_equation(p_piece_db
                                                   ,p_strategy_generator
                                                   ,p_info
                                                   ,p_initial_situation
                                                   ,p_hint
                                                   ,p_gui
                                                   );
        }
        else if("new_strategy" == p_feature_name || "new_text_strategy" == p_feature_name)
        {
            auto * l_strategy = new emp_strategy(p_strategy_generator, p_piece_db, p_gui, p_info, p_dump_file_name);
            if(!p_initial_situation.empty())
            {
                l_strategy->set_initial_state(p_initial_situation);
            }
            l_feature = l_strategy;
        }
        else if("CUDA_backtracker" == p_feature_name)
        {
            l_feature = new feature_CUDA_backtracker(p_piece_db, p_info, p_strategy_generator);
        }
        else if("CUDA_glutton_max" == p_feature_name)
        {
            l_feature = new feature_CUDA_glutton_max(p_piece_db, p_info);
        }
        else if("situation_profile" == p_feature_name)
        {
            l_feature = new feature_situation_profile(p_piece_db, p_info, p_strategy_generator, p_initial_situation);
        }
        else if("profile" == p_feature_name)
        {
            l_feature = new feature_profile(p_piece_db, p_info, p_strategy_generator);
        }
        else
        {
            throw quicky_exception::quicky_logic_exception("Unsupported feature \"" + p_feature_name + "\"", __LINE__, __FILE__);
        }
        return *l_feature;
    }
}
#endif //EDGE_MATCHING_PUZZLE_FACTORY_FEATURE_H
//EOF
