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
#include "signal_handler.h"
#include "parameter_manager.h"
#include "emp_piece.h"
#include "emp_pieces_parser.h"
#include "emp_gui.h"
#include "emp_piece_db.h"
#include "emp_FSM.h"

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

#include "emp_strategy_generator_factory.h"
#include "emp_strategy.h"

#include "quicky_exception.h"
#include <unistd.h>


#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <set>
#include <map>

using namespace edge_matching_puzzle;

//------------------------------------------------------------------------------
int main(int argc,char ** argv)
{
    try
    {
        // Defining application command line parameters
        parameter_manager::parameter_manager l_param_manager("edge_matching_puzzle.exe", "--", 3);
        parameter_manager::parameter_if l_definition_file("definition", false);
        l_param_manager.add(l_definition_file);
        parameter_manager::parameter_if l_ressources_path("ressources", false);
        l_param_manager.add(l_ressources_path);
        parameter_manager::parameter_if l_feature_name_parameter("feature_name", false);
        l_param_manager.add(l_feature_name_parameter);
        parameter_manager::parameter_if l_dump_file("dump_file",true);
        l_param_manager.add(l_dump_file);
        parameter_manager::parameter_if l_initial_situation("initial_situation", true);
        l_param_manager.add(l_initial_situation);
        parameter_manager::parameter_if l_strategy_param("strategy", true);
        l_param_manager.add(l_strategy_param);

        // Treating parameters
        l_param_manager.treat_parameters(argc,argv);

        std::string l_dump_file_name = l_dump_file.value_set() ? l_dump_file.get_value<std::string>() : "results.bin";

        // Get puzzle description
        std::vector<emp_piece> l_pieces;
        emp_pieces_parser l_piece_parser(l_definition_file.get_value<std::string>());
        unsigned int l_width = 0;
        unsigned int l_height = 0;
        l_piece_parser.parse(l_width, l_height, l_pieces);

        std::cout << l_pieces.size() << " pieces loaded" << std::endl ;
        if(l_pieces.size() != l_width * l_height)
        {
            std::stringstream l_stream_width;
            l_stream_width << l_width;
            std::stringstream l_stream_height;
            l_stream_height << l_height;
            std::stringstream l_stream_nb_pieces;
            l_stream_nb_pieces << l_pieces.size();
            throw quicky_exception::quicky_logic_exception("Inconsistency between puzzle dimensions (" + l_stream_width.str() + "*" + l_stream_height.str() + ") and piece number (" + l_stream_nb_pieces.str() + ")",__LINE__,__FILE__);
        }

        emp_gui l_gui(l_width,l_height,l_ressources_path.get_value<std::string>(), l_pieces);

        emp_piece_db l_piece_db(l_pieces, l_width, l_height);
        emp_FSM_info l_info(l_width, l_height, l_piece_db.get_piece_id_size(), l_piece_db.get_dumped_piece_id_size());

        emp_FSM_situation::init(l_info);
        std::string l_feature_name = l_feature_name_parameter.get_value<std::string>();

        // Generate strategy if needed
        std::unique_ptr<emp_strategy_generator> l_strategy_generator = nullptr;
        if(l_strategy_param.value_set())
        {
            l_strategy_generator.reset(emp_strategy_generator_factory::create(l_strategy_param.get_value<std::string>(), l_info));
        }
        else if("new_strategy" == l_feature_name)
        {
            l_strategy_generator.reset(emp_strategy_generator_factory::create("spiral", l_info));
        }
        else if("new_text_strategy" == l_feature_name)
        {
            l_strategy_generator.reset(emp_strategy_generator_factory::create("strategy.txt", l_info));
        }
        else if("system_equation" == l_feature_name || "simplex" == l_feature_name)
        {
            l_strategy_generator.reset(emp_strategy_generator_factory::create("basic", l_info));
        }
        else
        {
            throw quicky_exception::quicky_logic_exception("Should not occur", __LINE__, __FILE__);
        }
        if(l_strategy_generator)
        {
            l_strategy_generator->generate();
        }

        feature_if * l_feature = nullptr;
        if("display_all" == l_feature_name)
        {
            l_feature = new feature_display_all(l_piece_db, l_info, l_gui);
        }
        else if("display_max" == l_feature_name)
        {
            l_feature = new feature_display_max(l_piece_db, l_info, l_gui);
        }
        else if("display_solutions" == l_feature_name)
        {
            l_feature = new feature_display_solutions(l_piece_db, l_info, l_gui);
        }
        else if("dump_solutions" == l_feature_name)
        {
            l_feature = new feature_dump_solutions(l_piece_db, l_info, l_gui, l_dump_file_name);
        }
        else if("dump_summary" == l_feature_name)
        {
            l_feature = new feature_dump_summary(l_dump_file_name, l_info);
        }
        else if("display_dump" == l_feature_name)
        {
            l_feature = new feature_display_dump(l_dump_file_name, l_info, l_gui);
        }
        else if("display_situation" == l_feature_name)
        {
            l_feature = new feature_display_situation(l_initial_situation.get_value<std::string>(), l_info, l_gui);
        }
        else if("compute_stats" == l_feature_name)
        {
            l_feature = new feature_compute_stats(l_piece_db, l_info, l_gui);
        }
        else if("border_exploration" == l_feature_name)
        {
            l_feature = new feature_border_exploration(l_piece_db
                                                      ,l_info
                                                      ,l_initial_situation.get_value<std::string>()
                                                      );
        }
        else if("simplex" == l_feature_name)
        {
            l_feature = new feature_simplex(l_piece_db
                                           ,l_strategy_generator
                                           ,l_info
                                           ,l_initial_situation.get_value<std::string>()
                                           ,l_gui
                                           );
        }
        else if("system_equation" == l_feature_name)
        {
            l_feature = new feature_system_equation(l_piece_db
                                                   ,l_strategy_generator
                                                   ,l_info
                                                   ,l_initial_situation.get_value<std::string>()
                                                   ,l_gui
                                                   );
        }
        else if("new_strategy" == l_feature_name || "new_text_strategy" == l_feature_name)
        {
            auto * l_strategy = new emp_strategy(l_strategy_generator, l_piece_db, l_gui, l_info, l_dump_file_name);
            if(l_initial_situation.value_set())
            {
                l_strategy->set_initial_state(l_initial_situation.get_value<std::string>());
            }
            l_feature = l_strategy;
        }
        else
        {
            throw quicky_exception::quicky_logic_exception("Unsupported feature \"" + l_feature_name + "\"", __LINE__, __FILE__);
        }
        l_feature->run();
        delete l_feature;
    }
    catch(quicky_exception::quicky_runtime_exception & e)
    {
        std::cout << "ERROR : " << e.what() << " at " << e.get_file() << ":" << e.get_line() <<std::endl ;
        return(-1);
    }
    catch(quicky_exception::quicky_logic_exception & e)
    {
        std::cout << "ERROR : " << e.what() << " at " << e.get_file() << ":" << e.get_line() << std::endl ;
        return(-1);
    }
    return 0;
}
//EOF
