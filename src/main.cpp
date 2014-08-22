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

#include "parameter_manager.h"
#include "emp_piece.h"
#include "emp_pieces_parser.h"
#include "emp_gui.h"
#include "emp_piece_db.h"
#include "emp_FSM.h"
#include "emp_FSM_UI.h"

#include "algorithm_deep_raw.h"

#include "quicky_exception.h"
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <set>
#include <map>

using namespace edge_matching_puzzle;
using namespace parameter_manager;

//------------------------------------------------------------------------------
int main(int argc,char ** argv)
{
  try
    {
      // Defining application command line parameters
      parameter_manager::parameter_manager l_param_manager("edge_matching_puzzle.exe","--",2);
      parameter_if l_definition_file("definition",false);
      l_param_manager.add(l_definition_file);
      parameter_if l_ressources_path("ressources",false);
      l_param_manager.add(l_ressources_path);

      // Treating parameters
      l_param_manager.treat_parameters(argc,argv);

      // Get puzzle description
      std::vector<emp_piece> l_pieces;
      emp_pieces_parser l_piece_parser(l_definition_file.get_value<std::string>().c_str());
      unsigned int l_width = 0;
      unsigned int l_height = 0;
      l_piece_parser.parse(l_width,l_height,l_pieces);

      std::cout << l_pieces.size() << " pieces loaded" << std::endl ;
      if(l_pieces.size() != l_width * l_height)
        {
          std::stringstream l_stream_width;
          l_stream_width << l_width;
          std::stringstream l_stream_height;
          l_stream_height << l_height;
          std::stringstream l_stream_nb_pieces;
          l_stream_nb_pieces << l_pieces.size();
          throw quicky_exception::quicky_logic_exception("Inconsistency between puzzle dimensions ("+l_stream_width.str()+"*"+l_stream_height.str()+") and piece number ("+l_stream_nb_pieces.str()+")",__LINE__,__FILE__);
        }


      emp_gui l_gui(l_width,l_height,l_ressources_path.get_value<std::string>().c_str(),l_pieces);

      emp_piece_db l_piece_db(l_pieces,l_width,l_height);
      emp_FSM_info l_info(l_width,l_height);

      emp_FSM_situation::init(l_info);
      emp_FSM l_FSM(l_info,l_piece_db);
      emp_FSM_UI l_emp_FSM_UI(l_gui);

      FSM_framework::algorithm_deep_raw l_algo;
      l_algo.set_fsm(&l_FSM);
      l_algo.set_fsm_ui(&l_emp_FSM_UI);
      l_algo.run();

    }
  catch(quicky_exception::quicky_runtime_exception & e)
    {
      std::cout << "ERROR : " << e.what() << std::endl ;
      return(-1);
    }
  catch(quicky_exception::quicky_logic_exception & e)
    {
      std::cout << "ERROR : " << e.what() << std::endl ;
      return(-1);
    }
  return 0;
  
}
//EOF
