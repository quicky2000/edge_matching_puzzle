/* -*- C++ -*- */
/*    This file is part of edge_matching_puzzle
      Copyright (C) 2017  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#include "border_backtracker.h"
#include "light_border_pieces_db.h"
#include "border_color_constraint.h"
#include "border_constraint_generator.h"
#include "octet_array.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

namespace edge_matching_puzzle
{
  //------------------------------------------------------------------------------
  void border_backtracker_kernel(const light_border_pieces_db & p_border_pieces,
				 border_color_constraint  (&p_border_constraints)[23],
				 const octet_array & p_initial_constraint,
				 octet_array & p_solution
				 )
  {
    unsigned int l_index = 0;
    unsigned int l_max_index = 0;
    border_color_constraint l_available_pieces(true);
    bool l_ended = false;
    do
      {
	unsigned int l_previous_index = l_index ? l_index - 1 : 59;
	unsigned int l_piece_id = p_solution.get_octet(l_previous_index);
	unsigned int l_color =  l_piece_id ? p_border_pieces.get_right(l_piece_id - 1) : 0;
	border_color_constraint l_available_transitions = p_border_constraints[l_color];
	l_available_transitions & p_border_constraints[p_initial_constraint.get_octet(l_index)];
	unsigned int l_next_index = l_index < 59 ? l_index + 1 : 0;
	uint64_t l_corner_mask = (0 == l_index || 15 == l_index || 30 == l_index || 45 == l_index) ? 0xF : UINT64_MAX;
	l_available_transitions & l_corner_mask;
	l_available_transitions & l_available_pieces;
	l_available_transitions & (~(( ((uint64_t)1) << p_solution.get_octet(l_index)) - 1));

	int l_ffs = l_available_transitions.ffs();

	// Detect the end in case we have found no solution ( index 0 and no candidate)
	// or in case we are at the end ( next_index = 0 and there is one candidate)
	l_ended = (!l_index && !l_ffs) || (!l_next_index && l_ffs);

	// Remove the piece from list of available pieces if a transition was
	// possible or restablish it to prepare come back to previous state
	unsigned int l_toggled_index = l_ffs ? l_ffs : p_solution.get_octet(l_previous_index);
	l_available_pieces.toggle_bit(l_toggled_index - 1,true);

	// Prepare for next pieces
	p_solution.set_octet(l_index, l_ffs);
	l_index = l_ffs ? l_next_index : l_previous_index;

	l_max_index = ( l_index > l_max_index  && !l_ended ) ? l_index : l_max_index;
 
      }
    while(!l_ended);
    p_solution.set_octet(59,p_solution.get_octet(0) ? p_solution.get_octet(59) : l_max_index);
  }

  //------------------------------------------------------------------------------
  void extract_initial_constraint(const std::string & p_situation_string,
				  octet_array & p_initial_constraint,
				  const light_border_pieces_db & p_border_pieces
				  )
  {
    assert(256 * 4 == p_situation_string.size());
    for(unsigned int l_situation_index = 0 ;
	l_situation_index < 256 ;
	++l_situation_index
	)
      {
	std::string l_piece_id_str = p_situation_string.substr(l_situation_index * 4,3);
	if("---" != l_piece_id_str)
	  {
	    unsigned int l_piece_id = std::stoi(l_piece_id_str);
	    unsigned int l_constraint_index= 0;
	    bool l_meaningful = true;
	    if(l_situation_index < 16)
	      {
		l_constraint_index = l_situation_index;
	      }
	    else if(15 == l_situation_index % 16)
	      {
		l_constraint_index = 15 + (l_situation_index / 16);
	      }
	    else if(15 == l_situation_index / 16)
	      {
		l_constraint_index = 255 - l_situation_index + 30;
	      }
	    else if(0 == l_situation_index % 16)
	      {
		l_constraint_index = 45 - (l_situation_index / 16 ) + 15;
	      }
	    else
	      {
		l_meaningful = false;
	      }
	    if(l_meaningful)
	      {
		p_initial_constraint.set_octet(l_constraint_index, p_border_pieces.get_center(l_piece_id - 1));
	      }
	  }
      }
  }

  //------------------------------------------------------------------------------
  void constraint_to_string(std::string & p_result,
			    const octet_array & p_situation,
			    const unsigned int (&p_border_edges)[60]
			    )
  {
    p_result = "";
    char l_orientation2string[4] = {'N', 'E', 'S', 'W'};
    for(unsigned int l_y = 0;
	l_y < 16;
	++l_y
	)
      {
	for(unsigned int l_x = 0;
	    l_x < 16;
	    ++l_x
	    )
	  {
	    std::stringstream l_stream;
	    if(0 == l_y && 0 == l_x)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(0) << l_orientation2string[(p_border_edges[p_situation.get_octet(0) - 1] + 1) % 4];
		p_result += l_stream.str();
	      }
	    else if(0 == l_y && 15 == l_x)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(15) << l_orientation2string[p_border_edges[p_situation.get_octet(15) - 1]];
		p_result += l_stream.str();
	      }
	    else if(15 == l_y && 15 == l_x)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(30) << l_orientation2string[(p_border_edges[p_situation.get_octet(30) - 1] + 3) % 4];
		p_result += l_stream.str();
	      }
	    else if(15 == l_y && 0 == l_x)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(45) << l_orientation2string[(p_border_edges[p_situation.get_octet(45) - 1] + 2) % 4];
		p_result += l_stream.str();
	      }
	    else if(0 == l_y)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(l_x) << l_orientation2string[p_border_edges[p_situation.get_octet(l_x) - 1]];
		p_result += l_stream.str();
	      }
	    else if(15 == l_x)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(15 + l_y) << l_orientation2string[(p_border_edges[p_situation.get_octet(l_x) - 1] + 3) % 4];
		p_result += l_stream.str();
	      }
	    else if(15 == l_y)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(30 - l_x + 15) << l_orientation2string[(p_border_edges[p_situation.get_octet(l_x) - 1] + 2) % 4];
		p_result += l_stream.str();
	      }
	    else if(0 == l_x)
	      {
		l_stream << std::setw(3) << p_situation.get_octet(45 - l_y + 15) << l_orientation2string[(p_border_edges[p_situation.get_octet(l_x) - 1] + 1) % 4];
		p_result += l_stream.str();
	      }
	    else
	      {
		p_result += "----";
	      }
	  }
      }
  }
}
// EOF
