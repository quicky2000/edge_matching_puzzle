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

}
// EOF
