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
#ifndef _SEQUENTIAL_BORDER_BACKTRACKER_H_
#define _SEQUENTIAL_BORDER_BACKTRACKER_H_

#include "light_border_pieces_db.h"
#include "border_color_constraint.h"
#include "octet_array.h"
#include <iostream>

namespace edge_matching_puzzle
{
  /**
     Border backtracker optimised to solve several border constraints in a
     logical order by reusing previoulsy computed solution
  **/
  class sequential_border_backtracker
  {
  public:
    inline sequential_border_backtracker(void);
    inline void run(const light_border_pieces_db & p_border_pieces,
		    const border_color_constraint  (&p_border_constraints)[23],
		    const octet_array & p_initial_constraint,
		    octet_array & p_solution,
		    unsigned int p_start_index=0
		    );

    inline unsigned int get_max_index(void)const;
  private:
    inline void save_best_solution(const octet_array & p_solution,
				   const border_color_constraint & p_available_pieces
				   );

    /**
       Store max reached index
    **/
    unsigned int m_max_index;

    /**
       Store minimum changed index from last best solution save
    **/
    unsigned int m_min_best_index;

    /**
       Store the best achieved solution
    **/
    octet_array m_best_solution;

    /**
       Store available pieces corresponding to best solution
    **/
    border_color_constraint m_best_available_pieces;

    /**
       Store for which max index this index has been touched
    **/
    unsigned int m_corresponding_max_index[60];
  };

  //----------------------------------------------------------------------------
  void sequential_border_backtracker::save_best_solution(const octet_array & p_solution,
							 const border_color_constraint & p_available_pieces
							 )
  {
#ifdef DEBUG
    std::cout << "Save best solution [" << m_min_best_index << "," << m_max_index << "[" << std::endl;
#endif // DEBUG
    m_best_available_pieces = p_available_pieces;
    m_corresponding_max_index[m_min_best_index] = m_max_index;
    for(unsigned int l_index = m_min_best_index;
	l_index < m_max_index;
	++l_index
	)
      {
	m_best_solution.set_octet(l_index,p_solution.get_octet(l_index));
      }
    m_min_best_index = m_max_index;
#ifdef DEBUG
    for(unsigned int l_index = 0;
	l_index < m_max_index;
	++l_index
	)
      {
	std::cout << m_best_solution.get_octet(l_index) << " ";
      }
    std::cout << std::endl;
#endif // DEBUG
  }

  //------------------------------------------------------------------------------
  sequential_border_backtracker::sequential_border_backtracker(void):
    m_max_index(0),
    m_min_best_index(0)
  {
    for(unsigned int l_index =0;
	l_index < 60;
	++l_index
	)
      {
	m_corresponding_max_index[l_index] = 0;
      }
  }

  //------------------------------------------------------------------------------
  void sequential_border_backtracker::run(const light_border_pieces_db & p_border_pieces,
					  const border_color_constraint  (&p_border_constraints)[23],
					  const octet_array & p_initial_constraint,
					  octet_array & p_solution,
					  unsigned int p_start_index
					  )
  {
    unsigned int l_index = p_start_index;
    m_max_index = 0;
    border_color_constraint l_available_pieces(true);
    m_min_best_index = 0;
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

	if(l_index > m_max_index  && !l_ended)
	  {
#ifdef DEBUG
	    std::cout << "New max best index = " << l_index << std::endl;
	    for(unsigned int l_display_index = 0;
		l_display_index < l_index;
		++l_display_index
		)
	      {
		std::cout << p_solution.get_octet(l_display_index) << " " ;
	      }
	    std::cout << std::endl;
#endif // DEBUG
	    m_max_index = l_index;
	    save_best_solution(p_solution,
			       l_available_pieces
			       );
	  }
	else if(l_index < m_min_best_index)
	  {
#ifdef DEBUG
	    std::cout << "New min best index = " << l_index << std::endl;
#endif // DEBUG
	    m_min_best_index = l_index;
	  }
      }
    while(!l_ended);
#ifdef DEBUG
    std::cout << "========================================================" << std::endl;
    std::cout << "Best solution : [0," << m_max_index << "[" << std::endl;
    for(unsigned int l_index = 0;
	l_index < m_max_index;
	++l_index)
      {
	std::cout << m_best_solution.get_octet(l_index) << " ";
      }
    std::cout << std::endl;
#endif // DEBUG
  }

  //------------------------------------------------------------------------------
  unsigned int sequential_border_backtracker::get_max_index(void)const
  {
    return m_max_index;
  }
}
#endif // _SEQUENTIAL_BORDER_BACKTRACKER_H_
// EOF
