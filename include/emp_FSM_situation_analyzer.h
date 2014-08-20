/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
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
#ifndef EMP_FSM_SITUATION_ANALYZER_H
#define EMP_FSM_SITUATION_ANALYZER_H

#include "FSM_situation_analyzer.h"
#include "emp_FSM_situation.h"
#include "emp_FSM_transition.h"
#include "emp_constraint.h"
#include "emp_piece_db.h"

namespace edge_matching_puzzle
{
  class emp_FSM_situation_analyzer: public FSM_base::FSM_situation_analyzer<emp_FSM_situation,emp_FSM_transition>
    {
    public:
      inline emp_FSM_situation_analyzer(const emp_FSM_info & p_info,
					const emp_piece_db & p_piece_db);
      // Methods inherited from FSM_situation_analyzer
      inline const std::string & get_class_name(void)const;
      inline std::vector<const emp_FSM_transition*> & get_transitions(emp_FSM_situation & p_situation);
      
      // Specific methods
    private:
      const emp_piece_db & m_piece_db;
      const emp_FSM_info & m_info;
      static const std::string m_class_name;    
    };

  //----------------------------------------------------------------------------
  emp_FSM_situation_analyzer::emp_FSM_situation_analyzer(const emp_FSM_info & p_info,
							 const emp_piece_db & p_piece_db):
    m_piece_db(p_piece_db),
    m_info(p_info)
    {
    }

    //----------------------------------------------------------------------------
    const std::string & emp_FSM_situation_analyzer::get_class_name(void)const
      {
	return m_class_name;
      }

    //----------------------------------------------------------------------------
    std::vector<const emp_FSM_transition*> & emp_FSM_situation_analyzer::get_transitions(emp_FSM_situation & p_situation)
      {
	std::vector<const emp_FSM_transition*> & l_result = *(new std::vector<const emp_FSM_transition*>());

	const std::set<std::pair<unsigned int, unsigned int> > & l_available_positions = p_situation.get_context()->get_available_positions();
	for(auto l_position: l_available_positions)
	  {
	    unsigned int l_x = l_position.first;
	    unsigned int l_y = l_position.second;

	    unsigned int l_max_neighbours_nb = 0;
            std::set<emp_constraint> l_constraints;
	    if(0 < l_x)
	      {
		++l_max_neighbours_nb;
		if(p_situation.contains_piece(l_x - 1,l_y))
		  {
		    const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_x - 1,l_y);
		    l_constraints.insert(emp_constraint(m_piece_db.get_piece(l_piece.first).get_color(emp_types::t_orientation::EAST,l_piece.second),emp_types::t_orientation::WEST));
		  }
	      }
            else
              {
		    l_constraints.insert(emp_constraint(0,emp_types::t_orientation::WEST));
              }
	    if(0 < l_y)
	      {
		++l_max_neighbours_nb;
		if(p_situation.contains_piece(l_x,l_y - 1))
		  {
		    const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_x,l_y - 1);
		    l_constraints.insert(emp_constraint(m_piece_db.get_piece(l_piece.first).get_color(emp_types::t_orientation::SOUTH,l_piece.second),emp_types::t_orientation::NORTH));
		  }
	      }
            else
              {
		    l_constraints.insert(emp_constraint(0,emp_types::t_orientation::NORTH));
              }
	    if(l_x < m_info.get_width() - 1)
	      {
		++l_max_neighbours_nb;
		if(p_situation.contains_piece(l_x + 1,l_y))
		  {
		    const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_x + 1,l_y);
		    l_constraints.insert(emp_constraint(m_piece_db.get_piece(l_piece.first).get_color(emp_types::t_orientation::WEST,l_piece.second),emp_types::t_orientation::EAST));
		  }
	      }
            else
              {
		    l_constraints.insert(emp_constraint(0,emp_types::t_orientation::EAST));
              }
	    if(l_y < m_info.get_height() - 1)
	      {
		++l_max_neighbours_nb;
		if(p_situation.contains_piece(l_x,l_y + 1))
		  {
		    const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_x,l_y + 1);
		    l_constraints.insert(emp_constraint(m_piece_db.get_piece(l_piece.first).get_color(emp_types::t_orientation::NORTH,l_piece.second),emp_types::t_orientation::SOUTH));
		  }
	      }
            else
              {
		    l_constraints.insert(emp_constraint(0,emp_types::t_orientation::SOUTH));
              }
            std::vector<emp_types::t_oriented_piece> l_pieces;
            std::set<emp_types::t_piece_id> l_transition_used_pieces;
            m_piece_db.get_pieces((emp_types::t_kind)(4 - l_max_neighbours_nb),l_constraints,l_pieces);
            bool l_found = false;
            if(l_pieces.size())
              {
                for(auto l_piece : l_pieces)
                  {
                    if(!p_situation.get_context()->is_used(l_piece.first))
                      {
                        bool l_usable = true;
                        const std::set<emp_types::t_piece_id> * const l_identical_pieces = m_piece_db.get_identical_pieces(l_piece.first);
                        if(l_identical_pieces)
                          {
                            for(auto l_identic_iter : *l_identical_pieces)
                              {
                                if(l_transition_used_pieces.end() != l_transition_used_pieces.find(l_identic_iter))
                                  {
                                    l_usable = false;
                                    break;
                                  }
                              }
                          }
                        if(l_usable)
                          {
                            l_result.push_back(new emp_FSM_transition(l_x,l_y,l_piece));
                            l_transition_used_pieces.insert(l_piece.first);
                            l_found = true;
                          }
                      }
                  }
              }
            if(!l_found)
              {
                p_situation.set_invalid();
                l_result.clear();
                return l_result;
              }
	  }
	return l_result;
      }

}
#endif // EMP_FSM_SITUATION_ANALYZER_H
//EOF

