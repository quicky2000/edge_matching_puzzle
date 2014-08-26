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
#include "precomputed_transition_info.h"

namespace edge_matching_puzzle
{
  class emp_FSM_situation_analyzer: public FSM_base::FSM_situation_analyzer<emp_FSM_situation,emp_FSM_transition>
    {
    public:
      inline emp_FSM_situation_analyzer(const emp_FSM_info & p_info,
					const emp_piece_db & p_piece_db);
      inline ~emp_FSM_situation_analyzer(void);
      // Methods inherited from FSM_situation_analyzer
      inline const std::string & get_class_name(void)const;
      inline std::vector<const emp_FSM_transition*> & get_transitions(emp_FSM_situation & p_situation);
      
      // Specific methods
    private:
      const emp_piece_db & m_piece_db;
      const emp_FSM_info & m_info;
      precomputed_transition_info ** m_precomputed_transition_infos;
      static const std::string m_class_name;    
    };

  //----------------------------------------------------------------------------
  emp_FSM_situation_analyzer::emp_FSM_situation_analyzer(const emp_FSM_info & p_info,
							 const emp_piece_db & p_piece_db):
    m_piece_db(p_piece_db),
    m_info(p_info),
    m_precomputed_transition_infos(new precomputed_transition_info*[m_info.get_width()*m_info.get_height()])
    {
      unsigned int l_x = 0;
      unsigned int l_y = 0;
      emp_FSM_situation l_situation;
      l_situation.set_context(*(new emp_FSM_context(m_info.get_width()*m_info.get_height())));
      for(unsigned int l_index = 0 ;
	  l_index < m_info.get_width()*m_info.get_height();
	  ++l_index)
	{
	  l_situation.set_piece(l_x,l_y,emp_types::t_oriented_piece(1,(edge_matching_puzzle::emp_types::t_orientation)1));
	  unsigned int l_max_neighbours_nb = 0;
	  std::set<emp_constraint> l_constraints;
	  std::vector<precomputed_constraint> l_precomputed_constraints;
	  std::pair<unsigned int,unsigned int> l_next_position;
	  if(0 < l_x)
	    {
	      ++l_max_neighbours_nb;
	      if(l_situation.contains_piece(l_x - 1,l_y)) l_precomputed_constraints.push_back(precomputed_constraint(l_x - 1,l_y,emp_types::t_orientation::EAST,emp_types::t_orientation::WEST));
	    }
	  else
	    {
	      l_constraints.insert(emp_constraint(0,emp_types::t_orientation::WEST));
	    }
	  if(0 < l_y)
	      {
		++l_max_neighbours_nb;
		if(l_situation.contains_piece(l_x,l_y - 1)) l_precomputed_constraints.push_back(precomputed_constraint(l_x,l_y - 1,emp_types::t_orientation::SOUTH,emp_types::t_orientation::NORTH));
	      }
	  else
	    {
	      l_constraints.insert(emp_constraint(0,emp_types::t_orientation::NORTH));
	    }
	  if(l_x < m_info.get_width() - 1)
	    {
	      ++l_max_neighbours_nb;
	      if(l_situation.contains_piece(l_x + 1,l_y)) l_precomputed_constraints.push_back(precomputed_constraint(l_x + 1,l_y,emp_types::t_orientation::WEST,emp_types::t_orientation::EAST));
	    }
	  else
	    {
	      l_constraints.insert(emp_constraint(0,emp_types::t_orientation::EAST));
	    }
	  if(l_y < m_info.get_height() - 1)
	    {
	      ++l_max_neighbours_nb;
	      if(l_situation.contains_piece(l_x,l_y + 1)) l_precomputed_constraints.push_back(precomputed_constraint(l_x,l_y + 1,emp_types::t_orientation::NORTH,emp_types::t_orientation::SOUTH));
	    }
	  else
	    {
	      l_constraints.insert(emp_constraint(0,emp_types::t_orientation::SOUTH));
	    }
          m_precomputed_transition_infos[l_index] = new precomputed_transition_info((emp_types::t_kind)(4 - l_max_neighbours_nb),std::pair<unsigned int,unsigned int>(l_x,l_y),l_constraints,l_precomputed_constraints);

	  // Compute next position
	  if(m_info.get_width() * m_info.get_height() > l_index + 1)
	    {
	      if(l_x >= l_y)
		{
		  if(l_x + 1 < m_info.get_width() && !l_situation.contains_piece(l_x + 1, l_y))
		    {
		      l_x = l_x + 1;
		    }
		  else if(l_y + 1 < m_info.get_height()  && !l_situation.contains_piece(l_x, l_y + 1))
		    {
		      l_y = l_y + 1;
		    }
		  else 
		    {
		      assert(l_x  && !l_situation.contains_piece(l_x - 1, l_y));
		      l_x = l_x - 1;
		    }
		}
	      else
		{
		  if(l_x  && !l_situation.contains_piece(l_x - 1, l_y))
		    {
		      l_x = l_x - 1;
		    }
		  else if(l_y  && !l_situation.contains_piece(l_x, l_y - 1)) 
		    {
		      l_y = l_y - 1;
		    }
		  else
		    {
		      assert(l_x + 1 < m_info.get_width() && !l_situation.contains_piece(l_x + 1, l_y));
		      l_x = l_x + 1;
		    }
		}
	    }

	}
    }

    //----------------------------------------------------------------------------
    emp_FSM_situation_analyzer::~emp_FSM_situation_analyzer(void)
      {
       for(unsigned int l_index = 0 ;
	  l_index < m_info.get_width()*m_info.get_height();
	  ++l_index)
         {
           delete m_precomputed_transition_infos[l_index];
         }
       delete[] m_precomputed_transition_infos;
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
        unsigned int l_index = p_situation.get_level();
        std::vector<emp_types::t_oriented_piece> l_pieces;
        std::set<emp_types::t_piece_id> l_transition_used_pieces;

        precomputed_transition_info & l_precomputed_transition_info = *(m_precomputed_transition_infos[l_index]);

        std::set<emp_constraint> l_constraints = l_precomputed_transition_info.get_constraints();

        for(auto l_iter : l_precomputed_transition_info.get_precomputed_constraints())
          {
            const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_iter.get_x(),l_iter.get_y());
            l_constraints.insert(emp_constraint(m_piece_db.get_piece(l_piece.first).get_color(l_iter.get_color_orient(),l_piece.second),l_iter.get_side_orient()));
          }
        m_piece_db.get_pieces(l_precomputed_transition_info.get_kind(),l_constraints,l_pieces);

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
                        l_result.push_back(new emp_FSM_transition(l_precomputed_transition_info.get_position().first,l_precomputed_transition_info.get_position().second,l_piece));
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
	return l_result;
      }
}
#endif // EMP_FSM_SITUATION_ANALYZER_H
//EOF

