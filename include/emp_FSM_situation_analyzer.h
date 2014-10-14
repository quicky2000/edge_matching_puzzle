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

//#define ADDITIONAL_CHECK

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

      inline static void precompute_constraints(const unsigned int & p_x,
                                                const unsigned int & p_y,
                                                const emp_FSM_situation & p_situation,
                                                unsigned int & p_max_neighbours_nb,
                                                std::set<emp_constraint> & p_constraints,
                                                std::vector<precomputed_constraint> & p_precomputed_constraints,
                                                const emp_FSM_info & p_info);

      inline void compute_constraints(const unsigned int & p_x,
                                      const unsigned int & p_y,
                                      const emp_FSM_situation & p_situation,
                                      unsigned int & p_max_neighbours_nb,
                                      std::set<emp_constraint> & p_constraints)const;


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
#ifdef ADDITIONAL_CHECK
      unsigned int l_additional_x = 0;
      unsigned int l_additional_y = 0;
#endif // ADDITIONAL_CHECK
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

          precompute_constraints(l_x,l_y,l_situation,l_max_neighbours_nb,l_constraints,l_precomputed_constraints,m_info);
          m_precomputed_transition_infos[l_index] = new precomputed_transition_info((emp_types::t_kind)(4 - l_max_neighbours_nb),std::pair<unsigned int,unsigned int>(l_x,l_y),l_constraints,l_precomputed_constraints);
#ifdef ADDITIONAL_CHECK
          if(l_index >= (2 * m_info.get_width() + 2 * ( m_info.get_height() - 2)) && !l_situation.contains_piece(l_additional_x,l_additional_y))
            {
              unsigned int l_max_neighbours_nb = 0;
              std::set<emp_constraint> l_constraints;
              std::vector<precomputed_constraint> l_precomputed_constraints;
              precompute_constraints(l_additional_x,l_additional_y,l_situation,l_max_neighbours_nb,l_constraints,l_precomputed_constraints,m_info); 
              m_precomputed_transition_infos[l_index - 1]->set_check_info(*(new precomputed_transition_info((emp_types::t_kind)(4 - l_max_neighbours_nb),std::pair<unsigned int,unsigned int>(l_additional_x,l_additional_y),l_constraints,l_precomputed_constraints)));
            }
#endif // ADDITIONAL_CHECK

	  // Compute next position
	  if(m_info.get_width() * m_info.get_height() > l_index + 1)
	    {
              if(l_x >= l_y)
                {
                  if(l_x + 1 < m_info.get_width() && !l_situation.contains_piece(l_x + 1, l_y))
                    {
#ifdef ADDITIONAL_CHECK
                      l_additional_x = l_x;
                      l_additional_y = l_y + 1;
#endif // ADDITIONAL_CHECK
                      l_x = l_x + 1;
                    }
                  else if(l_y + 1 < m_info.get_height()  && !l_situation.contains_piece(l_x, l_y + 1))
                    {
#ifdef ADDITIONAL_CHECK
                      l_additional_x = l_x - 1;
                      l_additional_y = l_y;
#endif // ADDITIONAL_CHECK
                      l_y = l_y + 1;
                    }
                  else 
                    {
                      assert(l_x  && !l_situation.contains_piece(l_x - 1, l_y));
#ifdef ADDITIONAL_CHECK
                      l_additional_x = l_x;
                      l_additional_y = l_y - 1;
#endif // ADDITIONAL_CHECK
                      l_x = l_x - 1;
                    }
                }
              else
                {
                  if(l_x  && !l_situation.contains_piece(l_x - 1, l_y))
                    {
#ifdef ADDITIONAL_CHECK
                      l_additional_x = l_x;
                      l_additional_y = l_y - 1;
#endif // ADDITIONAL_CHECK
                      l_x = l_x - 1;
                    }
                  else if(l_y  && !l_situation.contains_piece(l_x, l_y - 1)) 
                    {
#ifdef ADDITIONAL_CHECK
                      l_additional_x = l_x + 1;
                      l_additional_y = l_y;
#endif // ADDITIONAL_CHECK
                      l_y = l_y - 1;
                    }
                  else
                    {
                      assert(l_x + 1 < m_info.get_width() && !l_situation.contains_piece(l_x + 1, l_y));
#ifdef ADDITIONAL_CHECK
                      l_additional_x = l_x;
                      l_additional_y = l_y + 1;
#endif // ADDITIONAL_CHECK
                      l_x = l_x + 1;
                    }
                }
	    }

	}
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation_analyzer::precompute_constraints(const unsigned int & p_x,
                                                            const unsigned int & p_y,
                                                            const emp_FSM_situation & p_situation,
                                                            unsigned int & p_max_neighbours_nb,
                                                            std::set<emp_constraint> & p_constraints,
                                                            std::vector<precomputed_constraint> & p_precomputed_constraints,
                                                            const emp_FSM_info & p_info)
    {
      if(0 < p_x)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x - 1,p_y)) p_precomputed_constraints.push_back(precomputed_constraint(p_x - 1,p_y,emp_types::t_orientation::EAST,emp_types::t_orientation::WEST));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::WEST));
        }
      if(0 < p_y)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x,p_y - 1)) p_precomputed_constraints.push_back(precomputed_constraint(p_x,p_y - 1,emp_types::t_orientation::SOUTH,emp_types::t_orientation::NORTH));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::NORTH));
        }
      if(p_x < p_info.get_width() - 1)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x + 1,p_y)) p_precomputed_constraints.push_back(precomputed_constraint(p_x + 1,p_y,emp_types::t_orientation::WEST,emp_types::t_orientation::EAST));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::EAST));
        }
      if(p_y < p_info.get_height() - 1)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x,p_y + 1)) p_precomputed_constraints.push_back(precomputed_constraint(p_x,p_y + 1,emp_types::t_orientation::NORTH,emp_types::t_orientation::SOUTH));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::SOUTH));
        }
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation_analyzer::compute_constraints(const unsigned int & p_x,
                                                         const unsigned int & p_y,
                                                         const emp_FSM_situation & p_situation,
                                                         unsigned int & p_max_neighbours_nb,
                                                         std::set<emp_constraint> & p_constraints)const
    {
 
      if(0 < p_x)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x - 1,p_y)) p_constraints.insert(emp_constraint(m_piece_db.get_piece(p_situation.get_piece(p_x - 1,p_y).first).get_color(emp_types::t_orientation::EAST,p_situation.get_piece(p_x - 1,p_y).second),emp_types::t_orientation::WEST));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::WEST));
        }
      if(0 < p_y)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x,p_y - 1)) p_constraints.insert(emp_constraint(m_piece_db.get_piece(p_situation.get_piece(p_x,p_y - 1).first).get_color(emp_types::t_orientation::SOUTH,p_situation.get_piece(p_x,p_y - 1).second),emp_types::t_orientation::NORTH));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::NORTH));
        }
      if(p_x < m_info.get_width() - 1)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x + 1,p_y)) p_constraints.insert(emp_constraint(m_piece_db.get_piece(p_situation.get_piece(p_x + 1,p_y).first).get_color(emp_types::t_orientation::WEST,p_situation.get_piece(p_x + 1,p_y).second),emp_types::t_orientation::EAST));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::EAST));
        }
      if(p_y < m_info.get_height() - 1)
        {
          ++p_max_neighbours_nb;
          if(p_situation.contains_piece(p_x,p_y + 1)) p_constraints.insert(emp_constraint(m_piece_db.get_piece(p_situation.get_piece(p_x,p_y + 1).first).get_color(emp_types::t_orientation::NORTH,p_situation.get_piece(p_x,p_y + 1).second),emp_types::t_orientation::SOUTH));
        }
      else
        {
          p_constraints.insert(emp_constraint(0,emp_types::t_orientation::SOUTH));
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
#ifdef ADDITIONAL_CHECK
                     if(l_usable && l_precomputed_transition_info.get_check_info())
                      {
                        std::set<emp_constraint> l_constraints = l_precomputed_transition_info.get_check_info()->get_constraints();
                        for(auto l_iter : l_precomputed_transition_info.get_check_info()->get_precomputed_constraints())
                          {
                            if(p_situation.contains_piece(l_iter.get_x(),l_iter.get_y()))
                              {
                                const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_iter.get_x(),l_iter.get_y());
                                l_constraints.insert(emp_constraint(m_piece_db.get_piece(l_piece.first).get_color(l_iter.get_color_orient(),l_piece.second),l_iter.get_side_orient()));
                              }
                            else
                              {
                                l_constraints.insert(emp_constraint(m_piece_db.get_piece(l_piece.first).get_color(l_iter.get_color_orient(),l_piece.second),l_iter.get_side_orient()));                                
                              }
                          }
                        std::vector<emp_types::t_oriented_piece> l_checked_pieces;
                        m_piece_db.get_pieces(l_precomputed_transition_info.get_check_info()->get_kind(),l_constraints,l_checked_pieces);
                        l_usable = l_checked_pieces.size();
                      }
#endif // ADDITIONAL_CHECK
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

