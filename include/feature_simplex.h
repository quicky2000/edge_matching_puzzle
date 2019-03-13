/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
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
#ifndef FEATURE_SIMPLEX_H
#define FEATURE_SIMPLEX_H

#include "feature_if.h"
#include "emp_piece_db.h"
#include "emp_FSM_info.h"
#include "emp_FSM_situation.h"
#include "simplex_variable.h"
#include "safe_types.h"
#include "fract.h"
#include "simplex_solver.h"
#include "emp_simplex_listener.h"
#include <string>
#include <vector>

namespace edge_matching_puzzle
{
  class feature_simplex:public feature_if
  {
  public:
    inline feature_simplex(const emp_piece_db & p_db
                          ,const emp_FSM_info & p_info
                          ,const std::string & p_initial_situation
			              );

    // Virtual methods inherited from feature_if
    inline void run(void);
    // End of virtual methods inherited from feature_if
    inline ~feature_simplex(void);

  private:
    inline void add_equations(const std::vector<simplex_variable*> & p_variables_pos1
                             ,const std::vector<simplex_variable*> & p_variables_pos2
                             ,unsigned int & p_equation_index
                             ,bool p_horizontal
			                 );

    inline unsigned int get_nb_piece_possibility(const emp_types::t_kind & p_kind
                                                ,bool p_minor = false
						                        );

    inline emp_types::t_kind get_position_kind(const unsigned int & p_x
                                              ,const unsigned int & p_y
					                          );

    inline unsigned int compute_combination(const emp_types::t_kind & p_kind1
                                           ,const emp_types::t_kind & p_kind2
					                       );

    inline void determine_simplex_parameters(const emp_piece_db & p_db);

    /**
       Compute index related to position X,Y
       @param X position
       @param Y position
       @return index related to position
    */
    inline unsigned int get_position_index(const unsigned int & p_x
                                          ,const unsigned int & p_y
					                      ) const;

    /**
       Compute the number of equations for a couple of positions
       There is an equation for each couple of variables where variable.piece_id
       from p_variable_pos1 is different from variable.piece_id from
       p_variable_pos2
       @param list of variables for position 1
       @param list of variables for position 2
       @return number of equations
     */
    static inline uint64_t get_nb_equations(const std::vector<simplex_variable*> & p_variables_pos1
                                           ,const std::vector<simplex_variable*> & p_variables_pos2
					                       );

    const emp_piece_db & m_db;
    const emp_FSM_info & m_info;
    emp_FSM_situation m_situation;
    emp_types::bitfield m_available_corners;
    emp_types::bitfield m_available_borders;
    emp_types::bitfield m_available_centers;
    emp_types::bitfield * const m_available_pieces[3];
    std::vector<simplex_variable*> m_simplex_variables;

    typedef simplex::simplex_solver<quicky_utils::fract<quicky_utils::safe_int32_t>> simplex_t;
    simplex_t * m_simplex;
  };
 
  //----------------------------------------------------------------------------
  feature_simplex::feature_simplex(const emp_piece_db & p_db
                                  ,const emp_FSM_info & p_info
                                  ,const std::string & p_initial_situation
				                  )
	: m_db(p_db)
	, m_info(p_info)
	, m_available_corners(4 * p_db.get_nb_pieces(emp_types::t_kind::CORNER),true)
	, m_available_borders(4 * p_db.get_nb_pieces(emp_types::t_kind::BORDER),true)
	, m_available_centers(4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER),true)
	, m_available_pieces{&m_available_centers, &m_available_borders, &m_available_corners}
	, m_simplex(nullptr)
  {
      // Initialise situation with initial situation string
      m_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));
      if("" != p_initial_situation)
      {
          m_situation.set(p_initial_situation);
      }

      // Mark used piece as unavailable
      for(unsigned int l_y = 0;
	      l_y < m_info.get_height();
	      ++l_y
	     )
      {
          for(unsigned int l_x = 0;
	          l_x < m_info.get_width();
	          ++l_x
	         )
          {
              if(m_situation.contains_piece(l_x,l_y))
              {
                  const emp_types::t_oriented_piece & l_piece = m_situation.get_piece(l_x,l_y);
                  emp_types::t_piece_id l_id = l_piece.first;
                  m_available_pieces[(unsigned int)p_db.get_piece(l_id).get_kind()]->set(0x0,4,4 * p_db.get_kind_index(l_id));
              }
          }
      }
      emp_types::bitfield l_matching_corners(4 * p_db.get_nb_pieces(emp_types::t_kind::CORNER));
      emp_types::bitfield l_matching_borders(4 * p_db.get_nb_pieces(emp_types::t_kind::BORDER));
      emp_types::bitfield l_matching_centers(4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER));
      emp_types::bitfield * const l_matching_pieces[3] = {&l_matching_centers,&l_matching_borders,&l_matching_corners};

      // Store all simplex variables related to a position index
      // Position index = width * Y + X)
      std::vector<simplex_variable*> * l_position_variables = new std::vector<simplex_variable*>[p_info.get_width() * p_info.get_height()];

      // Store all simplex variables related to a piece id
      // index 0 correspond to piece id 1)
      std::vector<simplex_variable*> * l_piece_id_variables = new std::vector<simplex_variable*>[p_info.get_width() * p_info.get_height()];

      // Determine for each position which piece match constraints
      for(unsigned int l_y = 0;
          l_y < m_info.get_height();
	      ++l_y
	     )
      {
          for(unsigned int l_x = 0;
	          l_x < m_info.get_width();
	          ++l_x
	         )
          {
              if(!m_situation.contains_piece(l_x,l_y))
              {
                  // Compute surrounding constraints

                  // Compute EAST constraint
                  emp_types::t_binary_piece l_east_constraint = 0x0;
                  if(l_x < m_info.get_width() - 1)
                  {
                      if(m_situation.contains_piece(l_x + 1,l_y))
                      {
                          const emp_types::t_oriented_piece l_oriented_piece = m_situation.get_piece(l_x + 1 ,l_y);
                          const emp_piece & l_east_piece = p_db.get_piece(l_oriented_piece.first);
                          l_east_constraint = l_east_piece.get_color(emp_types::t_orientation::WEST,l_oriented_piece.second);
                      }
                  }
                  else
                  {
                      l_east_constraint = p_db.get_border_color_id();
                  }
                  emp_types::t_binary_piece l_constraint = l_east_constraint;

                  // Compute NORTH constraint
                  emp_types::t_binary_piece l_north_constraint = 0x0;
                  if(l_y)
                  {
                      if(m_situation.contains_piece(l_x,l_y - 1))
                      {
                          const emp_types::t_oriented_piece l_oriented_piece = m_situation.get_piece(l_x,l_y - 1);
                          const emp_piece & l_north_piece = p_db.get_piece(l_oriented_piece.first);
                          l_north_constraint = l_north_piece.get_color(emp_types::t_orientation::SOUTH,l_oriented_piece.second);
                      }
                  }
                  else
                  {
                      l_north_constraint = p_db.get_border_color_id();
                  }
                  l_constraint = (l_constraint << p_db.get_color_id_size()) | l_north_constraint;

                  // Compute WEST constraint
                  emp_types::t_binary_piece l_west_constraint = 0x0;
                  if(l_x > 0)
                  {
                      if(m_situation.contains_piece(l_x - 1,l_y))
                      {
                          const emp_types::t_oriented_piece l_oriented_piece = m_situation.get_piece(l_x - 1,l_y);
                          const emp_piece & l_west_piece = p_db.get_piece(l_oriented_piece.first);
                          l_west_constraint = l_west_piece.get_color(emp_types::t_orientation::EAST,l_oriented_piece.second);
                      }
                  }
                  else
                  {
                      l_west_constraint = p_db.get_border_color_id();
                  }
                  l_constraint = (l_constraint << p_db.get_color_id_size()) | l_west_constraint;

                  // Compute SOUTH constraint
                  emp_types::t_binary_piece l_south_constraint = 0x0;
                  if(l_y < m_info.get_height() -1)
                  {
                      if(m_situation.contains_piece(l_x,l_y + 1))
                      {
                          const emp_types::t_oriented_piece l_oriented_piece = m_situation.get_piece(l_x,l_y + 1);
                          const emp_piece & l_south_piece = p_db.get_piece(l_oriented_piece.first);
                          l_south_constraint = l_south_piece.get_color(emp_types::t_orientation::NORTH,l_oriented_piece.second);
                      }
                  }
                  else
                  {
                      l_south_constraint = p_db.get_border_color_id();
                  }
                  l_constraint = (l_constraint << p_db.get_color_id_size()) | l_south_constraint;

                  emp_types::t_kind l_type = get_position_kind(l_x,l_y);
                  // Compute pieces matching to constraint
                  l_matching_pieces[(unsigned int)l_type]->apply_and(p_db.get_pieces(l_constraint),*m_available_pieces[(unsigned int)l_type]);

                  // Iterating on matching pieces
                  emp_types::bitfield l_loop_pieces(*l_matching_pieces[(unsigned int)l_type]);
                  int l_ffs = 0;
                  while((l_ffs = l_loop_pieces.ffs()) != 0)
                  {
                      // We decrement because 0 mean no piece in other cases this
                      // is the index of oriented piece in piece list by kind
                      unsigned int l_piece_kind_id = l_ffs - 1;
                      l_loop_pieces.set(0,1,l_piece_kind_id);
                      const emp_types::t_binary_piece l_piece = p_db.get_piece(l_type,l_piece_kind_id);
                      unsigned int l_truncated_piece = l_piece >> (4 * p_db.get_color_id_size());
                      emp_types::t_orientation l_orientation = (emp_types::t_orientation)(l_truncated_piece & 0x3);
                      unsigned int l_piece_id = 1 + (l_truncated_piece >> 2);
                      simplex_variable * l_variable = new simplex_variable(m_simplex_variables.size(), l_x, l_y, l_piece_id, l_orientation);
                      m_simplex_variables.push_back(l_variable);
                      l_position_variables[get_position_index(l_x, l_y)].push_back(l_variable);
                      l_piece_id_variables[l_piece_id - 1].push_back(l_variable);
                  }
              }
          }
      }

      // Compute equation number
      // Width * Height because one equation per variable to state that each piece can only have one position/orientation
      // Width * Height because one position can only have piece/orientation
      uint64_t l_nb_equation = 0;
      for(unsigned int l_index = 0;
          l_index < m_info.get_height() * m_info.get_width();
	      ++l_index
	     )
      {
          // If position is free then add position equation
          if(l_position_variables[l_index].size())
          {
              ++l_nb_equation;
          }
          // if piece is not used then add piece equations
          if(l_piece_id_variables[l_index].size())
          {
              ++l_nb_equation;
          }
      }

      for(unsigned int l_y = 0;
	      l_y < m_info.get_height();
	      ++l_y
	     )
      {
          for(unsigned int l_x = 0;
	          l_x < m_info.get_width();
	          ++l_x
	         )
          {
              //	    std::cout << "Compute equation numbers at (" << l_x << "," << l_y << ")" << std::endl;
              if(l_x < m_info.get_width() - 1)
              {
                  l_nb_equation += get_nb_equations(l_position_variables[get_position_index(l_x, l_y)]
                                                   ,l_position_variables[get_position_index(l_x + 1, l_y)]
                                                   );
              }
              if(l_y < m_info.get_height() - 1)
              {
                  l_nb_equation += get_nb_equations(l_position_variables[get_position_index(l_x, l_y)]
                                                   ,l_position_variables[get_position_index(l_x, l_y + 1)]
                                                   );
              }
          }
      }
      std::cout << m_simplex_variables.size() << std::endl;
      std::cout << l_nb_equation << std::endl;

      // Create simplex representing puzzle
      m_simplex = new simplex_t(m_simplex_variables.size(), l_nb_equation, 0, 0);

      for(unsigned int l_index = 0;
	      l_index < m_simplex_variables.size();
	      ++l_index
	     )
      {
          m_simplex->set_Z_coef(l_index,simplex_t::t_coef_type(1));
      }

      unsigned int l_equation_index = 0;

      // Create position equations
      for(unsigned int l_index = 0;
          l_index < m_info.get_height() * m_info.get_width();
          ++l_index
         )
      {
          if(l_position_variables[l_index].size())
          {
              for(auto l_iter: l_position_variables[l_index])
              {
                  m_simplex->set_A_coef(l_equation_index, l_iter->get_id(), simplex_t::t_coef_type(1));
              }
              m_simplex->set_B_coef(l_equation_index, simplex_t::t_coef_type(1));
              m_simplex->define_equation_type(l_equation_index, simplex::t_equation_type::INEQUATION_LT);
              ++l_equation_index;
          }
      }

      // Create piece equations
      for(unsigned int l_index = 0;
	      l_index < m_info.get_height() * m_info.get_width();
	      ++l_index
	     )
      {
          if(l_piece_id_variables[l_index].size())
          {
              for(auto l_iter:l_piece_id_variables[l_index])
              {
                  m_simplex->set_A_coef(l_equation_index, l_iter->get_id(), simplex_t::t_coef_type(1));
              }
              m_simplex->set_B_coef(l_equation_index, simplex_t::t_coef_type(1));
              m_simplex->define_equation_type(l_equation_index, simplex::t_equation_type::INEQUATION_LT);
              ++l_equation_index;
          }
      }

      delete[] l_piece_id_variables;

      for(unsigned int l_y = 0;
	      l_y < m_info.get_height();
	      ++l_y
	     )
      {
          for(unsigned int l_x = 0;
	          l_x < m_info.get_width();
	          ++l_x
	         )
          {
              if(l_x < m_info.get_width() - 1)
              {
                  add_equations(l_position_variables[get_position_index(l_x, l_y)]
                               ,l_position_variables[get_position_index(l_x + 1, l_y)]
                               ,l_equation_index
                               ,true
                               );
              }
              if(l_y < m_info.get_height() - 1)
              {
                  add_equations(l_position_variables[get_position_index(l_x, l_y)]
                               ,l_position_variables[get_position_index(l_x, l_y + 1)]
                               ,l_equation_index
                               ,false
                               );
              }
          }
      }
      delete[] l_position_variables;

      assert(l_equation_index == l_nb_equation);
      determine_simplex_parameters(p_db);
  }

  //----------------------------------------------------------------------------
  unsigned int feature_simplex::get_position_index(const unsigned int & p_x
                                                  ,const unsigned int & p_y
						                          )const
  {
      assert(p_x < m_info.get_width());
      assert(p_y < m_info.get_height());
      return m_info.get_width() * p_y + p_x;
  }

  //----------------------------------------------------------------------------
  void feature_simplex::add_equations(const std::vector<simplex_variable*> & p_variables_pos1
                                     ,const std::vector<simplex_variable*> & p_variables_pos2
                                     ,unsigned int & p_equation_index
                                     ,bool p_horizontal
				                     )
  {
      for(auto l_iter_pos1: p_variables_pos1)
      {
          for(auto l_iter_pos2: p_variables_pos2)
          {
              if(l_iter_pos1->get_piece_id() != l_iter_pos2->get_piece_id())
              {
                  m_simplex->set_A_coef(p_equation_index, l_iter_pos1->get_id(), simplex_t::t_coef_type(1));
                  m_simplex->set_A_coef(p_equation_index, l_iter_pos2->get_id(), simplex_t::t_coef_type(1));

                  emp_types::t_orientation l_border1 = p_horizontal ? emp_types::t_orientation::EAST : emp_types::t_orientation::SOUTH;
                  emp_types::t_orientation l_border2 = p_horizontal ? emp_types::t_orientation::WEST : emp_types::t_orientation::NORTH;

                  emp_types::t_color_id l_color1 = m_db.get_piece(l_iter_pos1->get_piece_id()).get_color(l_border1,l_iter_pos1->get_orientation());
                  emp_types::t_color_id l_color2 = m_db.get_piece(l_iter_pos2->get_piece_id()).get_color(l_border2,l_iter_pos2->get_orientation());

                  m_simplex->set_B_coef(p_equation_index, simplex_t::t_coef_type(l_color1 == l_color2 ? 2 : 1));
                  m_simplex->define_equation_type(p_equation_index, simplex::t_equation_type::INEQUATION_LT);
                  ++p_equation_index;
              }
          }
      }
  }

  //----------------------------------------------------------------------------
  uint64_t feature_simplex::get_nb_equations(const std::vector<simplex_variable*> & p_variables_pos1
                                            ,const std::vector<simplex_variable*> & p_variables_pos2
                                            )
  {
      unsigned int l_result = 0;
      for(auto l_iter_pos1: p_variables_pos1)
      {
          for(auto l_iter_pos2: p_variables_pos2)
          {
              if(l_iter_pos1->get_piece_id() != l_iter_pos2->get_piece_id())
              {
                  ++l_result;
              }
          }
      }
      return l_result;
  }

  //----------------------------------------------------------------------------
  void feature_simplex::determine_simplex_parameters(const emp_piece_db & p_db)
  {
      // Compute simplex variable number
      // Corner and border pieces can have only one orientation per position
      // Center pieces can have 4 for orientation per position
      unsigned int l_nb_variable = p_db.get_nb_pieces(emp_types::t_kind::CORNER) * p_db.get_nb_pieces(emp_types::t_kind::CORNER) + p_db.get_nb_pieces(emp_types::t_kind::BORDER) * p_db.get_nb_pieces(emp_types::t_kind::BORDER) + 4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER) * p_db.get_nb_pieces(emp_types::t_kind::CENTER);

      // Compute number of equations
      unsigned int l_nb_equation = 0;

      l_nb_equation += 2 * m_info.get_height() * m_info.get_width();

      for(unsigned int l_y = 0;
	      l_y < m_info.get_height();
	      ++l_y
	     )
      {
          for(unsigned int l_x = 0;
	          l_x < m_info.get_width();
	          ++l_x
	         )
          {
              emp_types::kind l_kind = get_position_kind(l_x,l_y);
              if(l_x < m_info.get_width() - 1)
              {
                  emp_types::kind l_kind_bis = get_position_kind(l_x + 1,l_y);
                  l_nb_equation += compute_combination(l_kind, l_kind_bis);
              }
              if(l_y < m_info.get_height() - 1)
              {
                  emp_types::kind l_kind_bis = get_position_kind(l_x,l_y + 1);
                  l_nb_equation += compute_combination(l_kind, l_kind_bis);
              }
          }
      }

      std::cout << "== SImplex characteristics ==" << std::endl;
      std::cout << "Nb variables : " << l_nb_variable << std::endl;
      std::cout << "Nb equations : " << l_nb_equation << std::endl;
  }

  //----------------------------------------------------------------------------
  unsigned int feature_simplex::compute_combination(const emp_types::t_kind & p_kind1
                                                   ,const emp_types::t_kind & p_kind2
                                                   )
  {
      return get_nb_piece_possibility(p_kind1) * get_nb_piece_possibility(p_kind2, p_kind1 == p_kind2);
  }

  //----------------------------------------------------------------------------
  unsigned int feature_simplex::get_nb_piece_possibility(const emp_types::t_kind & p_kind
                                                        ,bool p_minor
                                                        )
  {
      assert(p_kind < emp_types::t_kind::UNDEFINED);
      unsigned int l_coef = emp_types::t_kind::CENTER == p_kind ? 4 : 1;
      return l_coef * (m_db.get_nb_pieces(p_kind) - p_minor);
  }

  //----------------------------------------------------------------------------
  emp_types::t_kind feature_simplex::get_position_kind(const unsigned int & p_x
                                                      ,const unsigned int & p_y
                                                      )
  {
      assert(p_x < m_info.get_width());
      assert(p_y < m_info.get_height());
      emp_types::t_kind l_type = emp_types::t_kind::CENTER;
      if(!p_x || !p_y || m_info.get_width() - 1 == p_x || p_y == m_info.get_height() - 1)
      {
          l_type = emp_types::t_kind::BORDER;
          if((!p_x && !p_y) ||
	         (!p_x && p_y == m_info.get_height() - 1) ||
	         (!p_y && p_x == m_info.get_width() - 1) ||
	         (p_y == m_info.get_height() - 1 && p_x == m_info.get_width() - 1)
	        )
          {
              l_type = emp_types::t_kind::CORNER;
          }
      }
      return l_type;
  }

  //----------------------------------------------------------------------------
  void feature_simplex::run(void)
  {
      simplex_t::t_coef_type l_max(0);
      bool l_infinite = false;
      emp_simplex_listener<simplex_t::t_coef_type> l_listener(*m_simplex,m_simplex_variables,std::cout);
      if(m_simplex->find_max(l_max,l_infinite,&l_listener))
      {
          std::cout << "Max = " << l_max << std::endl;
          for(unsigned int l_index = 0;
          l_index < m_simplex->get_total_nb_equation();
          ++l_index
             )
          {
              std::cout << "Base variable[" << l_index << "] is X" << m_simplex->get_base_variable(l_index) + 1 << std::endl;
          }
      }
      else if(l_infinite)
      {
          std::cout << "Inifinite Max" << std::endl;
      }
    else
      {
          std::cout << "No Max found !?" << std::endl;
      }
  }

  //----------------------------------------------------------------------------
  feature_simplex::~feature_simplex(void)
  {
      delete m_simplex;
      m_simplex = nullptr;
      for(auto l_iter: m_simplex_variables)
      {
          delete l_iter;
      }
  }
}
#endif // FEATURE_SIMPLEX_H
//EOF
