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
#include "ext_uint.h"
#include "ext_int.h"
#include "fract.h"
#include "simplex_solver.h"
#include "emp_simplex_listener.h"
#include "emp_gui.h"
#include <string>
#include <vector>
#include <simplex_solver_glpk.h>
#include <map>

namespace edge_matching_puzzle
{
  class feature_simplex:public feature_if
  {
  public:
    inline feature_simplex(const emp_piece_db & p_db
                          ,const emp_FSM_info & p_info
                          ,const std::string & p_initial_situation
                          ,emp_gui & p_gui
			              );

    // Virtual methods inherited from feature_if
    inline void run() override;
    // End of virtual methods inherited from feature_if
    inline ~feature_simplex() override;

  private:
      /**
       * Perform operations related to relationship between 2 adjacent postions
       * @tparam T type used to define algorithm containing operations to perform
       * @param p_variables_pos1 simplex variables related to position 1
       * @param p_variables_pos2 simplex variables related to position 2
       * @param p_horizontal true if positions are related horizontally
       * @param p_lambda algorithm containing operations to perform
       */
      template <typename T>
      void treat_piece_relation_equation(const std::vector<simplex_variable*> & p_variables_pos1
                                        ,const std::vector<simplex_variable*> & p_variables_pos2
                                        ,bool p_horizontal
                                        , T & p_lambda
                                        );

    /**
     * Indicate if position defined by parameters is corner/border/center
     * @param p_x column index
     * @param p_y row index
     * @return kind of position: corner/border/center
     */
    inline emp_types::t_kind get_position_kind(const unsigned int & p_x
                                              ,const unsigned int & p_y
					                          ) const;

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
     * Record hint information: piece + its location
     * @param p_piece_id Piece Id
     * @param p_x X location
     * @param p_y Y location
     */
    inline void record_hint(unsigned int p_piece_id
                           ,unsigned int p_x
                           ,unsigned int p_y
                           );

    /**
     * Treat situation string representation to extract hints ( pieces whose
     * location is know but not orientation) and remove them from string
     * representation
     * @param p_situation string representation of situation, hints will be
     * extracted and removed
     */
    inline void extract_hints(std::string & p_situation);

    const emp_piece_db & m_db;
    const emp_FSM_info & m_info;
    emp_FSM_situation m_situation;
    emp_types::bitfield m_available_corners;
    emp_types::bitfield m_available_borders;
    emp_types::bitfield m_available_centers;
    emp_types::bitfield * const m_available_pieces[3];
    std::vector<simplex_variable*> m_simplex_variables;

    /**
     * Store all simplex variables related to a position index
     * Position index = width * Y + X)
     */
    std::vector<simplex_variable*> * m_position_variables;

    typedef simplex::simplex_solver<quicky_utils::fract<quicky_utils::safe_int32_t>> simplex_t;
    //typedef simplex::simplex_solver<quicky_utils::fract<quicky_utils::ext_int<int32_t>>> simplex_t;
    //typedef simplex::simplex_solver_glpk simplex_t;
    simplex_t * m_simplex;
    emp_gui & m_gui;

    /**
     * String representation of initail situation
     */
    std::string m_initial_situation;

    /**
     * Positions where a piece is placed at the beginning: hint
     */
     std::map<std::pair<unsigned int, unsigned int>, unsigned int> m_position_hint;

    /**
     * Pieces whose position is known at the beginning: hint
     */
     std::map<unsigned int, std::pair<unsigned int, unsigned int>> m_piece_hint;
  };
 
  //----------------------------------------------------------------------------
  feature_simplex::feature_simplex(const emp_piece_db & p_db
                                  ,const emp_FSM_info & p_info
                                  ,const std::string & p_initial_situation
				                  ,emp_gui & p_gui
			                      )
	: m_db(p_db)
	, m_info(p_info)
	, m_available_corners(4 * p_db.get_nb_pieces(emp_types::t_kind::CORNER),true)
	, m_available_borders(4 * p_db.get_nb_pieces(emp_types::t_kind::BORDER),true)
	, m_available_centers(4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER),true)
	, m_available_pieces{&m_available_centers, &m_available_borders, &m_available_corners}
    , m_position_variables(new std::vector<simplex_variable*>[p_info.get_width() * p_info.get_height()])
	, m_simplex(nullptr)
	, m_gui(p_gui)
	, m_initial_situation(p_initial_situation)
  {
      // Initialise situation with initial situation string
      m_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));
      if(!m_initial_situation.empty())
      {
          // Search if some pieces have a determined position but no orientation
          extract_hints(m_initial_situation);
          m_situation.set(m_initial_situation);
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

      // Mark hint pieces as unavailable
      for(auto l_iter: m_piece_hint)
      {
          m_available_pieces[(unsigned int)p_db.get_piece(l_iter.first).get_kind()]->set(0x0, 4, 4 * p_db.get_kind_index(l_iter.first));
      }

      emp_types::bitfield l_matching_corners(4 * p_db.get_nb_pieces(emp_types::t_kind::CORNER));
      emp_types::bitfield l_matching_borders(4 * p_db.get_nb_pieces(emp_types::t_kind::BORDER));
      emp_types::bitfield l_matching_centers(4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER));
      emp_types::bitfield * const l_matching_pieces[3] = {&l_matching_centers,&l_matching_borders,&l_matching_corners};

      // Store all simplex variables related to a piece id
      // index 0 correspond to piece id 1)
      auto * l_piece_id_variables = new std::vector<simplex_variable*>[p_info.get_width() * p_info.get_height()];

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
              emp_types::t_kind l_type = get_position_kind(l_x,l_y);
              emp_types::bitfield l_possible_neighborhood(4 * p_db.get_nb_pieces(l_type), true);

              const auto l_position_hint_iter = m_position_hint.find({l_x, l_y});
              if(m_position_hint.end() != l_position_hint_iter)
              {
                  for(auto l_orientation = (unsigned int)emp_types::orientation::NORTH;
                      l_orientation <= (unsigned int)emp_types::orientation::WEST;
                      ++l_orientation
                      )
                  {
                      simplex_variable *l_variable = new simplex_variable((unsigned int) m_simplex_variables.size()
                                                                         ,l_x
                                                                         ,l_y
                                                                         ,l_position_hint_iter->second
                                                                         ,(emp_types::orientation) l_orientation
                                                                         );
                      m_simplex_variables.push_back(l_variable);
                  }
              }
              else if(!m_situation.contains_piece(l_x,l_y))
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
                      else
                      {
                          const auto l_neighbor_hint_iter = m_position_hint.find({l_x + 1, l_y});
                          if(m_position_hint.end() != l_neighbor_hint_iter)
                          {
                              l_possible_neighborhood.apply_and(l_possible_neighborhood
                                                               ,m_db.compute_possible_neighborhood(l_type
                                                                                                  ,l_neighbor_hint_iter->second
                                                                                                  , emp_types::t_orientation::EAST
                                                                                                  )
                                                               );
                          }
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
                      else
                      {
                          const auto l_neighbor_hint_iter = m_position_hint.find({l_x, l_y - 1});
                          if(m_position_hint.end() != l_neighbor_hint_iter)
                          {
                              l_possible_neighborhood.apply_and(l_possible_neighborhood
                                                               ,m_db.compute_possible_neighborhood(l_type
                                                                                                  ,l_neighbor_hint_iter->second
                                                                                                  ,emp_types::t_orientation::NORTH
                                                                                                  )
                                                               );
                          }
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
                      else
                      {
                          const auto l_neighbor_hint_iter = m_position_hint.find({l_x - 1, l_y});
                          if(m_position_hint.end() != l_neighbor_hint_iter)
                          {
                              l_possible_neighborhood.apply_and(l_possible_neighborhood,
                                                                m_db.compute_possible_neighborhood(l_type
                                                                                                  ,l_neighbor_hint_iter->second
                                                                                                  , emp_types::t_orientation::WEST
                                                                                                  )
                                                               );
                          }
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
                      else
                      {
                          const auto l_neighbor_hint_iter = m_position_hint.find({l_x, l_y + 1});
                          if(m_position_hint.end() != l_neighbor_hint_iter)
                          {
                              l_possible_neighborhood.apply_and(l_possible_neighborhood
                                                               ,m_db.compute_possible_neighborhood(l_type
                                                                                                  ,l_neighbor_hint_iter->second
                                                                                                  ,emp_types::t_orientation::SOUTH
                                                                                                  )
                                                               );
                          }
                      }

                  }
                  else
                  {
                      l_south_constraint = p_db.get_border_color_id();
                  }
                  l_constraint = (l_constraint << p_db.get_color_id_size()) | l_south_constraint;

                  // Compute pieces matching to constraint
                  l_matching_pieces[(unsigned int)l_type]->apply_and(p_db.get_pieces(l_constraint),*m_available_pieces[(unsigned int)l_type]);

                  // Filter with possible neigborhood related to hints
                  l_matching_pieces[(unsigned int)l_type]->apply_and(*l_matching_pieces[(unsigned int)l_type], l_possible_neighborhood);

                  // Iterating on matching pieces
                  emp_types::bitfield l_loop_pieces(*l_matching_pieces[(unsigned int)l_type]);
                  int l_ffs = 0;
                  while((l_ffs = l_loop_pieces.ffs()) != 0)
                  {
                      // We decrement because 0 mean no piece in other cases this
                      // is the index of oriented piece in piece list by kind
                      unsigned int l_piece_kind_id = (unsigned int)l_ffs - 1;
                      l_loop_pieces.set(0,1,l_piece_kind_id);
                      const emp_types::t_binary_piece l_piece = p_db.get_piece(l_type,l_piece_kind_id);
                      unsigned int l_truncated_piece = l_piece >> (4 * p_db.get_color_id_size());
                      auto l_orientation = (emp_types::t_orientation)(l_truncated_piece & 0x3);
                      unsigned int l_piece_id = 1 + (l_truncated_piece >> 2);
                      simplex_variable * l_variable = new simplex_variable((unsigned int)m_simplex_variables.size(), l_x, l_y, l_piece_id, l_orientation);
                      m_simplex_variables.push_back(l_variable);
                      m_position_variables[get_position_index(l_x, l_y)].push_back(l_variable);
                      l_piece_id_variables[l_piece_id - 1].push_back(l_variable);
                  }
              }
          }
      }

      // Compute equation number
      // Width * Height because one equation per variable to state that each piece can only have one position/orientation
      // Width * Height because one position can only have piece/orientation
      uint32_t l_nb_equation = 0;
      for(unsigned int l_index = 0;
          l_index < m_info.get_height() * m_info.get_width();
	      ++l_index
	     )
      {
          // If position is free then add position equation
          if(!m_position_variables[l_index].empty())
          {
              ++l_nb_equation;
          }
          // if piece is not used then add piece equations
          if(!l_piece_id_variables[l_index].empty())
          {
              ++l_nb_equation;
          }
      }

      auto l_count_equation_algo = [& l_nb_equation](const simplex_variable & p_var1
                                                    ,const simplex_variable & p_var2
                                                    )
              {
                ++l_nb_equation;
              };

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
                  treat_piece_relation_equation(m_position_variables[get_position_index(l_x, l_y)]
                                               ,m_position_variables[get_position_index(l_x + 1, l_y)]
                                               ,true
                                               ,l_count_equation_algo
                                               );
              }
              if(l_y < m_info.get_height() - 1)
              {
                  treat_piece_relation_equation(m_position_variables[get_position_index(l_x, l_y)]
                                               ,m_position_variables[get_position_index(l_x, l_y + 1)]
                                               ,false
                                               ,l_count_equation_algo
                                               );
              }
          }
      }
      std::cout << "== Simplex characteristics ==" << std::endl;
      std::cout << "Nb variables : " << m_simplex_variables.size() << std::endl;
      std::cout << "Nb equations : " << l_nb_equation << std::endl;

      // Create simplex representing puzzle
      m_simplex = new simplex_t((unsigned int)m_simplex_variables.size(), l_nb_equation, 0, 0);

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
          if(!m_position_variables[l_index].empty())
          {
              for(auto l_iter: m_position_variables[l_index])
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
          if(!l_piece_id_variables[l_index].empty())
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

      // Use a local variable to give access to this member by waiting C++20
      // that allow "=, this" capture
      auto l_simplex = this->m_simplex;
      auto l_create_equation_algo = [=, & l_equation_index](const simplex_variable & p_var1
                                                           ,const simplex_variable & p_var2
                                                           )
      {
          l_simplex->set_A_coef(l_equation_index, p_var1.get_id(), simplex_t::t_coef_type(1));
          l_simplex->set_A_coef(l_equation_index, p_var2.get_id(), simplex_t::t_coef_type(1));
          l_simplex->set_B_coef(l_equation_index, simplex_t::t_coef_type(1));
          l_simplex->define_equation_type(l_equation_index, simplex::t_equation_type::INEQUATION_LT);
          ++l_equation_index;
      };

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
                  treat_piece_relation_equation(m_position_variables[get_position_index(l_x, l_y)]
                                               ,m_position_variables[get_position_index(l_x + 1, l_y)]
                                               ,true
                                               ,l_create_equation_algo
                                               );
              }
              if(l_y < m_info.get_height() - 1)
              {
                  treat_piece_relation_equation(m_position_variables[get_position_index(l_x, l_y)]
                                               ,m_position_variables[get_position_index(l_x, l_y + 1)]
                                               ,false
                                               ,l_create_equation_algo
                                               );
              }
          }
      }

      assert(l_equation_index == l_nb_equation);
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
  template <typename T>
  void
  feature_simplex::treat_piece_relation_equation(const std::vector<simplex_variable *> & p_variables_pos1
                                                ,const std::vector<simplex_variable *> & p_variables_pos2
                                                ,bool p_horizontal
                                                ,T & p_lambda
                                                )
  {
      emp_types::t_orientation l_border1 = p_horizontal ? emp_types::t_orientation::EAST : emp_types::t_orientation::SOUTH;
      emp_types::t_orientation l_border2 = p_horizontal ? emp_types::t_orientation::WEST : emp_types::t_orientation::NORTH;

      for(auto l_iter_pos1: p_variables_pos1)
      {
          for(auto l_iter_pos2: p_variables_pos2)
          {
              if(l_iter_pos1->get_piece_id() != l_iter_pos2->get_piece_id())
              {
                  emp_types::t_color_id l_color1 = m_db.get_piece(l_iter_pos1->get_piece_id()).get_color(l_border1,l_iter_pos1->get_orientation());
                  emp_types::t_color_id l_color2 = m_db.get_piece(l_iter_pos2->get_piece_id()).get_color(l_border2,l_iter_pos2->get_orientation());

                  if(l_color1 != l_color2)
                  {
                      p_lambda(*l_iter_pos1, *l_iter_pos2);
                  }
              }
          }
      }
  }

  //----------------------------------------------------------------------------
  emp_types::t_kind feature_simplex::get_position_kind(const unsigned int & p_x
                                                      ,const unsigned int & p_y
                                                      ) const
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
  void feature_simplex::run()
  {
      simplex_t::t_coef_type l_max(0);
      bool l_infinite = false;
      emp_simplex_listener<simplex_t::t_coef_type> l_listener(*m_simplex
                                                             ,m_simplex_variables
                                                             ,m_position_variables
                                                             ,m_info
                                                             ,m_initial_situation
                                                             ,m_gui
                                                             ,std::cout
                                                             );
      if(m_simplex->find_max(l_max,l_infinite,&l_listener))
      {
          std::cout << "Max = " << l_max << std::endl;
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
  feature_simplex::~feature_simplex()
  {
      delete m_simplex;
      m_simplex = nullptr;
      for(auto l_iter: m_simplex_variables)
      {
          delete l_iter;
      }
      // Content of vectors has been destroyed just before
      delete[] m_position_variables;

  }

  //----------------------------------------------------------------------------
  void
  feature_simplex::record_hint(unsigned int p_piece_id
                              ,unsigned int p_x
                              ,unsigned int p_y
                              )
  {
      assert(m_piece_hint.end() == m_piece_hint.find(p_piece_id));
      m_piece_hint.insert({p_piece_id, {p_x, p_y}});
      assert(m_position_hint.end() == m_position_hint.find({p_x, p_y}));
      m_position_hint.insert({{p_x, p_y}, p_piece_id});
  }

  //----------------------------------------------------------------------------
  void
  feature_simplex::extract_hints(std::string & p_situation)
  {
      // Search if some pieces have a determined position but no orientation
      unsigned int l_piece_width = emp_FSM_situation::get_piece_representation_width();
      assert(!(p_situation.size() % l_piece_width));
      for(unsigned int l_index = 0; l_index < m_info.get_height() * m_info.get_width(); ++l_index)
      {
          char l_orientation = p_situation[l_piece_width * (l_index + 1) - 1];
          if(' ' == l_orientation)
          {
              unsigned int l_x = l_index % m_info.get_width();
              unsigned int l_y = l_index / m_info.get_width();
              auto l_piece_id = (emp_types::t_piece_id) std::stoul(p_situation.substr(l_index * l_piece_width, l_piece_width - 1));
              std::cout << "Hint " << l_piece_id << " @(" << l_x << "," << l_y << ")" << std::endl;
              p_situation.replace(l_piece_width * l_index, l_piece_width, std::string(l_piece_width, '-'));
              record_hint(l_piece_id, l_x, l_y);
          }
      }
  }

}
#endif // FEATURE_SIMPLEX_H
//EOF
