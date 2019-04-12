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
#include "emp_variable_generator.h"
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

    const emp_FSM_info & m_info;
    emp_FSM_situation m_situation;

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
     * Object responsible of creating variables and storing them
     */
    emp_variable_generator * m_variable_generator;
  };
 
  //----------------------------------------------------------------------------
  feature_simplex::feature_simplex(const emp_piece_db & p_db
                                  ,const emp_FSM_info & p_info
                                  ,const std::string & p_initial_situation
				                  ,emp_gui & p_gui
			                      )
	: m_info(p_info)
	, m_simplex(nullptr)
	, m_gui(p_gui)
	, m_initial_situation(p_initial_situation)
	, m_variable_generator(nullptr)
  {
      // Initialise situation with initial situation string
      m_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));

      m_variable_generator = new emp_variable_generator(p_db, p_info, p_initial_situation, m_situation);

      m_initial_situation = m_variable_generator->get_initial_situation_str();

      // Store all simplex variables related to a piece id
      // index 0 correspond to piece id 1)
      auto * l_piece_id_variables = new std::vector<simplex_variable*>[p_info.get_width() * p_info.get_height()];

      // Regroup variables per pieces
      for(auto l_iter: m_variable_generator->get_variables())
      {
        l_piece_id_variables[l_iter->get_piece_id() - 1].push_back(l_iter);
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
          if(!m_variable_generator->get_position_variables(l_index).empty())
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

      // Compute number of equations concerning 2 pieces
      m_variable_generator->treat_piece_relations(l_count_equation_algo);

      std::cout << "== Simplex characteristics ==" << std::endl;
      std::cout << "Nb variables : " << m_variable_generator->get_variables().size() << std::endl;
      std::cout << "Nb equations : " << l_nb_equation << std::endl;

      // Create simplex representing puzzle
      m_simplex = new simplex_t((unsigned int)m_variable_generator->get_variables().size(), l_nb_equation, 0, 0);

      for(unsigned int l_index = 0;
	      l_index < m_variable_generator->get_variables().size();
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
          if(!m_variable_generator->get_position_variables(l_index).empty())
          {
              for(auto l_iter: m_variable_generator->get_position_variables(l_index))
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

      // Create equations related to 2 pieces
      m_variable_generator->treat_piece_relations(l_create_equation_algo);

      assert(l_equation_index == l_nb_equation);
  }

  //----------------------------------------------------------------------------
  void feature_simplex::run()
  {
      simplex_t::t_coef_type l_max(0);
      bool l_infinite = false;
      emp_simplex_listener<simplex_t::t_coef_type> l_listener(*m_simplex
                                                             ,*m_variable_generator
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
      delete m_variable_generator;
      m_variable_generator = nullptr;
  }

}
#endif // FEATURE_SIMPLEX_H
//EOF
