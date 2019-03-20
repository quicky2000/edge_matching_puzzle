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
#ifndef _EMP_SIMPLEX_LISTENER_H_
#define _EMP_SIMPLEX_LISTENER_H_

#include "simplex_array.h"
#include <iostream>
#include "simplex_variable.h"
#include "simplex_listener_target_if.h"
#include "emp_FSM_info.h"
#include "emp_FSM_situation.h"
#include <vector>

namespace edge_matching_puzzle
{
  template <typename COEF_TYPE>
  class emp_simplex_listener
  {
  public:
    inline emp_simplex_listener(const simplex::simplex_listener_target_if<COEF_TYPE> & p_simplex
                               ,const std::vector<simplex_variable *> & p_variables
                               ,const std::vector<simplex_variable *> *const p_position_variables
                               ,const emp_FSM_info & p_info
                               ,const std::string & p_initial_situation
                               ,std::ostream & p_ostream = std::cout
                               );
    inline void start_iteration(const unsigned int & p_nb_iteration);
    inline void new_input_var_event(const unsigned int & p_input_variable_index);
    inline void new_output_var_event(const unsigned int & p_input_variable_index);
    inline void new_Z0(COEF_TYPE p_z0);
    private:
      unsigned int m_nb_iteration;
      const simplex::simplex_listener_target_if<COEF_TYPE> & m_simplex;
      std::ostream & m_ostream;
      const std::vector<simplex_variable*> & m_variables;

      /**
       * Store all simplex variables related to a position index
       * Position index = width * Y + X)
       */
      const std::vector<simplex_variable*> * const m_position_variables;

      /**
       * Problem info
       */
      const emp_FSM_info & m_info;

      emp_FSM_situation m_situation;
      const std::string m_initial_situation;
  };

  //----------------------------------------------------------------------------
  template <typename COEF_TYPE>
  emp_simplex_listener<COEF_TYPE>::emp_simplex_listener(const simplex::simplex_listener_target_if<COEF_TYPE> & p_simplex
                                                       ,const std::vector<simplex_variable*> & p_variables
                                                       ,const std::vector<simplex_variable*> * const p_position_variables
                                                       ,const emp_FSM_info & p_info
                                                       ,const std::string & p_initial_situation
                                                       ,std::ostream & p_ostream
                                                       )
    : m_nb_iteration(0)
    , m_simplex(p_simplex)
    , m_ostream(p_ostream)
    , m_variables(p_variables)
    , m_position_variables(p_position_variables)
    , m_info(p_info)
    , m_initial_situation(p_initial_situation)
  {
      // Initialise situation with initial situation string
      m_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));
  }

  //----------------------------------------------------------------------------
  template <typename COEF_TYPE>
  void emp_simplex_listener<COEF_TYPE>::start_iteration(const unsigned int & p_nb_iteration)
  {
    m_nb_iteration = p_nb_iteration;
  }
 
  //----------------------------------------------------------------------------
  template <typename COEF_TYPE>
  void emp_simplex_listener<COEF_TYPE>::new_input_var_event(const unsigned int & p_input_variable_index)
  {
      m_ostream << "Iteration[" << m_nb_iteration << "] : New input variable selected : " << p_input_variable_index;
      if(p_input_variable_index < m_variables.size())
      {
          m_ostream << " => " << *m_variables[p_input_variable_index];
      }
      m_ostream << std::endl;
  }
  
  //----------------------------------------------------------------------------
  template <typename COEF_TYPE>
  void emp_simplex_listener<COEF_TYPE>::new_output_var_event(const unsigned int & p_output_variable_index)
  {
      //assert(p_output_variable_index < m_variables.size());
      m_ostream << "Iteration[" << m_nb_iteration << "] : New output variable selected : " << p_output_variable_index;
      if(p_output_variable_index < m_variables.size())
      {
          m_ostream << " => " << *m_variables[p_output_variable_index];
      }
      m_ostream << std::endl;
  }

  //----------------------------------------------------------------------------
  template<typename COEF_TYPE>
  void emp_simplex_listener<COEF_TYPE>::new_Z0(const COEF_TYPE p_z0)
  {
      m_ostream << "Iteration[" << m_nb_iteration << "] : New Z0 : " << p_z0 << std::endl;

      // Total of encountered variables values to stop iterations when we are
      // sure there are no mor emon null values
      COEF_TYPE l_total = (COEF_TYPE)0;

      if("" != m_initial_situation)
      {
          m_situation.set(m_initial_situation);
      }
      else
      {
          m_situation.reset();
      }

      std::vector<COEF_TYPE> l_values = m_simplex.get_variable_values();
      emp_types::t_oriented_piece l_oriented_piece;

      for(unsigned int l_y = 0;
          l_y < m_info.get_height() && l_total < p_z0;
          ++l_y
         )
      {
          for(unsigned int l_x = 0;
              l_x < m_info.get_width() && l_total < p_z0;
              ++l_x
              )
          {
              // Store number of values occurences for : 0, 1 or others
              unsigned int l_position_values[3] = {0,
                                                   0,
                                                   0
              };

              // Position index
              unsigned int l_index = m_info.get_width() * l_y + l_x;

              // Iterator on simplex variables related to current position
              for (auto l_iter:m_position_variables[l_index])
              {
                  const simplex_variable & l_variable = *l_iter;
                  COEF_TYPE l_value = l_values[l_variable.get_id()];
                  l_total += l_value;
                  if (l_value == (COEF_TYPE) 0)
                  {
                      ++l_position_values[0];
                      l_oriented_piece = l_variable.get_oriented_piece();
                  }
                  else if (l_value == (COEF_TYPE) 1)
                  {
                      ++l_position_values[1];
                  } else
                  {
                      ++l_position_values[2];
                  }
              }
              std::cout << "Position[" << l_x << ", " << l_y << "] : [" << l_position_values[0] << "," << l_position_values[1] << "," << l_position_values[2] << "] ";
              if (1 == l_position_values[1] && !l_position_values[2])
              {
                  std::cout << "FIXED";
                  m_situation.set_piece(l_x, l_y, l_oriented_piece);
              }
              else if (!l_position_values[1] && !l_position_values[2])
              {
                  std::cout << "NONE";
              }
              else
              {
                  std::cout << "UNDETERMINED";
              }
              std::cout << std::endl;
          }
      }
      std::cout << m_situation.to_string() << std::endl;
  }

}
#endif // _EMP_EMP_SIMPLEX_LISTENER_H_
//EOF
