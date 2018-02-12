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
#include <vector>
namespace simplex
{
  template <typename COEF_TYPE, typename ARRAY_TYPE>
    class simplex_solver_base;
}


namespace edge_matching_puzzle
{
  template <typename COEF_TYPE,typename ARRAY_TYPE=simplex::simplex_array<COEF_TYPE>>
  class emp_simplex_listener
  {
  public:
    inline emp_simplex_listener(const simplex::simplex_solver_base<COEF_TYPE,ARRAY_TYPE> & p_simplex,
                                const std::vector<simplex_variable*> & p_variables,
                                std::ostream & p_ostream = std::cout
                               );
    inline void start_iteration(const unsigned int & p_nb_iteration);
    inline void new_input_var_event(const unsigned int & p_input_variable_index);
    inline void new_output_var_event(const unsigned int & p_input_variable_index);
    inline void new_Z0(COEF_TYPE p_z0);
    private:
      unsigned int m_nb_iteration;
      const simplex::simplex_solver_base<COEF_TYPE,ARRAY_TYPE> & m_simplex;
      std::ostream & m_ostream;
      const std::vector<simplex_variable*> & m_variables;
  };

  //----------------------------------------------------------------------------
  template <typename COEF_TYPE,typename ARRAY_TYPE>
    emp_simplex_listener<COEF_TYPE,ARRAY_TYPE>::emp_simplex_listener(const simplex::simplex_solver_base<COEF_TYPE,ARRAY_TYPE> & p_simplex,
                                                                     const std::vector<simplex_variable*> & p_variables,
                                                                     std::ostream & p_ostream
                                                                    ):
            m_nb_iteration(0),
            m_simplex(p_simplex),
            m_ostream(p_ostream),
            m_variables(p_variables)
  {
  }

  //----------------------------------------------------------------------------
  template <typename COEF_TYPE,typename ARRAY_TYPE>
  void emp_simplex_listener<COEF_TYPE,ARRAY_TYPE>::start_iteration(const unsigned int & p_nb_iteration)
  {
    m_nb_iteration = p_nb_iteration;
  }
 
  //----------------------------------------------------------------------------
  template <typename COEF_TYPE,typename ARRAY_TYPE>
  void emp_simplex_listener<COEF_TYPE,ARRAY_TYPE>::new_input_var_event(const unsigned int & p_input_variable_index
					     )
  {
      m_ostream << "Iteration[" << m_nb_iteration << "] : New input variable selected : " << p_input_variable_index;
      if(p_input_variable_index < m_variables.size())
      {
          m_ostream << " => " << *m_variables[p_input_variable_index];
      }
      m_ostream << std::endl;
  }
  
  //----------------------------------------------------------------------------
  template <typename COEF_TYPE,typename ARRAY_TYPE>
  void emp_simplex_listener<COEF_TYPE,ARRAY_TYPE>::new_output_var_event(const unsigned int & p_output_variable_index
					     )
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
  template<typename COEF_TYPE,typename ARRAY_TYPE>
  void emp_simplex_listener<COEF_TYPE,ARRAY_TYPE>::new_Z0(const COEF_TYPE p_z0)
  {
    m_ostream << "Iteration[" << m_nb_iteration << "] : New Z0 : " << p_z0 << std::endl;
    //m_simplex.display_array(m_ostream);
  }

}
#endif // _EMP_EMP_SIMPLEX_LISTENER_H_
//EOF
