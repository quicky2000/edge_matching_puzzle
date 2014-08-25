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
#ifndef PRECOMPUTED_TRANSITION_INFO_H
#define PRECOMPUTED_TRANSITION_INFO_H

#include "precomputed_constraint.h"
#include "emp_constraint.h"
#include <vector>

namespace edge_matching_puzzle
{
  class precomputed_transition_info
  {
  public:
    inline precomputed_transition_info(const emp_types::t_kind & p_kind,
                                       const std::pair<unsigned int,unsigned int> & p_position,
                                       const std::set<emp_constraint> & p_constraints,
                                       const std::vector<precomputed_constraint> & p_precomputed_constraints);
    inline const emp_types::t_kind & get_kind(void)const;
    inline const std::pair<unsigned int,unsigned int> & get_position(void)const;
    inline const std::vector<precomputed_constraint> & get_precomputed_constraints(void)const;
    inline const std::set<emp_constraint> & get_constraints(void)const;

  private:
    const emp_types::t_kind m_kind;
    const std::pair<unsigned int,unsigned int> m_position;
    const std::set<emp_constraint> m_constraints;
    const std::vector<precomputed_constraint> m_precomputed_constraints;
  };

  //----------------------------------------------------------------------------
  precomputed_transition_info::precomputed_transition_info(const emp_types::t_kind & p_kind,
                                                           const std::pair<unsigned int,unsigned int> & p_position,
                                                           const std::set<emp_constraint> & p_constraints,
                                                           const std::vector<precomputed_constraint> & p_precomputed_constraints):
    m_kind(p_kind),
    m_position(p_position),
    m_constraints(p_constraints),
    m_precomputed_constraints(p_precomputed_constraints)
      {
      }
    //----------------------------------------------------------------------------
    const emp_types::t_kind & precomputed_transition_info::get_kind(void)const
      {
        return m_kind;
      }

    //----------------------------------------------------------------------------
    const std::pair<unsigned int,unsigned int> & precomputed_transition_info::get_position(void)const
      {
        return m_position;
      }

    //----------------------------------------------------------------------------
    const std::vector<precomputed_constraint> & precomputed_transition_info::get_precomputed_constraints(void)const
      {
        return m_precomputed_constraints;
      }

    //----------------------------------------------------------------------------
    const std::set<emp_constraint> & precomputed_transition_info::get_constraints(void)const
      {
        return m_constraints; 
      }

}
#endif // PRECOMPUTED_TRANSITION_INFO_H
//EOF
