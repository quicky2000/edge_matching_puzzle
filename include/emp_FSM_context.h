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
#ifndef EMP_FSM_CONTEXT_H
#define EMP_FSM_CONTEXT_H

#include "FSM_context.h"
#include "emp_FSM_transition.h"
#include "quicky_bitfield.h"
#include <set>

namespace edge_matching_puzzle
{
  class emp_FSM_context: public FSM_base::FSM_context<emp_FSM_transition>
  {
  public:
    inline emp_FSM_context(const unsigned int & p_size);
    inline emp_FSM_context(const emp_FSM_context & p_context);
    inline ~emp_FSM_context(void){};
    
    // Methods inherited from interface
    inline const std::string to_string(void)const;
    inline void to_string(std::string &)const;
    
    // Specific methods
    inline void use_piece(const emp_types::t_piece_id & p_id);
    inline bool is_used(const emp_types::t_piece_id & p_id)const;
    inline void clear(void);
  private:
    quicky_utils::quicky_bitfield m_used_pieces;
  };
  //----------------------------------------------------------------------------
  emp_FSM_context::emp_FSM_context(const unsigned int & p_size):
    FSM_base::FSM_context<emp_FSM_transition>(),
    m_used_pieces(p_size)
      {
      }

    //----------------------------------------------------------------------------
    emp_FSM_context::emp_FSM_context(const emp_FSM_context & p_context):
      FSM_base::FSM_context<emp_FSM_transition>(),
      m_used_pieces(p_context.m_used_pieces)
        {
        }

      //----------------------------------------------------------------------------
      void emp_FSM_context::clear(void)
      {
        m_used_pieces.reset();
      }

      //----------------------------------------------------------------------------
      const std::string emp_FSM_context::to_string(void)const
        {
          return "";
        }
      //----------------------------------------------------------------------------
      void emp_FSM_context::to_string(std::string & p_string)const
      {
        p_string = "";
      }

      //----------------------------------------------------------------------------
      void emp_FSM_context::use_piece(const emp_types::t_piece_id & p_id)
      {
        m_used_pieces.set(1,1,p_id - 1);
      }

      //----------------------------------------------------------------------------
      bool emp_FSM_context::is_used(const emp_types::t_piece_id & p_id)const
      {
        unsigned int l_result;
        m_used_pieces.get(l_result,1,p_id - 1);
        return l_result;
      }

}
#endif // EMP_FSM_CONTEXT_H
//EOF
