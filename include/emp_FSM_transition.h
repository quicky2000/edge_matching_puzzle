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
#ifndef EMP_FSM_TRANSITION_H
#define EMP_FSM_TRANSITION_H

#include "FSM_transition_if.h"
#include "emp_types.h"
#include <sstream>

namespace edge_matching_puzzle
{
  class emp_FSM_transition: public FSM_interfaces::FSM_transition_if
  {
  public:
    inline emp_FSM_transition(const unsigned int & p_x,
                              const unsigned int & p_y,
                              const emp_types::t_oriented_piece & p_piece);
    inline const unsigned int & get_x(void)const;
    inline const unsigned int & get_y(void)const;
    inline const emp_types::t_oriented_piece & get_piece(void)const;
    // Methods inherited from FSM_transition_if
    inline const std::string to_string(void)const;
    inline void to_string(std::string &)const;
    // End of methods inherited from FSM_transition_if
  private:
    const unsigned int m_x;
    const unsigned int m_y;
    const emp_types::t_oriented_piece m_piece;
  };

  //----------------------------------------------------------------------------
  emp_FSM_transition::emp_FSM_transition(const unsigned int & p_x,
                                         const unsigned int & p_y,
                                         const emp_types::t_oriented_piece & p_piece):
    
    m_x(p_x),
    m_y(p_y),
    m_piece(p_piece)
      {
      }
    //----------------------------------------------------------------------------
    const std::string emp_FSM_transition::to_string(void)const
      {
        std::stringstream l_stream_x;
        l_stream_x << m_x;
        std::stringstream l_stream_y;
        l_stream_y << m_y;
        std::stringstream l_stream_piece_id;
        l_stream_piece_id << m_piece.first;
        return "("+l_stream_x.str()+","+l_stream_y.str()+")=("+l_stream_piece_id.str()+","+emp_types::orientation2string(m_piece.second)+")";
      }
    //----------------------------------------------------------------------------
    void emp_FSM_transition::to_string(std::string & p_string)const
    {
        std::stringstream l_stream_x;
        l_stream_x << m_x;
        std::stringstream l_stream_y;
        l_stream_y << m_y;
        std::stringstream l_stream_piece_id;
        l_stream_piece_id << m_piece.first;
        p_string = "("+l_stream_x.str()+","+l_stream_y.str()+")=("+l_stream_piece_id.str()+","+emp_types::orientation2string(m_piece.second)+")";
    }
    //----------------------------------------------------------------------------
    const unsigned int & emp_FSM_transition::get_x(void)const
      {
        return m_x;
      }

    //----------------------------------------------------------------------------
    const unsigned int & emp_FSM_transition::get_y(void)const
      {
        return m_y;
      }
    //----------------------------------------------------------------------------
    const emp_types::t_oriented_piece & emp_FSM_transition::get_piece(void)const
      {
        return m_piece;
      }

}
#endif // EMP_FSM_TRANSITION_H
//EOF
