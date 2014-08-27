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

#include "emp_FSM_situation.h"

namespace edge_matching_puzzle
{
  emp_FSM_info const * emp_FSM_situation::m_info = NULL;
  unsigned int emp_FSM_situation::m_piece_representation_width = 0;
  unsigned int emp_FSM_situation::m_piece_nb_bits = 0;
  unsigned int emp_FSM_situation::m_situation_nb_bits = 0;
}
//EOF

