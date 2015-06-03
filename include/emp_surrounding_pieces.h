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
#ifndef EMP_SURROUNDING_PIECES_H
#define EMP_SURROUNDING_PIECES_H

#include "emp_types.h"
#include <set>

namespace edge_matching_puzzle
{
  /**
   * Class used to store one combination of pieces id that can be used to surround an other piece
   **/
  class emp_surrounding_pieces
  {
  public:
    inline emp_surrounding_pieces(void);
    inline void add_piece_id(const emp_types::t_piece_id & p_piece_id);
    inline bool contains_piece_id(const emp_types::t_piece_id & p_piece_id)const;
    inline bool operator <(const emp_surrounding_pieces & p_other)const;
  private:
    std::set<emp_types::t_piece_id> m_pieces;
  };

  //----------------------------------------------------------------------------
  emp_surrounding_pieces::emp_surrounding_pieces(void)
    {
    }

  //----------------------------------------------------------------------------
  void emp_surrounding_pieces::add_piece_id(const emp_types::t_piece_id & p_piece_id)
  {
    m_pieces.insert(p_piece_id);
  }

  //----------------------------------------------------------------------------
  bool emp_surrounding_pieces::contains_piece_id(const emp_types::t_piece_id & p_piece_id)const
  {
    return m_pieces.end() != m_pieces.find(p_piece_id);
  }

  //----------------------------------------------------------------------------
  bool emp_surrounding_pieces::operator <(const emp_surrounding_pieces & p_other)const
  {
    return m_pieces < p_other.m_pieces;
  }
}
#endif // EMP_SURROUNDING_PIECES_H
