/* -*- C++ -*- */
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
#ifndef _LIGHT_BORDER_PIECES_DB_
#define _LIGHT_BORDER_PIECES_DB_

#include <cinttypes>
#include <cassert>
#include <iomanip>
#include <iostream>

namespace edge_matching_puzzle
{
  /**
     Binary representation of pieces placed on eternity border including corners
     Pieces are represented in an oriebntation independant way considering that
     left and ride sides are the one on the left and the right side of the piece
     when point of view face center color
     NB : In eternity 2 center colors are different form borders to borders colors
     There are 22 colors and 0 represent no color which is usefull for corner
     pieces that have no center colors so 5 bits are sufficient to code one color id
     There a 3 color per pieces so in a 32 bits word we can store 2 pieces
     representation in order to occupy less memory despite being more expensive
     in term of computation because in CUDA global memory access has cost a lot
     of latency cycles whereas computation take few cycles
  **/
  class light_border_pieces_db
  {
    friend inline std::ostream & operator<<(std::ostream & p_stream,
					    const light_border_pieces_db & p_db
					    );
  public:
    inline void set_colors(unsigned int p_border_id,
			   uint32_t p_left_color,
			   uint32_t p_center_color,
			   uint32_t p_right_color
			   );

    inline uint32_t get_left(unsigned int p_border_id) const;
    inline uint32_t get_center(unsigned int p_border_id) const;
    inline uint32_t get_right(unsigned int p_border_id) const;
    inline void get_colors(unsigned int p_border_id,
			   uint32_t & p_left_color,
			   uint32_t & p_center_color,
			   uint32_t & p_right_color
			   ) const;
    inline void get_colors(unsigned int p_border_id,
			   uint32_t & p_left_color,
			   uint32_t & p_right_color
			   ) const;
  private:
    /**
       There are 60 pieces : 4 corners + 4 * 14 borders
       One 32 bits word contains 2 pieces representation
    **/
    uint32_t m_pieces[30];
  };

  //------------------------------------------------------------------------------
  void light_border_pieces_db::set_colors(unsigned int p_border_id,
					  uint32_t p_left_color,
					  uint32_t p_center_color,
					  uint32_t p_right_color
					  )
  {
    assert(p_border_id < 60);
    assert(p_left_color);
    assert(p_right_color);
    assert(p_border_id <= 4 || p_center_color);
    uint32_t l_color_mask = p_left_color;
    l_color_mask = (l_color_mask << 5) | p_center_color;
    l_color_mask = (l_color_mask << 5) | p_right_color;
    uint32_t l_reset_mask = (p_border_id & 0x1) ? 0xFFFF : 0xFFFF0000;
    if(p_border_id & 0x1)
      {
	l_color_mask = l_color_mask << 16;
      }
    m_pieces[p_border_id >> 1] &= l_reset_mask;
    m_pieces[p_border_id >> 1] |= l_color_mask;
  }

  //------------------------------------------------------------------------------
  void light_border_pieces_db::get_colors(unsigned int p_border_id,
					  uint32_t & p_left_color,
					  uint32_t & p_center_color,
					  uint32_t & p_right_color
					  ) const
  {
    uint32_t l_color_mask = m_pieces[p_border_id >> 1] >> (16 * (p_border_id & 0x1));
    p_right_color = l_color_mask & 0x1F;
    p_center_color = (l_color_mask >> 5) & 0x1F;
    p_left_color = (l_color_mask >> 10) & 0x1F;
  }

  //------------------------------------------------------------------------------
  void light_border_pieces_db::get_colors(unsigned int p_border_id,
					  uint32_t & p_left_color,
					  uint32_t & p_right_color
					  ) const
  {
    uint32_t l_color_mask = m_pieces[p_border_id >> 1] >> (16 * (p_border_id & 0x1));
    p_right_color = l_color_mask & 0x1F;
    p_left_color = (l_color_mask >> 10) & 0x1F;
  }

  //------------------------------------------------------------------------------
  uint32_t light_border_pieces_db::get_left(unsigned int p_border_id) const
  {
    assert(p_border_id < 60);
    return (m_pieces[p_border_id >> 1] >> (10 + 16 * (p_border_id & 0x1))) & 0x1F;
  }

  //------------------------------------------------------------------------------
  uint32_t light_border_pieces_db::get_center(unsigned int p_border_id) const
  {
    assert(p_border_id < 60);
    return (m_pieces[p_border_id >> 1] >> (5 + 16 * (p_border_id & 0x1))) & 0x1F;
  }

  //------------------------------------------------------------------------------
  uint32_t light_border_pieces_db::get_right(unsigned int p_border_id) const
  {
    assert(p_border_id < 60);
    return (m_pieces[p_border_id >> 1] >> (16 * (p_border_id & 0x1))) & 0x1F;
  }

  //------------------------------------------------------------------------------
  std::ostream & operator<<(std::ostream & p_stream,
			    const light_border_pieces_db & p_db
			    )
  {
    for(unsigned int l_index = 0;
	l_index < 30;
	++l_index)
      {
	p_stream << "pieces[" << l_index << "] = 0x" << std::hex << p_db.m_pieces[l_index] << std::dec << std::endl;
      }
    return p_stream;
  }
}
#endif // _LIGHT_BORDER_PIECES_DB_
// EOF
