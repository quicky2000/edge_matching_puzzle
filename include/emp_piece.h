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
#ifndef EMP_PIECE_H
#define EMP_PIECE_H

#include "emp_types.h"
#include <inttypes.h>
#include <array>
#include <cassert>
#include <iostream>

namespace edge_matching_puzzle
{
  class emp_piece
  {
    friend std::ostream & operator<<(std::ostream & p_stream,const emp_piece & p_piece);
  public:
    inline emp_types::t_kind get_kind(void)const;
    inline emp_piece(const emp_types::t_piece_id & p_id,
                     const std::array<emp_types::t_color_id,((unsigned int)(emp_types::t_orientation::WEST))+1> & p_colours);
    inline const emp_types::t_color_id & get_color(const emp_types::t_orientation & p_border)const;
    inline const emp_types::t_color_id & get_color(const emp_types::t_orientation & p_border,
                                                   const emp_types::t_orientation & p_orientation)const;
    inline const emp_types::t_piece_id & get_id(void)const;    
#if GCC_VERSION > 40702
  protected:
#endif // GCC_VERSION > 40702
    inline const std::array<emp_types::t_color_id,4> & get_colors(void)const;
  private:
    const emp_types::t_piece_id m_id;
    const emp_types::t_kind m_kind;
    const std::array<emp_types::t_color_id,4> m_colours;
  };

  //----------------------------------------------------------------------------
  inline std::ostream & operator<<(std::ostream & p_stream,const edge_matching_puzzle::emp_piece & p_piece)
    {
      p_stream << "(id=" << p_piece.m_id << "," << emp_types::kind2string(p_piece.m_kind);
      for(auto l_color : p_piece.m_colours)
        {
          p_stream << "," << l_color ;
        }
      p_stream << ")";
      return p_stream;
    }

    //----------------------------------------------------------------------------
    emp_piece::emp_piece(const emp_types::t_piece_id & p_id,
                         const std::array<emp_types::t_color_id,((unsigned int)(emp_types::t_orientation::WEST))+1> & p_colours):
      m_id(p_id),
      m_kind((emp_types::t_kind)((0 == p_colours[(unsigned int)emp_types::t_orientation::NORTH] ) + 
                                 (0 == p_colours[(unsigned int)emp_types::t_orientation::EAST] ) + 
                                 (0 == p_colours[(unsigned int)emp_types::t_orientation::SOUTH] ) + 
                                 (0 == p_colours[(unsigned int)emp_types::t_orientation::WEST] )
                                 )
             ),
      m_colours(p_colours)
      {
      }

    //----------------------------------------------------------------------------
    emp_types::t_kind emp_piece::get_kind(void)const
      {
	return m_kind;
      }
    //--------------------------------------------------------------------------
    const emp_types::t_color_id & emp_piece::get_color(const emp_types::t_orientation & p_border)const
      {
        assert((unsigned int) p_border < m_colours.size());
        return m_colours[(unsigned int)p_border];
      }

    //--------------------------------------------------------------------------
    const emp_types::t_color_id & emp_piece::get_color(const emp_types::t_orientation & p_border,
                                                       const emp_types::t_orientation & p_orientation)const
      {
        return m_colours[((unsigned int)p_border + (unsigned int) p_orientation) % 4];
      }

    //--------------------------------------------------------------------------
    const emp_types::t_piece_id & emp_piece::get_id(void)const
      {
        return m_id;
      }

    //--------------------------------------------------------------------------
    const std::array<emp_types::t_color_id,4> & emp_piece::get_colors(void)const
      {
        return m_colours;
      }

}


#endif // EMP_PIECE_H
//EOF
