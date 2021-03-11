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
#ifndef EMP_PIECE_CORNER_H
#define EMP_PIECE_CORNER_H

#include "emp_piece.h"
#include "emp_constraint.h"
#include "quicky_exception.h"
#include "common.h"
#include <cinttypes>

namespace edge_matching_puzzle
{
    class emp_piece_corner: public emp_piece
    {

      public:

        [[maybe_unused]]
        inline
        emp_piece_corner(const emp_types::t_piece_id & p_id
                        ,const std::array<emp_types::t_color_id, static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours
                        );

        inline explicit
        emp_piece_corner(const emp_piece & p_piece);

        inline
        const std::pair<emp_types::t_color_id, emp_types::t_color_id> & get_border_colors()const;

        inline
        const std::pair<emp_types::t_orientation,emp_types::t_orientation> & get_colors_orientations()const;

        [[maybe_unused]]
        inline
        const emp_constraint & get_first_border()const;

        [[maybe_unused]]
        inline
        const emp_constraint & get_second_border()const;

        /**
         * Return piece orientation whose borders match orientations provided as paramter
         * @param p_border1 orientation of border
         * @param p_border2 orientation of border
         * @return piece orientation matching
         */
        inline
        emp_types::t_orientation compute_orientation(emp_types::t_orientation p_border1
                                                    ,emp_types::t_orientation p_border2
                                                    )const;

      private:

        inline static
        std::pair<emp_types::t_color_id, emp_types::t_color_id>
        compute_border_colors(const std::array<emp_types::t_color_id, static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours);

        inline static
        std::pair<emp_types::t_orientation, emp_types::t_orientation>
        compute_colors_orientations(const std::array<emp_types::t_color_id, static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours);

        inline static
        emp_constraint compute_constraint(const std::array<emp_types::t_color_id, static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours
                                         ,bool p_first
                                         );

        const std::pair<emp_types::t_color_id, emp_types::t_color_id> m_border_colors;

        const std::pair<emp_types::t_orientation,emp_types::t_orientation> m_colors_orientations;

        const emp_constraint m_first_border;

        const emp_constraint m_second_border;
    };

    //----------------------------------------------------------------------------
    [[maybe_unused]]
    emp_piece_corner::emp_piece_corner(const emp_types::t_piece_id & p_id
                                      ,const std::array<emp_types::t_color_id,static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours
                                      )
    :emp_piece(p_id, p_colours)
    ,m_border_colors(compute_border_colors(p_colours))
    ,m_colors_orientations(compute_colors_orientations(p_colours))
    ,m_first_border(compute_constraint(p_colours,true))
    ,m_second_border(compute_constraint(p_colours,false))
    {
        if(get_kind() != emp_types::t_kind::CORNER)
        {
            throw quicky_exception::quicky_logic_exception("Try to build a corner piece with a "+emp_types::kind2string(get_kind()),__LINE__,__FILE__);
        }
    }

    //----------------------------------------------------------------------------
    emp_piece_corner::emp_piece_corner(const emp_piece & p_piece)
    :emp_piece(p_piece.get_id(),p_piece.get_colors())
    ,m_border_colors(compute_border_colors(p_piece.get_colors()))
    ,m_colors_orientations(compute_colors_orientations(p_piece.get_colors()))
    ,m_first_border(compute_constraint(p_piece.get_colors(),true))
    ,m_second_border(compute_constraint(p_piece.get_colors(),false))
    {
    }

    //----------------------------------------------------------------------------
    [[maybe_unused]]
    const emp_constraint & emp_piece_corner::get_first_border()const
    {
        return m_first_border;
    }

    //----------------------------------------------------------------------------
    [[maybe_unused]]
    const emp_constraint & emp_piece_corner::get_second_border()const
    {
        return m_second_border;
    }

    //----------------------------------------------------------------------------
    emp_constraint emp_piece_corner::compute_constraint(const std::array<emp_types::t_color_id
                                                       ,static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours
                                                       ,bool p_first
                                                       )
    {
        bool l_second = false;
        for(unsigned int l_index = 0 ; l_index <= (unsigned int)emp_types::t_orientation::WEST; ++l_index)
	    {
            if(p_colours[l_index])
            {
                 if(p_first || l_second)
                 {
                     return {p_colours[l_index],(emp_types::t_orientation)l_index};
                 }
                 else
                 {
                     l_second = true;
                 }
            }
	    }
        throw quicky_exception::quicky_logic_exception("Unable to compute constraint",__LINE__,__FILE__);
    }

    //----------------------------------------------------------------------------
    std::pair<emp_types::t_orientation, emp_types::t_orientation>
    emp_piece_corner::compute_colors_orientations(const std::array<emp_types::t_color_id
                                                 ,static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours
                                                 )
    {
        std::pair<emp_types::t_orientation, emp_types::t_orientation> l_colors_orientations;
        bool l_first = true;
        for(unsigned int l_index = 0 ; l_index <= (unsigned int)emp_types::t_orientation::WEST; ++l_index)
	    {
            if(p_colours[l_index])
            {
                if(l_first)
                {
                    l_colors_orientations.first = (emp_types::t_orientation)l_index;
                    l_first = false;
                }
                else
                {
                    l_colors_orientations.second = (emp_types::t_orientation)l_index;
                    break;
                }
            }
	    }
        return l_colors_orientations;
    }

    //----------------------------------------------------------------------------
    std::pair<emp_types::t_color_id, emp_types::t_color_id>
    emp_piece_corner::compute_border_colors(const std::array<emp_types::t_color_id
                                           ,static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1> & p_colours
                                           )
    {
        std::pair<emp_types::t_color_id, emp_types::t_color_id> l_border_colors;
        for(unsigned int l_index = 0 ; l_index <= (unsigned int)emp_types::t_orientation::WEST; ++l_index)
        {
            if(p_colours[l_index])
            {
                if(!l_border_colors.first)
                {
                    l_border_colors.first = p_colours[l_index];
                }
                else
                {
                    l_border_colors.second = p_colours[l_index];
                    break;
                }
            }
        }
        return l_border_colors;
	}

	//----------------------------------------------------------------------------
    const std::pair<emp_types::t_color_id, emp_types::t_color_id> & emp_piece_corner::get_border_colors()const
	{
        return m_border_colors;
	}

	//----------------------------------------------------------------------------
	const std::pair<emp_types::t_orientation,emp_types::t_orientation> & emp_piece_corner::get_colors_orientations()const
	{
        return m_colors_orientations;
	}

	//-------------------------------------------------------------------------
    emp_types::t_orientation
    emp_piece_corner::compute_orientation(emp_types::t_orientation p_border1
                                         ,emp_types::t_orientation p_border2
                                         )const
    {
        for(auto l_iter: emp_types::get_orientations())
        {
            if(0 == get_color(p_border1, l_iter) && 0 == get_color(p_border2, l_iter))
            {
                return l_iter;
            }
        }
        throw quicky_exception::quicky_logic_exception("Unable to find a corner orientation for piece " + std::to_string(get_id()) + " matching provided borders (" + emp_types::orientation2short_string(p_border1) + "," + emp_types::orientation2short_string(p_border2)
                                                      ,__LINE__
                                                      ,__FILE__
                                                      );
    }

}
#endif // EMP_PIECE_CORNER_H
//EOF
