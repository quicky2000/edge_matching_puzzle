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

#ifndef EMP_TYPES_H
#define EMP_TYPES_H

#include "quicky_bitfield.h"
#include "quicky_exception.h"
#include <string>
#include <array>

namespace edge_matching_puzzle
{
    class emp_types
    {
      public:
        typedef enum class orientation {NORTH=0,EAST,SOUTH,WEST} t_orientation;
        typedef enum class kind {CENTER=0,BORDER,CORNER,UNDEFINED} t_kind;
        typedef unsigned int t_piece_id;
        typedef unsigned int t_color_id;
        typedef uint32_t t_binary_piece;
        typedef std::pair<emp_types::t_piece_id,emp_types::t_orientation> t_oriented_piece;

        inline static
        const std::string & kind2string(const t_kind & p_kind);

        inline static
        const std::string & orientation2string(const t_orientation & p_orientation);

        inline static
        const char & orientation2short_string(const t_orientation & p_orientation);

        inline static
        t_orientation short_string2orientation(const char & p_char);

        /**
         * Return list of orientation values ,useful to iterate on it
         * @return list of orientation values
         */
        inline static
        std::array<t_orientation, static_cast<unsigned int>(t_orientation::WEST) + 1> get_orientations();

        /**
         * Return previous clockwise orientation related to argument orientation
         * @param p_orientation ref orientation
         * @return previous clockwise orientation related to argument orientation
         */
        inline static
        emp_types::t_orientation get_previous_orientation(emp_types::t_orientation p_orientation);

        /**
         * Return next clockwise orientation related to argument orientation
         * @param p_orientation ref orientation
         * @return next clockwise orientation related to argument orientation
         */
        inline static
        emp_types::t_orientation get_next_orientation(emp_types::t_orientation p_orientation);

        inline static
        emp_types::t_orientation get_opposite(emp_types::t_orientation p_orientation);

        typedef quicky_utils::quicky_bitfield<uint64_t> bitfield;

      private:

        static const std::string m_kind_strings[((uint32_t)t_kind::UNDEFINED) + 1];

        static const std::string m_orientation_strings[((uint32_t)t_orientation::WEST) + 1];

        static const char m_short_orientation_strings[((uint32_t)t_orientation::WEST) + 1];

    };

    //----------------------------------------------------------------------------
    const std::string & emp_types::kind2string(const t_kind & p_kind)
    {
        return m_kind_strings[(uint32_t) p_kind];
    }

    //----------------------------------------------------------------------------
    const std::string & emp_types::orientation2string(const t_orientation & p_orientation)
    {
        return m_orientation_strings[(uint32_t) p_orientation];
    }

    //----------------------------------------------------------------------------
    const char & emp_types::orientation2short_string(const t_orientation & p_orientation)
    {
        return m_short_orientation_strings[(uint32_t) p_orientation];
    }

    //----------------------------------------------------------------------------
    emp_types::t_orientation emp_types::short_string2orientation(const char & p_char)
    {
        switch(p_char)
        {
            case 'N': return emp_types::t_orientation::NORTH;
            case 'E': return emp_types::t_orientation::EAST;
            case 'S': return emp_types::t_orientation::SOUTH;
            case 'W': return emp_types::t_orientation::WEST;
            default:
                throw quicky_exception::quicky_logic_exception("Unkown short string orientation '" + std::string(1,p_char) +"'",__LINE__,__FILE__);
        }
    }

    //-------------------------------------------------------------------------
    std::array<emp_types::t_orientation, static_cast<unsigned int>(emp_types::t_orientation::WEST) + 1>
    emp_types::get_orientations()
    {
        return {emp_types::t_orientation::NORTH
               ,emp_types::t_orientation::EAST
               ,emp_types::t_orientation::SOUTH
               ,emp_types::t_orientation::WEST
               };
    }

    //-------------------------------------------------------------------------
    emp_types::t_orientation
    emp_types::get_previous_orientation(emp_types::t_orientation p_orientation)
    {
        return static_cast<emp_types::t_orientation>((static_cast<unsigned int>(p_orientation) + 3) % 4);
    }
    //-------------------------------------------------------------------------
    emp_types::t_orientation
    emp_types::get_next_orientation(emp_types::t_orientation p_orientation)
    {
        return static_cast<emp_types::t_orientation>((static_cast<unsigned int>(p_orientation) + 1) % 4);
    }

    //-------------------------------------------------------------------------
    emp_types::t_orientation
    emp_types::get_opposite(emp_types::t_orientation p_orientation)
    {
        return static_cast<emp_types::t_orientation>((static_cast<unsigned int>(p_orientation) + 2) % 4);
    }
}
#endif //EMP_TYPES_H
//EOF
