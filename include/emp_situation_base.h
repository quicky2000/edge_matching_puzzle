/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2021  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_EMP_SITUATION_BASE_H
#define EDGE_MATCHING_PUZZLE_EMP_SITUATION_BASE_H

#include "emp_FSM_info.h"

namespace edge_matching_puzzle
{
    /**
     * Information common to all emp_situation implementations, mainly
     * size information
     */
    class emp_situation_base
    {
      public:

        inline static
        void init(const emp_FSM_info & p_info);

        inline static
        unsigned int get_piece_representation_width();

        inline static
        const emp_FSM_info & get_info();

      protected:

        inline static
        unsigned int get_nb_bits();

        inline static
        unsigned int get_piece_nb_bits();

      private:
        static
        unsigned int m_piece_representation_width;

        static
        emp_FSM_info const * m_info;

        static
        unsigned int m_piece_nb_bits;

        static
        unsigned int m_situation_nb_bits;

    };

    //----------------------------------------------------------------------------
    void emp_situation_base::init(const emp_FSM_info & p_info)
    {
        m_info = &p_info;
        std::stringstream l_stream;
        l_stream << p_info.get_width() * p_info.get_height();
        m_piece_representation_width = l_stream.str().size() + 1;

        unsigned int l_piece_code_number = p_info.get_width() * p_info.get_height();
        m_piece_nb_bits = 0;
        while(l_piece_code_number)
        {
            ++m_piece_nb_bits;
            l_piece_code_number = l_piece_code_number >> 1;
        }
        std::cout << "Pieces id coded on " << m_piece_nb_bits << " bits" << std::endl;
        m_situation_nb_bits = p_info.get_width() * p_info.get_height() * ( m_piece_nb_bits + 2);
        std::cout << "Situation coded on " << m_situation_nb_bits << " bits" << std::endl ;
    }

    //-------------------------------------------------------------------------
    const emp_FSM_info &
    emp_situation_base::get_info()
    {
        assert(m_info);
        return *m_info;
    }

    //----------------------------------------------------------------------------
    unsigned int
    emp_situation_base::get_nb_bits()
    {
        return m_situation_nb_bits;
    }

    //----------------------------------------------------------------------------
    unsigned int
    emp_situation_base::get_piece_representation_width()
    {
        return m_piece_representation_width;
    }

    //----------------------------------------------------------------------------
    unsigned int
    emp_situation_base::get_piece_nb_bits()
    {
        return m_piece_nb_bits;
    }

}
#endif //EDGE_MATCHING_PUZZLE_EMP_SITUATION_BASE_H
// EOF