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

#ifndef EDGE_MATCHING_PUZZLE_SITUATION_STRING_FORMATTER_H
#define EDGE_MATCHING_PUZZLE_SITUATION_STRING_FORMATTER_H

#include "emp_FSM_info.h"
#include <string>

namespace edge_matching_puzzle
{
    /**
     * Help class to deal with string conversion of various situation
     * implementations
     * @tparam SITUATION_TYPE Implementation of situation
     */
    template<typename SITUATION_TYPE>
    class situation_string_formatter
    {
      public:

        /**
         * Convert situation to string representation
         * @param p_situation situation to convert
         * @return string representation of situation
         */
        static
        std::string to_string(const SITUATION_TYPE & p_situation);

        /**
         * Set situation content from string representation
         * @param p_situation situation to fill up
         * @param p_string String representation
         */
        static
        void set(SITUATION_TYPE & p_situation
                ,const std::string & p_string
                );

      private:
    };

    template<typename SITUATION_TYPE>
    std::string situation_string_formatter<SITUATION_TYPE>::to_string(const SITUATION_TYPE & p_situation)
    {
        const emp_FSM_info & l_info{p_situation.get_info()};
        std::string l_result(l_info.get_nb_pieces() * p_situation.get_piece_representation_width(),'-');
        for(unsigned int l_index = 0 ; l_index < l_info.get_width() * l_info.get_height() ; ++l_index)
        {
            unsigned int l_x = l_index % l_info.get_width();
            unsigned int l_y = l_index / l_info.get_width();
            if(p_situation.contains_piece(l_x, l_y))
            {
                const emp_types::t_oriented_piece & l_oriented_piece = p_situation.get_piece(l_x, l_y);
                // Updating the unique identifier
                std::stringstream l_stream;
                l_stream << std::setw(p_situation.get_piece_representation_width() - 1) << l_oriented_piece.first;
                std::string l_piece_str(l_stream.str() + emp_types::orientation2short_string(l_oriented_piece.second));
                l_result.replace(l_index * p_situation.get_piece_representation_width(), p_situation.get_piece_representation_width(),l_piece_str);
            }
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    template <typename SITUATION_TYPE>
    void
    situation_string_formatter<SITUATION_TYPE>::set(SITUATION_TYPE & p_situation,
                                                    const std::string & p_string
                                                   )
    {
        p_situation.reset();
        if(p_string.size() == p_situation.get_info().get_nb_pieces() * p_situation.get_piece_representation_width())
        {
            for(unsigned int l_y = 0 ; l_y < p_situation.get_info().get_height() ; ++l_y)
            {
                for(unsigned int l_x = 0 ; l_x < p_situation.get_info().get_width() ; ++l_x)
                {
                    unsigned int l_index = (l_x + l_y * p_situation.get_info().get_width()) * p_situation.get_piece_representation_width();
                    std::string l_piece_id_str = p_string.substr(l_index, p_situation.get_piece_representation_width() - 1);
                    if(l_piece_id_str != std::string(p_situation.get_piece_representation_width() - 1 ,'-'))
                    {
                        std::string l_piece_orientation_str = p_string.substr(l_index + p_situation.get_piece_representation_width() - 1,1);
                        emp_types::t_piece_id l_piece_id = strtol(l_piece_id_str.c_str(), nullptr, 10);
                        emp_types::t_orientation l_piece_orientation;
                        bool l_found = false;
                        for(auto l_orient_index = static_cast<unsigned int>(emp_types::t_orientation::NORTH);
                            !l_found && l_orient_index <= static_cast<unsigned int>(emp_types::t_orientation::WEST);
                            ++l_orient_index
                                )
                        {
                            if(l_piece_orientation_str[0] == emp_types::orientation2short_string((emp_types::t_orientation)l_orient_index))
                            {
                                l_piece_orientation = (emp_types::t_orientation)l_orient_index;
                                l_found = true;
                            }
                        }
                        if(l_found)
                        {
                            p_situation.set_piece(l_x,l_y,std::pair<emp_types::t_piece_id,emp_types::t_orientation>(l_piece_id, l_piece_orientation));
                        }
                        else
                        {
                            throw quicky_exception::quicky_logic_exception("Unkown short string orientation : \"" + l_piece_orientation_str + "\"", __LINE__, __FILE__);
                        }
                    }
                }
            }
        }
        else
        {
            std::stringstream l_real_size;
            l_real_size << p_string.size();
            std::stringstream l_theoric_size;
            l_theoric_size << p_situation.get_info().get_nb_pieces() * p_situation.get_piece_representation_width();
            throw quicky_exception::quicky_logic_exception("Real size (" + l_real_size.str() + ") doesn`t match theoric size(" + l_theoric_size.str() +")", __LINE__, __FILE__);
        }
    }
}
#endif //EDGE_MATCHING_PUZZLE_SITUATION_STRING_FORMATTER_H
// EOF