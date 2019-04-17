/*    This file is part of edge_matching_puzzle
      Copyright (C) 2019  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#include "feature_system_equation.h"

namespace edge_matching_puzzle
{
    //-------------------------------------------------------------------------
    feature_system_equation::feature_system_equation(const emp_piece_db & p_db
                                                    ,const emp_FSM_info & p_info
                                                    ,const std::string & p_initial_situation
                                                    ,emp_gui & p_gui
                                                    )
    : m_variable_generator(p_db, p_info, p_initial_situation, m_situation)
    , m_gui(p_gui)
    {
        m_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));

        auto l_nb_variables = (unsigned int)m_variable_generator.get_variables().size();

        std::cout << "Nb variables : " << l_nb_variables << std::endl;

        // Prepare maks that will be applied when selecting a piece
        m_pieces_and_masks.clear();
        assert(m_pieces_and_masks.empty());
        m_pieces_and_masks.resize(l_nb_variables,emp_types::bitfield(l_nb_variables, true));

        // Mask other variables with same position or same piece id
        for(auto l_iter: m_variable_generator.get_variables())
        {
            unsigned int l_position_index = p_info.get_position_index(l_iter->get_x(), l_iter->get_y());
            unsigned int l_id = l_iter->get_id();

            // Mask bits corresponding to other variables with same position
            for(auto l_iter_variable: m_variable_generator.get_position_variables(l_position_index))
            {
                m_pieces_and_masks[l_id].set(0, 1, l_iter_variable->get_id());
            }

            // Mask bits corresponding to other variables with same ppiece id
            for(auto l_iter_variable: m_variable_generator.get_piece_variables(l_iter->get_piece_id()))
            {
                m_pieces_and_masks[l_id].set(0, 1, l_iter_variable->get_id());
            }
        }

        // Use a local variable to give access to this member by waiting C++20
        // that allow "=, this" capture
        auto & l_pieces_and_masks = m_pieces_and_masks;
        auto & l_variable_generator = m_variable_generator;

        auto l_lamda = [=, & l_pieces_and_masks, & l_variable_generator](const simplex_variable & p_var1
                                                ,const simplex_variable & p_var2
                                                )
        {
            unsigned int l_id1 = p_var1.get_id();
            assert(l_id1 < l_variable_generator.get_variables().size());
            unsigned int l_id2 = p_var2.get_id();
            assert(l_id2 < l_variable_generator.get_variables().size());
            l_pieces_and_masks[l_id1].set(0, 1, l_id2);
            l_pieces_and_masks[l_id2].set(0, 1, l_id1);
        };

        // Mask variables due to incompatible borders
        m_variable_generator.treat_piece_relations(l_lamda);
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::run()
    {
        m_gui.display(m_situation);

        emp_types::bitfield l_availabe_variables(m_variable_generator.get_variables().size(), true);
        unsigned int l_bit_index = 0;
        while((l_bit_index = l_availabe_variables.ffs()))
        {
            --l_bit_index;
            simplex_variable & l_variable = *m_variable_generator.get_variables()[l_bit_index];
            l_availabe_variables.apply_and(l_availabe_variables, m_pieces_and_masks[l_bit_index]);
            m_situation.set_piece(l_variable.get_x(), l_variable.get_y(), l_variable.get_oriented_piece());
            m_gui.display(m_situation);
            m_gui.refresh();
        }
    }

}
// EOF