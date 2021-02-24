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
#ifndef EMP_FSM_SITUATION_H
#define EMP_FSM_SITUATION_H

#include "FSM_situation.h"
#include "emp_FSM_context.h"
#include "emp_FSM_info.h"
#include "emp_types.h"
#include "emp_situation_base.h"
#include "quicky_exception.h"
#include "situation_string_formatter.h"
#include <map>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace edge_matching_puzzle
{
    class emp_FSM_situation: public FSM_base::FSM_situation<emp_FSM_context>
                           , public emp_situation_base
    {
      public:

        inline
        emp_FSM_situation();

        // Methods inherited from FSM_situation
        inline
        const std::string to_string() const override;

        inline
        const std::string get_string_id() const override;

        inline
        void to_string(std::string &) const override;

        inline
        void get_string_id(std::string &) const override;

        inline
        bool is_final() const override;

        inline
        bool less(const FSM_interfaces::FSM_situation_if *p_situation) const override;
        // End of methods inherited from FSM_situation

        // Dedicated methods
        inline
        const emp_types::t_oriented_piece & get_piece(const unsigned int & p_x
                                                     ,const unsigned int & p_y
                                                     ) const;

        inline
        void set_piece(const unsigned int & p_x
                      ,const unsigned int & p_y
                      ,const emp_types::t_oriented_piece & p_piece
                      );

        inline
        bool contains_piece(const unsigned int & p_x
                           ,const unsigned int & p_y
                           ) const;

        inline
        void set(const emp_types::bitfield & p_bitfield);

        inline
        void set(const std::string & p_string);

        inline
        unsigned int get_level()const;

        inline
        void compute_string_id(std::string & p_id)const;

        inline
        void compute_bin_id(emp_types::bitfield & p_bitfield)const;

        inline
        ~emp_FSM_situation() override;

        inline
        emp_FSM_situation(const emp_FSM_situation & p_situation);

        inline
        void reset();

      private:

        emp_types::t_oriented_piece * m_content;

        unsigned int m_content_size;

        emp_types::bitfield m_used_positions;
    };

    //----------------------------------------------------------------------------
    unsigned int emp_FSM_situation::get_level() const
    {
        return m_content_size;
    }

    //----------------------------------------------------------------------------
    emp_FSM_situation::emp_FSM_situation()
    :m_content(new emp_types::t_oriented_piece[get_info().get_nb_pieces()])
    ,m_content_size(0)
    ,m_used_positions(get_info().get_nb_pieces())
    {
        std::transform(&m_content[0], &m_content[0] + get_info().get_nb_pieces(), m_content, [](emp_types::t_oriented_piece){return emp_types::t_oriented_piece {0, emp_types::t_orientation::NORTH }; });
    }

    //----------------------------------------------------------------------------
    emp_FSM_situation::emp_FSM_situation(const emp_FSM_situation & p_situation)
    :FSM_base::FSM_situation<emp_FSM_context>(p_situation)
    ,m_content(new emp_types::t_oriented_piece[get_info().get_nb_pieces()])
    ,m_content_size(p_situation.m_content_size)
    ,m_used_positions(p_situation.m_used_positions)
    {
        std::copy(&p_situation.m_content[0], &p_situation.m_content[0] + get_info().get_nb_pieces(), m_content);
    }

    //----------------------------------------------------------------------------
    emp_FSM_situation::~emp_FSM_situation()
    {
        delete[] m_content;
    }

    //----------------------------------------------------------------------------
    const std::string emp_FSM_situation::to_string() const
    {
        std::string l_unique_id;
        compute_string_id(l_unique_id);
        return l_unique_id;
    }

    //----------------------------------------------------------------------------
    const std::string emp_FSM_situation::get_string_id() const
    {
        std::string l_unique_id;
        compute_string_id(l_unique_id);
        return l_unique_id;
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::to_string(std::string & p_string) const
    {
        compute_string_id(p_string);
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::get_string_id(std::string & p_string_id) const
    {
        compute_string_id(p_string_id);
    }
    //----------------------------------------------------------------------------
    bool emp_FSM_situation::is_final() const
    {
        return m_content_size == get_info().get_nb_pieces();
    }

    //----------------------------------------------------------------------------
    bool emp_FSM_situation::less(const FSM_interfaces::FSM_situation_if *p_situation) const
    {
        const emp_FSM_situation * l_situation = dynamic_cast<const emp_FSM_situation *>(p_situation);
        assert(l_situation);
        if(m_content_size != l_situation->m_content_size) return m_content_size < l_situation->m_content_size;
        return memcmp(m_content, l_situation->m_content, get_info().get_nb_pieces() * sizeof(emp_types::t_oriented_piece)) < 0;
    }

    //----------------------------------------------------------------------------
    const emp_types::t_oriented_piece & emp_FSM_situation::get_piece(const unsigned int & p_x
                                                                    ,const unsigned int & p_y
                                                                    ) const
    {
        assert(get_info().get_nb_pieces() > get_info().get_position_index(p_x, p_y));
        return m_content[get_info().get_position_index(p_x, p_y)];
    }

    //----------------------------------------------------------------------------
    bool emp_FSM_situation::contains_piece(const unsigned int & p_x
                                          ,const unsigned int & p_y
                                          )const
    {
        unsigned int l_result;
        m_used_positions.get(l_result, 1, get_info().get_position_index(p_x, p_y));
        return l_result;
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::compute_string_id(std::string & p_id) const
    {
        p_id = situation_string_formatter<emp_FSM_situation>::to_string(*this);
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::set(const emp_types::bitfield & p_bitfield)
    {
        this->reset();
        for(unsigned int l_y = 0 ; l_y < get_info().get_height() ; ++l_y)
        {
            for(unsigned int l_x = 0 ; l_x < get_info().get_width() ; ++l_x)
            {
                unsigned int l_offset = (l_x + l_y * get_info().get_width()) * (get_piece_nb_bits() + 2);
                unsigned int l_data;
                p_bitfield.get(l_data,get_piece_nb_bits() + 2,l_offset);
                emp_types::t_piece_id l_id = l_data >> 2;
                if(l_id)
                {
                    set_piece(l_x, l_y, std::pair<emp_types::t_piece_id,emp_types::t_orientation>(l_id, ((emp_types::t_orientation)(l_data & 0x3))));
                }
            }
        }
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::set(const std::string & p_string)
    {
        situation_string_formatter<emp_FSM_situation>::set(*this, p_string);
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::reset()
    {
        m_used_positions.reset();
        m_content_size = 0;
        std::transform(&m_content[0], &m_content[0] + get_info().get_nb_pieces(), m_content, [](emp_types::t_oriented_piece){return emp_types::t_oriented_piece {0, emp_types::t_orientation::NORTH }; });
        this->get_context()->clear();
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::compute_bin_id(emp_types::bitfield & p_bitfield) const
    {
        p_bitfield.reset();
        for(unsigned int l_index = 0 ; l_index < get_info().get_nb_pieces() ; ++l_index)
        {
            if(contains_piece(l_index % get_info().get_width(), l_index / get_info().get_width()))
            {
                unsigned int l_offset = l_index * ( get_piece_nb_bits() + 2);
                unsigned int l_data = (m_content[l_index].first << 2) + (unsigned int)m_content[l_index].second;
                p_bitfield.set(l_data,get_piece_nb_bits() + 2,l_offset);
            }
        }
    }

    //----------------------------------------------------------------------------
    void emp_FSM_situation::set_piece(const unsigned int & p_x
                                     ,const unsigned int & p_y
                                     ,const emp_types::t_oriented_piece & p_piece
                                     )
    {
        if(contains_piece(p_x,p_y))
        {
            std::stringstream l_stream_x;
            l_stream_x << p_x;
            std::stringstream l_stream_y;
            l_stream_y << p_y;
            throw quicky_exception::quicky_logic_exception("Already a piece at position(" + l_stream_x.str() + "," + l_stream_y.str() + ")", __LINE__, __FILE__);
        }
        // Inserting value
        m_content[get_info().get_position_index(p_x, p_y)] = p_piece;
        m_used_positions.set(1,1,get_info().get_position_index(p_x, p_y));
        ++m_content_size;

        // Updating context
        this->get_context()->use_piece(p_piece.first);
    }

}
#endif // EMP_FSM_SITUATION_H
//EOF
