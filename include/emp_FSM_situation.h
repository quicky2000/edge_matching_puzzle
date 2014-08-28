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
#include "quicky_exception.h"
#include "quicky_bitfield.h"
#include <map>
#include <sstream>
#include <iomanip>

namespace edge_matching_puzzle
{
  class emp_FSM_situation: public FSM_base::FSM_situation<emp_FSM_context>
  {
  public:
    inline emp_FSM_situation(void);
    inline static void init(const emp_FSM_info & p_info);

    // Methods inherited from FSM_situation
    inline const std::string to_string(void)const;
    inline const std::string get_string_id(void)const;
    inline void to_string(std::string &)const;
    inline void get_string_id(std::string &)const;
    inline bool is_final(void)const;
    inline bool less(const FSM_interfaces::FSM_situation_if *p_situation)const;
    // End of methods inherited from FSM_situation

    // Dedicated method
    inline const emp_types::t_oriented_piece & get_piece(const unsigned int & p_x,
							 const unsigned int & p_y)const;
    inline void set_piece(const unsigned int & p_x,
			  const unsigned int & p_y,
			  const emp_types::t_oriented_piece & p_piece);
    inline bool contains_piece(const unsigned int & p_x,
                               const unsigned int & p_y)const;

    inline void set(const quicky_utils::quicky_bitfield & p_bitfield);

    inline const unsigned int get_level(void)const;
    inline void compute_string_id(std::string & p_id)const;
    inline void compute_bin_id(quicky_utils::quicky_bitfield & p_bitfield)const;
    static inline const unsigned int & get_nb_bits(void);
 private:
 
    typedef std::map<std::pair<unsigned int,unsigned int>, emp_types::t_oriented_piece> t_content;
    t_content m_content;

    static unsigned int m_piece_representation_width;
    static emp_FSM_info const * m_info;
    static unsigned int m_piece_nb_bits;
    static unsigned int m_situation_nb_bits;
  };

  //----------------------------------------------------------------------------
  const unsigned int emp_FSM_situation::get_level(void)const
  {
    return m_content.size();
  }

  //----------------------------------------------------------------------------
  const unsigned int & emp_FSM_situation::get_nb_bits(void)
    {
      return m_situation_nb_bits;
    }

  //----------------------------------------------------------------------------
  emp_FSM_situation::emp_FSM_situation(void)
    {
      assert(m_info);
    }

  //----------------------------------------------------------------------------
  void emp_FSM_situation::init(const emp_FSM_info & p_info)
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
    std::cout << "Pieces id coded on " << m_piece_nb_bits << " bits" << std::endl ;
    m_situation_nb_bits = p_info.get_width() * p_info.get_height() * ( m_piece_nb_bits + 2);
    std::cout << "Situation coded on " << m_situation_nb_bits << " bits" << std::endl ;
  }

  //----------------------------------------------------------------------------
  const std::string emp_FSM_situation::to_string(void)const
    {
      std::string l_unique_id;
      compute_string_id(l_unique_id);
      return l_unique_id;
    }
  //----------------------------------------------------------------------------
  const std::string emp_FSM_situation::get_string_id(void)const
    {
      std::string l_unique_id;
      compute_string_id(l_unique_id);
      return l_unique_id;
    }
  //----------------------------------------------------------------------------
  void emp_FSM_situation::to_string(std::string & p_string)const
  {
    compute_string_id(p_string);
  }
  //----------------------------------------------------------------------------
  void emp_FSM_situation::get_string_id(std::string & p_string_id)const
  {
    compute_string_id(p_string_id);
  }
  //----------------------------------------------------------------------------
  bool emp_FSM_situation::is_final(void)const
  {
    //return m_content.size() == 2 * m_info->get_width() + 2 * ( m_info->get_height() - 2);
    return m_content.size() == m_info->get_width() * m_info->get_height();
  }
  //----------------------------------------------------------------------------
  bool emp_FSM_situation::less(const FSM_interfaces::FSM_situation_if *p_situation)const
  {
    const emp_FSM_situation * l_situation = dynamic_cast<const emp_FSM_situation *>(p_situation);
    assert(l_situation);
    if(m_content.size() != l_situation->m_content.size()) return m_content.size() < l_situation->m_content.size();
    return m_content < l_situation->m_content;
  }
  //----------------------------------------------------------------------------
  const emp_types::t_oriented_piece & emp_FSM_situation::get_piece(const unsigned int & p_x,
                                                                   const unsigned int & p_y)const
    {
      t_content::const_iterator l_iter = m_content.find(std::pair<unsigned int,unsigned int>(p_x,p_y));
      if(m_content.end() == l_iter)
        {
          std::stringstream l_stream_x;
          l_stream_x << p_x;
          std::stringstream l_stream_y;
          l_stream_y << p_y;
          throw quicky_exception::quicky_logic_exception("No piece at position("+l_stream_x.str()+","+l_stream_y.str()+")",__LINE__,__FILE__);
        }
      return l_iter->second;
    }
  //----------------------------------------------------------------------------
  bool emp_FSM_situation::contains_piece(const unsigned int & p_x,
                                         const unsigned int & p_y)const
  {
    return m_content.end() != m_content.find(std::pair<unsigned int,unsigned int>(p_x,p_y));
  }

  //----------------------------------------------------------------------------
  void emp_FSM_situation::compute_string_id(std::string & p_id)const
  {
    p_id = std::string(m_info->get_width() * m_info->get_height() * m_piece_representation_width,'-');
    for(auto l_iter : m_content)
      {
        // Updating the unique identifier
        std::stringstream l_stream;
        l_stream << std::setw(m_piece_representation_width - 1) << l_iter.second.first;
        std::string l_piece_str(l_stream.str()+emp_types::orientation2short_string(l_iter.second.second));
        p_id.replace((l_iter.first.second * m_info->get_width() + l_iter.first.first) * m_piece_representation_width,m_piece_representation_width,l_piece_str);
      }
  }

  //----------------------------------------------------------------------------
  void emp_FSM_situation::set(const quicky_utils::quicky_bitfield & p_bitfield)
  {
    m_content.clear();
    this->get_context()->clear();
    for(unsigned int l_y = 0 ; l_y < m_info->get_height() ; ++l_y)
      {
        for(unsigned int l_x = 0 ; l_x < m_info->get_width() ; ++l_x)
          {
            unsigned int l_offset = (l_x + l_y * m_info->get_width()) * (m_piece_nb_bits + 2);
            unsigned int l_data;
            p_bitfield.get(l_data,m_piece_nb_bits + 2,l_offset);
            emp_types::t_piece_id l_id = l_data >> 2;
            if(l_id) set_piece(l_x,l_y,std::pair<emp_types::t_piece_id,emp_types::t_orientation>(l_id,((emp_types::t_orientation)(l_data & 0x3))));
          }
      }
  }


  //----------------------------------------------------------------------------
  void emp_FSM_situation::compute_bin_id(quicky_utils::quicky_bitfield & p_bitfield)const
  {
    p_bitfield.reset();
    for(auto l_iter : m_content)
      {
	unsigned int l_index = l_iter.first.second * m_info->get_width() + l_iter.first.first;
	unsigned int l_offset = l_index * ( m_piece_nb_bits + 2);
        unsigned int l_data = (l_iter.second.first << 2) + (unsigned int)l_iter.second.second;
	p_bitfield.set(l_data,m_piece_nb_bits+2,l_offset);
      }
  }

  //----------------------------------------------------------------------------
  void emp_FSM_situation::set_piece(const unsigned int & p_x,
                                    const unsigned int & p_y,
                                    const emp_types::t_oriented_piece & p_piece)
  {
    std::pair<unsigned int,unsigned int> l_position(p_x,p_y);
    if(m_content.end() != m_content.find(l_position))
      {
	std::stringstream l_stream_x;
	l_stream_x << p_x;
	std::stringstream l_stream_y;
	l_stream_y << p_y;
	throw quicky_exception::quicky_logic_exception("Already a piece at position("+l_stream_x.str()+","+l_stream_y.str()+")",__LINE__,__FILE__);
      }
    // Inserting value
    m_content.insert(t_content::value_type(l_position,p_piece));

    // Updating context
    this->get_context()->use_piece(p_piece.first);
  }
  
}
#endif // EMP_FSM_SITUATION_H
//EOF
