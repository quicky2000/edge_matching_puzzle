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
#ifndef EMP_SITUATION_BINARY_READER_H
#define EMP_SITUATION_BINARY_READER_H

#include "emp_FSM_situation.h"
#include "emp_FSM_info.h"
#include "quicky_bitfield.h"
#include "emp_basic_strategy_generator.h"
#include "emp_stream_strategy_generator.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace edge_matching_puzzle
{
  class emp_situation_binary_reader
  {
  public:
    inline emp_situation_binary_reader(const std::string & p_name,
                                       const emp_FSM_info & p_FSM_info);
    inline ~emp_situation_binary_reader(void);
    inline void read(const uint64_t & p_index,
                     emp_FSM_situation & p_situation,
                     uint64_t & p_number);
    inline const uint64_t & get_total_situations(void)const;
    inline const uint64_t & get_nb_recorded(void)const;
  private:
    const unsigned int m_reader_version;
    uint32_t m_file_version;
    std::ifstream m_file;
    typedef quicky_utils::quicky_bitfield<uint32_t> t_v0_bitfield;
    t_v0_bitfield *m_v0_bitfield;
    typedef quicky_utils::quicky_bitfield<uint64_t> t_v1_bitfield;
    t_v1_bitfield *m_v1_bitfield;
    unsigned int m_record_size;
    std::streampos m_start;
    const emp_FSM_info m_FSM_info;
    uint64_t m_situation_number;
    uint64_t m_total_situation;
    emp_strategy_generator *m_generator;
    uint32_t m_solution_dump;
    unsigned int m_input_field_size;
  };

  //----------------------------------------------------------------------------
  emp_situation_binary_reader::emp_situation_binary_reader(const std::string & p_name,
                                                           const emp_FSM_info & p_FSM_info):
    m_reader_version(1),
    m_file_version(0),
    m_v0_bitfield(nullptr),
    m_v1_bitfield(nullptr),
    m_record_size(0),
    m_start(0),
    m_FSM_info(p_FSM_info),
    m_situation_number(0),
    m_total_situation(0),
    m_generator(nullptr),
    m_solution_dump(0),
    m_input_field_size(0)
    {
      m_file.open(p_name.c_str(),std::ifstream::binary);
      if(!m_file) throw quicky_exception::quicky_runtime_exception("Unable to read file \""+p_name+"\"",__LINE__,__FILE__);
      m_file.read((char*)&m_file_version,sizeof(m_file_version));
      std::cout << "File Format Version : " << m_file_version << std::endl ;

      std::stringstream l_file_version;
      l_file_version << m_file_version ;
      if(m_reader_version < m_file_version) 
        {
          std::stringstream l_reader_version;
          l_reader_version << m_reader_version ;
          throw quicky_exception::quicky_logic_exception("File has a greater version than reader version "+l_file_version.str()+" != "+l_reader_version.str(),__LINE__,__FILE__);
        }
      uint32_t l_width,l_height;
      m_file.read((char*)&l_width,sizeof(l_width));
      m_file.read((char*)&l_height,sizeof(l_height));
      std::cout << "Width = " << l_width << std::endl ;
      std::cout << "Height = " << l_height << std::endl ;
      if(l_width != m_FSM_info.get_width() || l_height != m_FSM_info.get_height())
        {
          std::stringstream l_stream ;
          l_stream << "(" << m_FSM_info.get_width() << "*" << m_FSM_info.get_height() << ") != (" << l_width << "*" << l_height << ")";
          throw quicky_exception::quicky_logic_exception("Reader and file does not have the puzzle dimensions : "+l_stream.str(),__LINE__,__FILE__);
        }


      switch(m_file_version)
	{
	case 0:
	  m_generator = new emp_basic_strategy_generator(m_FSM_info.get_width(),m_FSM_info.get_height());
	  // In this case m_solution_dump is always 0 as it was introduced with v1 version of format
	  m_input_field_size = (2 + (m_solution_dump ? p_FSM_info.get_piece_id_size() : p_FSM_info.get_dumped_piece_id_size()));
	  m_v0_bitfield = new t_v0_bitfield(p_FSM_info.get_width() * p_FSM_info.get_height() * m_input_field_size);
	  m_v1_bitfield = new t_v1_bitfield(p_FSM_info.get_width() * p_FSM_info.get_height() * m_input_field_size);
	  m_record_size = sizeof(uint64_t) + m_v0_bitfield->size();
	  break;
	case 1:
          m_file.read((char*)&m_solution_dump,sizeof(m_solution_dump));
          std::cout << "Solution dump : " << (m_solution_dump ? "YES" : "NO") << std::endl ;
	  m_generator = new emp_stream_strategy_generator(m_FSM_info.get_width(),m_FSM_info.get_height(),m_file);
	  m_input_field_size = (2 + (m_solution_dump ? p_FSM_info.get_piece_id_size() : p_FSM_info.get_dumped_piece_id_size()));
	  m_v1_bitfield = new t_v1_bitfield(p_FSM_info.get_width() * p_FSM_info.get_height() * m_input_field_size);
	  m_record_size = sizeof(uint64_t) + m_v1_bitfield->size();
	  break;
	default:
	  throw quicky_exception::quicky_logic_exception("Generator creation is not supported for file version "+l_file_version.str(),__LINE__,__FILE__);
	}
      m_generator->generate();

      m_start = m_file.tellg();
      m_file.seekg(-sizeof(uint64_t),m_file.end);
      std::streampos l_lenght = m_file.tellg() - m_start;

      m_situation_number = l_lenght / m_record_size;
      std::cout << "Number of recorded situations = " << m_situation_number << std::endl ;

      if(l_lenght % m_record_size)
        {
          std::cout << "File is truncated" << std::endl ;
        }
      else
        {
          m_file.read((char*)&m_total_situation,sizeof(m_total_situation));
          std::cout << "Total situations explored : " << m_total_situation << std::endl ;
        }
    }

    //----------------------------------------------------------------------------
    const uint64_t & emp_situation_binary_reader::get_total_situations(void)const
      {
        return m_total_situation;
      }

    //----------------------------------------------------------------------------
    const uint64_t & emp_situation_binary_reader::get_nb_recorded(void)const
      {
        return m_situation_number;
      }

    //----------------------------------------------------------------------------
    void emp_situation_binary_reader::read(const uint64_t & p_index,
                                           emp_FSM_situation & p_situation,
                                           uint64_t & p_number)
    {
      if(p_index < m_situation_number)
        {
          m_file.seekg(m_start +((std::streampos)(p_index * m_record_size)));
	  switch(m_file_version)
	    {
	    case 0:
	      m_v0_bitfield->read_from(m_file);
	      // Convert from old bitfield format (uint32_t) to new bitfield format ( uint64_t)
              for(unsigned int l_index = 0 ; l_index < m_FSM_info.get_width() * m_FSM_info.get_height() ; ++l_index)
              {
		unsigned int l_data=0;
		m_v0_bitfield->get(l_data,m_input_field_size,l_index * m_input_field_size);
		m_v1_bitfield->set(l_data,m_input_field_size,l_index * m_input_field_size);
	      }
	      p_situation.set(*m_v1_bitfield);
	      break;
	    case 1:
              {
		m_v1_bitfield->read_from(m_file);
                unsigned int l_output_field_size = m_FSM_info.get_dumped_piece_id_size() + 2;
                t_v1_bitfield l_sorted_bitfield(l_output_field_size * m_FSM_info.get_width() * m_FSM_info.get_height());
                for(unsigned int l_index = 0 ; l_index < m_FSM_info.get_width() * m_FSM_info.get_height() ; ++l_index)
                  {
                    std::pair<unsigned int,unsigned int> l_current_position = m_generator->get_position(l_index);
                    unsigned int l_new_index = l_current_position.second * m_FSM_info.get_width() + l_current_position.first;
                    unsigned int l_data=0;
                    m_v1_bitfield->get(l_data,m_input_field_size,l_index * m_input_field_size);
                    
                    unsigned int l_new_data = (((l_data >> 2 ) + (m_solution_dump ? 1 :0)) << 2)+ (l_data & 0x3);
                    l_sorted_bitfield.set(l_new_data,l_output_field_size,l_new_index * l_output_field_size);
                  }
                p_situation.set(l_sorted_bitfield);
              }
	      break;
	    default:
              {
                std::stringstream l_stream;
                l_stream << m_reader_version;
                throw quicky_exception::quicky_logic_exception("Generator creation is not supported for file version " + l_stream.str(),__LINE__,__FILE__);
              }
	    }

          m_file.read((char*)&p_number,sizeof(p_number));
        }
      else
        {
          std::stringstream l_stream1;
          l_stream1 << p_index ;
          std::stringstream l_stream2;
          l_stream2 << m_situation_number;
          throw quicky_exception::quicky_logic_exception("Requested index ("+l_stream1.str()+") is greater than number of record "+l_stream2.str(),__LINE__,__FILE__);
        }
    }

    //----------------------------------------------------------------------------
    emp_situation_binary_reader::~emp_situation_binary_reader(void)
      {
	delete m_generator;
        delete m_v1_bitfield;
        delete m_v0_bitfield;
        if(m_file) m_file.close();
      }
 
  
}
#endif // EMP_SITUATION_BINARY_READER_H
//EOF
