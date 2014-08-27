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
#include "quicky_bitfield.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace edge_matching_puzzle
{
  class emp_situation_binary_reader
  {
  public:
    inline emp_situation_binary_reader(const std::string & p_name,
                                       const unsigned int & p_width,
                                       const unsigned int & p_height);
    inline ~emp_situation_binary_reader(void);
    inline void read(const uint64_t & p_index,
                     emp_FSM_situation & p_situation,
                     uint64_t & p_number);
  private:
    const unsigned int m_version;
    std::ifstream m_file;
    quicky_utils::quicky_bitfield m_bitfield;
    const unsigned int m_record_size;
    std::streampos m_start;
    uint64_t m_situation_number;
  };

  //----------------------------------------------------------------------------
  emp_situation_binary_reader::emp_situation_binary_reader(const std::string & p_name,
                                                           const unsigned int & p_width,
                                                           const unsigned int & p_height):
    m_version(0),
    m_file(NULL),
    m_bitfield(emp_FSM_situation::get_nb_bits()),
    m_record_size(sizeof(uint64_t)+m_bitfield.size()),
    m_start(0),
    m_situation_number(0)
    {
      m_file.open(p_name.c_str(),std::ifstream::binary);
      if(!m_file) throw quicky_exception::quicky_runtime_exception("Unable to read file \""+p_name+"\"",__LINE__,__FILE__);
      unsigned int l_version;
      m_file.read((char*)&l_version,sizeof(l_version));
      std::cout << "Version : " << l_version << std::endl ;
      if(m_version != l_version) 
        {
          std::stringstream l_stream1;
          l_stream1 << m_version ;
          std::stringstream l_stream2;
          l_stream2 << l_version ;
          throw quicky_exception::quicky_logic_exception("Reader and file does not have the same version "+l_stream1.str()+" != "+l_stream2.str(),__LINE__,__FILE__);
        }

      unsigned int l_width,l_height;
      m_file.read((char*)&l_width,sizeof(l_width));
      m_file.read((char*)&l_height,sizeof(l_height));
      std::cout << "Width = " << l_width << std::endl ;
      std::cout << "Height = " << l_height << std::endl ;
      if(l_width != p_width || l_height != p_height)
        {
          std::stringstream l_stream ;
          l_stream << "(" << p_width << "*" << p_height << ") != (" << l_width << "*" << l_height << ")";
          throw quicky_exception::quicky_logic_exception("Reader and file does not have the puzzle dimensions : "+l_stream.str(),__LINE__,__FILE__);
        }
      m_start = m_file.tellg();
      m_file.seekg(-sizeof(uint64_t),m_file.end);
      std::streampos l_length = m_file.tellg() - m_start;

      if(l_length % m_record_size) throw quicky_exception::quicky_logic_exception("Number of recorded situation is incorrect",__LINE__,__FILE__);
      m_situation_number = l_length / m_record_size;

      uint64_t l_total_situation;
      m_file.read((char*)&l_total_situation,sizeof(l_total_situation));
      std::cout << "Number of recorded situations = " << m_situation_number << std::endl ;
      std::cout << "Total situations explored : " << l_total_situation << std::endl ;
    }

    //----------------------------------------------------------------------------
    void emp_situation_binary_reader::read(const uint64_t & p_index,
                                           emp_FSM_situation & p_situation,
                                           uint64_t & p_number)
    {
      if(p_index < m_situation_number)
        {
          m_file.seekg(m_start +((std::streampos)(p_index * m_record_size)));
          m_bitfield.read_from(m_file);
          p_situation.set(m_bitfield);
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
        if(m_file) m_file.close();
      }
 
  
}
#endif // EMP_SITUATION_BINARY_READER_H
//EOF
