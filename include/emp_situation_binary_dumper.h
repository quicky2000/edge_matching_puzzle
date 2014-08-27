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
#ifndef EMP_SITUATION_BINARY_DUMPER_H
#define EMP_SITUATION_BINARY_DUMPER_H

#include "emp_FSM_situation.h"
#include "quicky_bitfield.h"
#include <fstream>
#include <iostream>

namespace edge_matching_puzzle
{

  class emp_situation_binary_dumper
  {
  public:
    inline emp_situation_binary_dumper(const std::string & p_name,
                                       const unsigned int & p_width,
                                       const unsigned int & p_height);
    inline ~emp_situation_binary_dumper(void);
    inline void dump(const emp_FSM_situation & p_situation);
    inline void dump(const uint64_t & p_total_number);
  private:
    const unsigned int m_version;
    std::ofstream m_file;
    quicky_utils::quicky_bitfield m_bitfield;
  };

  //----------------------------------------------------------------------------
  emp_situation_binary_dumper::emp_situation_binary_dumper(const std::string & p_name,
                                                           const unsigned int & p_width,
                                                           const unsigned int & p_height):
    m_version(0),
    m_file(NULL),
    m_bitfield(emp_FSM_situation::get_nb_bits())
    {
      m_file.open(p_name.c_str(),std::ofstream::binary);
      if(!m_file) throw quicky_exception::quicky_runtime_exception("Unable to create file \""+p_name+"\"",__LINE__,__FILE__);
      m_file.write((char*)&m_version,sizeof(m_version));
      m_file.write((char*)&p_width,sizeof(p_width));
      m_file.write((char*)&p_height,sizeof(p_height));
    }

    //----------------------------------------------------------------------------
    void emp_situation_binary_dumper::dump(const emp_FSM_situation & p_situation)
    {
      p_situation.compute_bin_id(m_bitfield);
      m_bitfield.dump_in(m_file);
    }

    //----------------------------------------------------------------------------
    void emp_situation_binary_dumper::dump(const uint64_t & p_total_number)
    {
      m_file.write((char*)&p_total_number,sizeof(p_total_number));
    }

    //----------------------------------------------------------------------------
    emp_situation_binary_dumper::~emp_situation_binary_dumper(void)
      {
        if(m_file) m_file.close();
      }
 
}

#endif // EMP_SITUATION_BINARY_DUMPER_H
//EOF
