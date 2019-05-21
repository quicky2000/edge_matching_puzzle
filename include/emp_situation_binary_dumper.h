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
#include "emp_strategy_generator.h"
#include "emp_basic_strategy_generator.h"
#include "quicky_bitfield.h"
#include <fstream>
#include <iostream>
#include <memory>

namespace edge_matching_puzzle
{

  class emp_situation_binary_dumper
  {
  public:
    inline emp_situation_binary_dumper(const std::string & p_name,
                                       const emp_FSM_info & p_FSM_info,
                                       const std::unique_ptr<emp_strategy_generator> & p_generator = nullptr,
                                       bool p_solution_dump=false);
    inline ~emp_situation_binary_dumper();
    inline void dump(const emp_FSM_situation & p_situation);
    inline void dump(const uint64_t & p_total_number);
    template <typename T>
    inline void dump(const quicky_utils::quicky_bitfield<T> & p_bitfield,
                     const uint64_t & p_total_number);
  private:
    const uint32_t m_version;
    std::ofstream m_file;
    quicky_utils::quicky_bitfield<uint64_t> m_v1_bitfield;
    const uint32_t m_solution_dump;
  };

  //----------------------------------------------------------------------------
  emp_situation_binary_dumper::emp_situation_binary_dumper(const std::string & p_name,
                                                           const emp_FSM_info & p_FSM_info,
                                                           const std::unique_ptr<emp_strategy_generator> & p_generator,
                                                           bool p_solution_dump):
    m_version(1),
    m_v1_bitfield(p_FSM_info.get_width() * p_FSM_info.get_height() * (2 + (p_solution_dump ? p_FSM_info.get_piece_id_size() : p_FSM_info.get_dumped_piece_id_size()))),
    m_solution_dump((uint32_t)p_solution_dump)
    {
      m_file.open(p_name.c_str(),std::ofstream::binary);
      if(!m_file) throw quicky_exception::quicky_runtime_exception("Unable to create file \""+p_name+"\"",__LINE__,__FILE__);
      m_file.write((char*)&m_version,sizeof(m_version));
      m_file.write((char*)&p_FSM_info.get_width(),sizeof(p_FSM_info.get_width()));
      m_file.write((char*)&p_FSM_info.get_height(),sizeof(p_FSM_info.get_height()));
      m_file.write((char*)&m_solution_dump,sizeof(m_solution_dump));

      emp_basic_strategy_generator l_basic_generator(p_FSM_info.get_width(),p_FSM_info.get_height());
      l_basic_generator.generate();
      const emp_strategy_generator & l_generator = p_generator ? *p_generator : l_basic_generator;

      for(unsigned int l_index = 0 ; l_index < p_FSM_info.get_width() * p_FSM_info.get_height() ; ++l_index)
        {
          const std::pair<uint32_t,uint32_t> & l_position = l_generator.get_position(l_index);
          m_file.write((char*)&l_position.first,sizeof(l_position.first));
          m_file.write((char*)&l_position.second,sizeof(l_position.second));
        }
    }

    //----------------------------------------------------------------------------
    void emp_situation_binary_dumper::dump(const emp_FSM_situation & p_situation)
    {
      p_situation.compute_bin_id(m_v1_bitfield);
      m_v1_bitfield.dump_in(m_file);
    }

    //----------------------------------------------------------------------------
    template <typename T>
    void emp_situation_binary_dumper::dump(const quicky_utils::quicky_bitfield<T> & p_bitfield,const uint64_t & p_total_number)
    {
      assert(p_bitfield.bitsize() == m_v1_bitfield.bitsize());
      p_bitfield.dump_in(m_file);
      dump(p_total_number);
    }

    //----------------------------------------------------------------------------
    void emp_situation_binary_dumper::dump(const uint64_t & p_total_number)
    {
      m_file.write((char*)&p_total_number,sizeof(p_total_number));
    }

    //----------------------------------------------------------------------------
    emp_situation_binary_dumper::~emp_situation_binary_dumper()
      {
        if(m_file) m_file.close();
      }
 
}

#endif // EMP_SITUATION_BINARY_DUMPER_H
//EOF
