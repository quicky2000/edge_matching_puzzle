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
#ifndef EMP_TEXT_STRATEGY_GENERATOR_H
#define EMP_TEXT_STRATEGY_GENERATOR_H

#include "emp_strategy_generator.h"
#include <string>
#include <fstream>
#include <sstream>

namespace edge_matching_puzzle
{
  class emp_text_strategy_generator: public emp_strategy_generator
  {
  public:
    inline emp_text_strategy_generator(const unsigned int & p_width,
				       const unsigned int & p_height,
				       const std::string & p_file_name);
    inline ~emp_text_strategy_generator(void);
    inline void generate(void);
  private:
    std::ifstream m_text;
    std::string m_file_name;
  };

  //----------------------------------------------------------------------------
  emp_text_strategy_generator::emp_text_strategy_generator(const unsigned int & p_width,
							   const unsigned int & p_height,
							   const std::string & p_file_name):
  emp_strategy_generator("text_generator",p_width,p_height),
    m_file_name(p_file_name)
  {
    m_text.open(p_file_name.c_str());
    if(!m_text.is_open())
      {
	throw quicky_exception::quicky_runtime_exception("Unable to open file \""+p_file_name+"\"",__LINE__,__FILE__);
      }
  }

  //----------------------------------------------------------------------------
  emp_text_strategy_generator::~emp_text_strategy_generator(void)
  {
    if(m_text.is_open())
      {
	m_text.close();
      }
  }

  //-----------------------------------------------------------------------------
  void emp_text_strategy_generator::generate(void)
  {
    unsigned int l_index = 0;
    unsigned int l_line_number = 1;
    while(!m_text.eof() && l_index < get_width() * get_height())
      {
	std::string l_line;
	getline(m_text,l_line);
	if(l_line.size())
	  {
	    if('#' != l_line[0])
	      {
		size_t l_pos = l_line.find(',');
		if(std::string::npos == l_pos)
		  {
		    std::stringstream l_stream;
		    l_stream << l_line_number ;
		    throw quicky_exception::quicky_logic_exception("Missing ',' character in line " + l_stream.str() + " : \"" + l_line + "\"",__LINE__,__FILE__);
		  }
		unsigned int l_x = atoi(l_line.substr(0,l_pos).c_str());
		if(l_pos + 1 >= l_line.size())
		  {
		    std::stringstream l_stream;
		    l_stream << l_line_number ;
		    throw quicky_exception::quicky_logic_exception("Missing information after ',' character in line " + l_stream.str() + " : \"" + l_line + "\"",__LINE__,__FILE__);
		  }
		unsigned int l_y = atoi(l_line.substr(l_pos + 1).c_str());
		++l_index;
		add_coordinate(l_x,l_y);
	      }
	  }
	++l_line_number;
      }
    if(l_index != get_width() * get_height())
      {
	std::stringstream l_index_stream;
	l_index_stream << l_index;
	std::stringstream l_theoric_stream;
	l_theoric_stream << get_width() * get_height();
	throw quicky_exception::quicky_logic_exception("Incomplete strategy file \"" + m_file_name + "\" : " + l_index_stream.str() + " < " + l_theoric_stream.str(),__LINE__,__FILE__);
      }
  }
}

#endif // EMP_TEXT_STRATEGY_GENERATOR_H
//EOF
