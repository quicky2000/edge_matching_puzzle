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
#ifndef EMP_STREAM_STRATEGY_GENERATOR_H
#define EMP_STREAM_STRATEGY_GENERATOR_H

#include "emp_strategy_generator.h"
#include <fstream>

namespace edge_matching_puzzle
{
  class emp_stream_strategy_generator: public emp_strategy_generator
  {
  public:
    inline emp_stream_strategy_generator(const unsigned int & p_width,
					 const unsigned int & p_height,
					 std::ifstream & p_istream);
    inline void generate(void);
  private:
    std::ifstream & m_stream;
  };

  //----------------------------------------------------------------------------
  emp_stream_strategy_generator::emp_stream_strategy_generator(const unsigned int & p_width,
							       const unsigned int & p_height,
							       std::ifstream & p_istream):
    emp_strategy_generator("stream_generator",p_width,p_height),
    m_stream(p_istream)
    {
    }

    //-----------------------------------------------------------------------------
    void emp_stream_strategy_generator::generate(void)
    {
      for(unsigned int l_index = 0 ; l_index < get_width() * get_height() ; ++l_index)
	{
	  uint32_t l_x;
	  uint32_t l_y;
	  m_stream.read((char*)&l_x,sizeof(l_x));
	  m_stream.read((char*)&l_y,sizeof(l_y));
	  add_coordinate(l_x,l_y);
	}
      
    }
}

#endif // EMP_STREAM_STRATEGY_GENERATOR_H
//EOF
