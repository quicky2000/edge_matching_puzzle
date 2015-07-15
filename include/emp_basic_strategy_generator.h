/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2015  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EMP_BASIC_STRATEGY_GENERATOR_H
#define EMP_BASIC_STRATEGY_GENERATOR_H

#include "emp_strategy_generator.h"

namespace edge_matching_puzzle
{
  class emp_basic_strategy_generator: public emp_strategy_generator
  {
  public:
    inline emp_basic_strategy_generator(const unsigned int & p_width,
					const unsigned int & p_height);
    inline void generate(void);
  private:
  };

  //----------------------------------------------------------------------------
  emp_basic_strategy_generator::emp_basic_strategy_generator(const unsigned int & p_width,
							     const unsigned int & p_height):
    emp_strategy_generator("basic_generator",p_width,p_height)
    {
    }

    //----------------------------------------------------------------------------
    void emp_basic_strategy_generator::generate(void)
    {
      for(uint32_t l_y = 0 ; l_y < get_height() ; ++l_y)
        {
          for(uint32_t l_x = 0 ; l_x < get_width() ; ++l_x)
            {
	      add_coordinate(l_x,l_y);
	    }
	}
    }
}

#endif //EMP_BASIC_STRATEGY_GENERATOR_H
//EOF
