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

#ifndef EMP_SPIRAL_STRATEGY_GENERATOR_H
#define EMP_SPIRAL_STRATEGY_GENERATOR_H

#include "emp_strategy_generator.h"
#include "emp_FSM_situation.h"

namespace edge_matching_puzzle
{

  /**
     This strategy generator first generate border coordinates and then centers
     following a spiral
  **/
  class emp_spiral_strategy_generator: public emp_strategy_generator
  {
  public:
    inline emp_spiral_strategy_generator(const unsigned int & p_width,
                                         const unsigned int & p_height);

    inline void generate(void);

  private:
  };

  //----------------------------------------------------------------------------
  emp_spiral_strategy_generator::emp_spiral_strategy_generator(const unsigned int & p_width,
                                                               const unsigned int & p_height):
    emp_strategy_generator("spiral",p_width,p_height)
    {
    }

    //----------------------------------------------------------------------------
    void emp_spiral_strategy_generator::generate(void)
    {
      unsigned int l_x = 0;
      unsigned int l_y = 0;
      emp_FSM_situation l_situation;
      l_situation.set_context(*(new emp_FSM_context(get_width() * get_height())));
      for(unsigned int l_index = 0 ;
          l_index < get_width() * get_height();
          ++l_index)
        {
	  l_situation.set_piece(l_x,l_y,emp_types::t_oriented_piece(1,(edge_matching_puzzle::emp_types::t_orientation)1));
          add_coordinate(l_x,l_y);
	  if(get_width() * get_height() > l_index + 1)
	    {
              if(l_x >= l_y)
                {
                  if(l_x + 1 < get_width() && !l_situation.contains_piece(l_x + 1, l_y))
                    {
                      l_x = l_x + 1;
                    }
                  else if(l_y + 1 < get_height()  && !l_situation.contains_piece(l_x, l_y + 1))
                    {
                      l_y = l_y + 1;
                    }
                  else
                    {
                      assert(l_x  && !l_situation.contains_piece(l_x - 1, l_y));
                      l_x = l_x - 1;
                    }
                }
              else
                {
                  if(l_x  && !l_situation.contains_piece(l_x - 1, l_y))
                    {
                      l_x = l_x - 1;
                    }
                  else if(l_y  && !l_situation.contains_piece(l_x, l_y - 1))
                    {
                      l_y = l_y - 1;
                    }
                  else
                    {
                      assert(l_x + 1 < get_width() && !l_situation.contains_piece(l_x + 1, l_y));
                      l_x = l_x + 1;
                    }
                }
	    }
	}
    }
}

#endif // EMP_SPIRAL_STRATEGY_GENERATORH
//EOF
