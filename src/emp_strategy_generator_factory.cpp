/*    This file is part of edge_matching_puzzle
      Copyright (C) 2019  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#include "emp_strategy_generator_factory.h"
#include "emp_basic_strategy_generator.h"
#include "emp_spiral_strategy_generator.h"
#include "emp_text_strategy_generator.h"

namespace edge_matching_puzzle
{
    emp_strategy_generator * emp_strategy_generator_factory::create(const std::string & p_name
                                                                   ,const emp_FSM_info & p_info
                                                                   )
    {
        if("basic" == p_name)
        {
            return new emp_basic_strategy_generator(p_info.get_width(), p_info.get_height());
        }
        else if("spiral" == p_name)
        {
            return new emp_spiral_strategy_generator(p_info.get_width(), p_info.get_height());
        }
        else
        {
            return new emp_text_strategy_generator(p_info.get_width(), p_info.get_height(), p_name);
        }
    }
}
// EOF