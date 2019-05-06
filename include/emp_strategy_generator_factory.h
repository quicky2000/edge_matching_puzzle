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

#ifndef _EMP_STRATEGY_GENERATOR_FACTORY_H
#define _EMP_STRATEGY_GENERATOR_FACTORY_H

#include "emp_FSM_info.h"
#include <string>

namespace edge_matching_puzzle
{
    class emp_strategy_generator;

    class emp_strategy_generator_factory
    {
      public:
        static
        emp_strategy_generator * create(const std::string & p_name
                                       ,const emp_FSM_info & p_info
                                       );
      private:
    };
}
#endif // _EMP_STRATEGY_GENERATOR_FACTORY_H
// EOF
