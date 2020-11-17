/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2020  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EMP_SITUATION_UTILS_H
#define EMP_SITUATION_UTILS_H

#include "feature_if.h"
#include <memory>

namespace edge_matching_puzzle
{
    template<unsigned int SIZE>
    class situation_capability;


    class feature_CUDA_backtracker: public feature_if
    {
      public:

        void run() override ;

    };

    /**
     * Launch CUDA kernels
     */
    void launch( unsigned int p_nb_transition
               , const situation_capability<512> & p_situation
               , const std::shared_ptr< situation_capability<512>[]> & p_results
               , const std::shared_ptr<situation_capability<512>[]> & p_transitions
               );
}
#endif //EMP_SITUATION_UTILS_H
