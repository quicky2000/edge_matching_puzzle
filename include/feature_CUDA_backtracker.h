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
#include "emp_piece_db.h"
#include "emp_FSM_info.h"
#include "emp_variable_generator.h"

#include <memory>

namespace edge_matching_puzzle
{
    template<unsigned int SIZE>
    class situation_capability;


    class feature_CUDA_backtracker: public feature_if
    {
      public:

        feature_CUDA_backtracker( const emp_piece_db & p_piece_deb
                                , const emp_FSM_info & p_info
                                , std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                );

        void run() override ;

      private:

        const emp_piece_db & m_piece_db;

        const emp_FSM_info & m_info;

        /**
         * Contains initial situation
         * Should be declared before variable generator to be fully built
         */
        emp_FSM_situation m_initial_situation;

        /**
         * Generate variables of equation system representing the puzzle
         */
        emp_variable_generator m_variable_generator;

        const emp_strategy_generator & m_strategy_generator;

    };

    /**
     * Launch CUDA kernels
     */
    void launch( const emp_piece_db & p_piece_db
               , const emp_FSM_info & p_info
               , const emp_variable_generator & p_variable_generator
               , const emp_strategy_generator & p_strategy_generator
               );
}
#endif //EMP_SITUATION_UTILS_H
