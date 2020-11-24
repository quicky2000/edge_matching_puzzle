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
#include "situation_capability.h"
#include "feature_CUDA_backtracker.h"
#include "quicky_exception.h"

#include <cassert>
#include <random>
#include <chrono>
#include <memory>
#include <iostream>

namespace edge_matching_puzzle
{
    //-------------------------------------------------------------------------
    void feature_CUDA_backtracker::run()
    {
#ifdef ENABLE_CUDA_CODE
        launch(m_piece_db, m_info, m_variable_generator, m_strategy_generator);
#else
        throw quicky_exception::quicky_logic_exception("You must enable CUDA core for this feature", __LINE__, __FILE__);
#endif // ENABLE_CUDA_CODE
    }

    //-------------------------------------------------------------------------
    feature_CUDA_backtracker::feature_CUDA_backtracker( const emp_piece_db & p_piece_db
                                                      , const emp_FSM_info & p_info
                                                      , std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                                      )
    : m_piece_db(p_piece_db)
    , m_info(p_info)
    , m_variable_generator(p_piece_db, *p_strategy_generator, p_info, "", m_initial_situation)
    , m_strategy_generator(*p_strategy_generator)
    {

    }
}

// EOF
