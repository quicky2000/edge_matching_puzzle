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
    template <unsigned int SIZE>
    void populate_capability( std::mt19937 & p_generator
                            , situation_capability<SIZE> & p_capability
                            )
    {
        for (unsigned int l_position_piece_index = 0; l_position_piece_index < SIZE; ++l_position_piece_index)
        {
            for(unsigned int l_word_index = 0; l_word_index < 32; ++l_word_index)
            {
                p_capability.get_capability(l_position_piece_index).set_word(l_word_index, p_generator());
            }
        }
    }

    //-------------------------------------------------------------------------
    void feature_CUDA_backtracker::run()
    {
#ifdef ENABLE_CUDA_CODE
        // Generate a situation
        std::mt19937 l_rand_generator{(unsigned int)std::chrono::system_clock::now().time_since_epoch().count()};
        situation_capability<512> l_situation_capability;
        populate_capability(l_rand_generator, l_situation_capability);

        // Generate transitions
        unsigned int l_nb_transition = 32 * 32 * 32 * 4;
        std::cout  << "Size of situation capability : " << sizeof(situation_capability<512>) << std::endl;
	    std::cout << "Allocate transitions" << std::endl;
        std::shared_ptr<situation_capability<512>[]> l_transitions(new situation_capability<512>[l_nb_transition]);
        for(unsigned int l_index = 0; l_index < l_nb_transition; ++l_index)
        {
            populate_capability(l_rand_generator, l_transitions[l_index]);
        }

        // Allocate results. shared_ptr can be used on array as we are in C++17
	    std::cout << "Allocate results" << std::endl;
        std::shared_ptr<situation_capability<512>[]> l_results(new situation_capability<512>[l_nb_transition]);

        auto l_start = std::chrono::steady_clock::now();
        for(unsigned int l_index = 0; l_index < l_nb_transition; ++l_index)
        {
            l_results[l_index].apply_and(l_situation_capability, l_transitions[l_index]);
        }
        auto l_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> l_elapsed_seconds = l_end - l_start;
        std::cout << "CPU elapsed time: " << l_elapsed_seconds.count() << "s" << std::endl;
        launch(l_nb_transition, l_situation_capability, l_results, l_transitions);
#else
        throw quicky_exception::quicky_logic_exception("You must enable CUDA core for this feature", __LINE__, __FILE__);
#endif // ENABLE_CUDA_CODE
    }
}

// EOF
