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
#ifndef EDGE_MATCHING_PUZZLE_CUDA_BACKTRACKER_STACK_H
#define EDGE_MATCHING_PUZZLE_CUDA_BACKTRACKER_STACK_H

#include "CUDA_situation_capability.h"

#ifndef __NVCC__
#error This code should be compiled with nvcc
#endif // __NVCC__

namespace edge_matching_puzzle
{
    template <unsigned int NB_PIECES>
    class CUDA_backtracker_stack
    : public my_cuda::CUDA_memory_managed_item
    {
      public:

        inline
        unsigned int get_variable_index(unsigned int p_step_index) const;

        /**
         * Store variable index that was selected at this step
         * @param p_step_index
         * @param p_variable_id
         */
        inline
        __host__ __device__
        void set_variable_index( unsigned int p_step_index
                               , unsigned int p_variable_id
                               );

        inline
        __host__ __device__
        CUDA_situation_capability<2 * NB_PIECES> & get_available_variables(unsigned int p_step_index);

        using info_t = typename CUDA_situation_capability<2 * NB_PIECES>::info_t;

      private:
        unsigned int m_variables_index[NB_PIECES];

        /**
         * One extra item compared to NB_PIECES to be able to systematically
         * apply mask even with latest piece
         */
        CUDA_situation_capability<2 * NB_PIECES> m_available_variables[NB_PIECES + 1];
    };

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    unsigned int
    CUDA_backtracker_stack<NB_PIECES>::get_variable_index(unsigned int p_step_index) const
    {
        //assert(p_step_index < NB_PIECES);
        return m_variables_index[p_step_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    __host__ __device__
    CUDA_situation_capability<2 * NB_PIECES> &
    CUDA_backtracker_stack<NB_PIECES>::get_available_variables(unsigned int p_step_index)
    {
        //assert(p_step_index < NB_PIECES + 1);
        return m_available_variables[p_step_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    __host__ __device__
    void
    CUDA_backtracker_stack<NB_PIECES>::set_variable_index(unsigned int p_step_index, unsigned int p_variable_id)
    {
        //assert(p_step_index < NB_PIECES);
        m_variables_index[p_step_index] = p_variable_id;
    }
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_BACKTRACKER_STACK_H
