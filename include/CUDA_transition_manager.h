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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_TRANSITION_MANAGER_H
#define EDGE_MATCHING_PUZZLE_CUDA_TRANSITION_MANAGER_H

#ifndef __NVCC__
#error This code should be compiled with nvcc
#endif // __NVCC__

#include "CUDA_memory_managed_pointer.h"
#include "CUDA_situation_capability.h"

namespace edge_matching_puzzle
{
    template<unsigned int NB_PIECES>
    class CUDA_transition_manager
    : public my_cuda::CUDA_memory_managed_item
    {
      public:

        inline explicit
        CUDA_transition_manager(unsigned int p_nb);

        inline
        ~CUDA_transition_manager();

        inline
        void create_transition(unsigned int p_index);

        inline
        __device__ __host__
        const CUDA_situation_capability<2 * NB_PIECES> & get_transition(unsigned int p_index) const;


        inline
        CUDA_situation_capability<2 * NB_PIECES> & get_transition(unsigned int p_index);

      private:
        using transition_ptr = my_cuda::CUDA_memory_managed_ptr<CUDA_situation_capability<2 * NB_PIECES>>;
        transition_ptr * m_transitions;
        unsigned int m_size;
    };

    //-------------------------------------------------------------------------
    template<unsigned int NB_PIECES>
    CUDA_transition_manager<NB_PIECES>::CUDA_transition_manager(unsigned int p_nb)
    : m_transitions{new transition_ptr[p_nb]}
    , m_size(p_nb)
    {
        for(unsigned int l_index = 0; l_index < p_nb; ++l_index)
        {
            m_transitions[l_index] = 0;
        }
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    CUDA_transition_manager<NB_PIECES>::~CUDA_transition_manager()
    {
        for(unsigned int l_index = 0; l_index < m_size; ++l_index)
        {
            delete(m_transitions[l_index].get());
        }
        delete[] m_transitions;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    __device__ __host__
    const CUDA_situation_capability<2 * NB_PIECES> &
    CUDA_transition_manager<NB_PIECES>::get_transition(unsigned int p_index) const
    {
        //assert(p_index < m_size);
        //assert(m_transitions[p_index].get());
        return *m_transitions[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    CUDA_situation_capability<2 * NB_PIECES> &
    CUDA_transition_manager<NB_PIECES>::get_transition(unsigned int p_index)
    {
        //assert(p_index < m_size);
        //assert(m_transitions[p_index].get());
        return *m_transitions[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    CUDA_transition_manager<NB_PIECES>::create_transition(unsigned int p_index)
    {
        //assert(p_index < m_size);
        //assert(nullptr == m_transitions[p_index].get());
        m_transitions[p_index] = new CUDA_situation_capability<NB_PIECES * 2>();
    }

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_TRANSITION_MANAGER_H
// EOF