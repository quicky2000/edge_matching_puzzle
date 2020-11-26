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

#ifndef EDGE_MATCHING_PUZZLE_TRANSITION_MANAGER_H
#define EDGE_MATCHING_PUZZLE_TRANSITION_MANAGER_H

#include "CUDA_memory_managed_item.h"
#include "situation_capability.h"

namespace edge_matching_puzzle
{
    template<unsigned int NB_PIECES>
    class transition_manager
#ifdef __NVCC__
    : public CUDA_memory_managed_item
#endif
    {
      public:

        inline explicit
        transition_manager(unsigned int p_nb);

        inline
        ~transition_manager();

        inline
        void create_transition(unsigned int p_index);

        inline
#ifdef __NVCC__
        __device__ __host__
#endif // __NVCC__
        const situation_capability<2 * NB_PIECES> & get_transition(unsigned int p_index) const;


        inline
        situation_capability<2 * NB_PIECES> & get_transition(unsigned int p_index);

      private:
        typedef situation_capability<2 * NB_PIECES> * transition_ptr;
        transition_ptr * m_transitions;
        unsigned int m_size;
    };

    //-------------------------------------------------------------------------
    template<unsigned int NB_PIECES>
    transition_manager<NB_PIECES>::transition_manager(unsigned int p_nb)
#ifndef __NVCC__
    : m_transitions{new transition_ptr[p_nb]}
#else // __NVCC__
    : m_transitions{nullptr}
#endif // __NVCC__

    , m_size(p_nb)
    {
#ifdef __NVCC__
	cudaMallocManaged(&m_transitions, p_nb *sizeof(transition_ptr));
#endif // __NVCC__
        for(unsigned int l_index = 0; l_index < p_nb; ++l_index)
        {
            m_transitions[l_index] = 0;
        }
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    transition_manager<NB_PIECES>::~transition_manager()
    {
        for(unsigned int l_index = 0; l_index < m_size; ++l_index)
        {
            delete(m_transitions[l_index]);
        }
#ifndef __NVCC__
        delete[] m_transitions;
#else // __NVCC__
	cudaFree(m_transitions);
#endif // __NVCC__
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
#ifdef __NVCC__
    __device__ __host__
#endif // __NVCC__
    const situation_capability<2 * NB_PIECES> &
    transition_manager<NB_PIECES>::get_transition(unsigned int p_index) const
    {
#ifndef __NVCC__
        assert(p_index < m_size);
        assert(m_transitions[p_index]);
#endif // __NVCC__
        return *m_transitions[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    situation_capability<2 * NB_PIECES> &
    transition_manager<NB_PIECES>::get_transition(unsigned int p_index)
    {
#ifndef __NVCC__
        assert(p_index < m_size);
        assert(m_transitions[p_index]);
#endif // __NVCC__
        return *m_transitions[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    transition_manager<NB_PIECES>::create_transition(unsigned int p_index)
    {
#ifndef __NVCC__
        assert(p_index < m_size);
        assert(nullptr == m_transitions[p_index]);
#endif // __NVCC__
        m_transitions[p_index] = new situation_capability<NB_PIECES * 2>();
    }

}
#endif //EDGE_MATCHING_PUZZLE_TRANSITION_MANAGER_H
