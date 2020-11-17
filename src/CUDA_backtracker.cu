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

#include "feature_CUDA_backtracker.h"
#include "situation_capability.h"
#include "my_cuda.h"
#include "quicky_exception.h"
#include <chrono>

namespace edge_matching_puzzle
{

    // threadIdx.x : position in the warp
    // ThreadIdx.y : position index
    // blockIdx
    template<unsigned int SIZE>
    __global__
    void kernel_and_info( situation_capability<SIZE> * p_result_capability
                        , const situation_capability<SIZE> * p_situation_capability
                        , const situation_capability<SIZE> * p_transition_capability
                        , unsigned int p_nb
                        )
    {
        assert(warpSize == blockDim.x);
        unsigned int l_transition_index = threadIdx.y + blockIdx.x * blockDim.y;
        if(l_transition_index < p_nb)
        {
            const situation_capability<SIZE> & l_situation_capability = * p_situation_capability;
            const situation_capability<SIZE> & l_transition_capability = p_transition_capability[l_transition_index];
            situation_capability<SIZE> & l_result_capability = p_result_capability[l_transition_index];
            for (unsigned int l_position_piece_index = 0; l_position_piece_index < SIZE; ++l_position_piece_index)
            {
                uint32_t l_thread_situation_capability = l_situation_capability.get_capability(l_position_piece_index).get_word(threadIdx.x);
                uint32_t l_thread_transition_capability = l_transition_capability.get_capability(l_position_piece_index).get_word(threadIdx.x);
                uint32_t l_thread_result_capability = l_thread_situation_capability & l_thread_transition_capability;
                l_result_capability.get_capability(l_position_piece_index).set_word(threadIdx.x, l_thread_result_capability);
            }
        }
    }

    //-------------------------------------------------------------------------
    void launch( unsigned int p_nb_transition
               , const situation_capability<512> & p_situation
               , const std::shared_ptr< situation_capability<512>[]> & p_results
               , const std::shared_ptr<situation_capability<512>[]> & p_transitions
               )
    {
        dim3 l_block_info(32,32);
        assert(!(p_nb_transition % 32));
        dim3 l_grid_info(p_nb_transition / 32);

        auto l_start = std::chrono::steady_clock::now();

        situation_capability<512> * l_situation_ptr;
        cudaError_t l_cuda_status = cudaMalloc(&l_situation_ptr, sizeof(situation_capability<512>));
        if(cudaSuccess != l_cuda_status) { throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);}

        l_cuda_status = cudaMemcpy(l_situation_ptr, & p_situation, sizeof(situation_capability<512>), cudaMemcpyHostToDevice);
        if(cudaSuccess != l_cuda_status) { throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);}

        situation_capability<512> * l_results_ptr;
        l_cuda_status = cudaMalloc(&l_results_ptr, p_nb_transition * sizeof(situation_capability<512>));
        if(cudaSuccess != l_cuda_status) { throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);}

        situation_capability<512> * l_transitions_ptr;
        l_cuda_status = cudaMalloc(&l_transitions_ptr, p_nb_transition * sizeof(situation_capability<512>));
        if(cudaSuccess != l_cuda_status) { throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);}
        l_cuda_status = cudaMemcpy(l_transitions_ptr, p_transitions.get(), p_nb_transition * sizeof(situation_capability<512>), cudaMemcpyHostToDevice);
        if(cudaSuccess != l_cuda_status) { throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status), __LINE__, __FILE__);}

        auto l_prepare_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> l_elapsed_seconds = l_prepare_end - l_start;
        std::cout << "Prepare elapsed time: " << l_elapsed_seconds.count() << "s" << std::endl;

        // Reset CUDA error status
        cudaGetLastError();
        std::cout << "Launch kernels" << std::endl;
        kernel_and_info<512><<<l_grid_info,l_block_info>>>(l_results_ptr, l_situation_ptr, l_transitions_ptr, p_nb_transition);
        gpuErrChk(cudaGetLastError());

        auto l_run_end = std::chrono::steady_clock::now();
        l_elapsed_seconds = l_run_end - l_prepare_end;
        std::cout << "Run elapsed time: " << l_elapsed_seconds.count() << "s" << std::endl;

        situation_capability<512> l_new_result;
        for(unsigned int l_transition_index = 0; l_transition_index < p_nb_transition; ++l_transition_index)
        {
            l_cuda_status = cudaMemcpy((void *) &l_new_result, (const void *) &l_results_ptr[l_transition_index], sizeof(situation_capability<512>), cudaMemcpyDeviceToHost);
            if (cudaSuccess != l_cuda_status) {throw quicky_exception::quicky_runtime_exception(cudaGetErrorString(l_cuda_status),__LINE__,__FILE__);}
            if (!(l_new_result == p_results[l_transition_index]))
            {
                throw quicky_exception::quicky_logic_exception("Bad comparison at index " + std::to_string(l_transition_index), __LINE__, __FILE__);
            }
        }
        cudaFree(l_situation_ptr);
        cudaFree(l_results_ptr);
        cudaFree(l_transitions_ptr);

        auto l_total_end = std::chrono::steady_clock::now();
        l_elapsed_seconds = l_total_end - l_start;
        std::cout << "Total elapsed time: " << l_elapsed_seconds.count() << "s" << std::endl;
    }

}
// EOF