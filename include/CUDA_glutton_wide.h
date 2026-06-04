/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2024  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_WIDE_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_WIDE_H

#include "CUDA_common_glutton.h"
#include "CUDA_glutton_situation.h"
#include "emp_situation.h"

/**
 * This file declare functions that will be implemented for
 * CUDA: performance. Corresponding implementation is in CUDA_glutton_wide.cu
 * CPU: alternative implementation to debug algorithm. Corresponding implementation is in CUDA_glutton_wide.cpp
 */

namespace edge_matching_puzzle
{
    class CUDA_glutton_wide: public CUDA_common_glutton
    {
    public:

        inline
        CUDA_glutton_wide(const emp_piece_db & p_piece_db
                         ,const emp_FSM_info & p_info
                         )
        : CUDA_common_glutton(p_piece_db, p_info)
        {

        }

        void
        run();

        inline
        void
        play(const CUDA_glutton_situation & p_start_situation
            ,unsigned int p_x
            ,unsigned int p_y
            ,const emp_types::t_oriented_piece & p_piece
            ,CUDA_glutton_situation & p_end_situation
        )
        {
            //p_start_situation.play_to(static_cast<position_index_t>(get_info().get_position_index(p_x, p_y)))
        }

        [[nodiscard]]
        inline static
        std::unique_ptr<CUDA_glutton_situations>
        prepare_situation(const emp_piece_db & p_piece_db
                         ,const emp_FSM_info & p_info
                         ,emp_situation & p_start_situation
        )
        {
            auto * l_initial_capability = prepare_initial_capability(p_piece_db, p_info);
#if __cplusplus >= 201402
            return std::make_unique<CUDA_glutton_situations>(p_info.get_nb_pieces(), l_initial_capability);
#else // __cplusplus
            return std::unique_ptr<CUDA_glutton_situations>(new CUDA_glutton_situations(p_info.get_nb_pieces(), l_initial_capability));
#endif // __cplusplus
        }


    private:

        /**
         * Compute first set bit in position info after provided index
         * @param p_info position info to check
         * @param p_start_index index after which to check bit value
         * @return index of first set bit after provided index or 0 if no bit is set after provided index
         */
        inline
        uint32_t
        ffs(const CUDA_piece_position_info2 & p_info
           ,uint32_t p_start_index
           ) const;

    };

    //-------------------------------------------------------------------------
    uint32_t
    CUDA_glutton_wide::ffs(const CUDA_piece_position_info2 & p_info
                          ,uint32_t p_start_index
                          ) const
    {
#ifdef ENABLE_CUDA_CODE
        // TO IMPLEMENT
#else // ENABLE_CUDA_CODE
        uint32_t l_start_word_index = CUDA_common_struct_glutton::compute_word_index(p_start_index);
        uint32_t l_start_bit_index = CUDA_common_struct_glutton::compute_bit_index(p_start_index);
        uint32_t l_bit_mask = ((1u << l_start_bit_index) - 1u);
        for(unsigned int l_word_index = l_start_word_index; l_word_index < 32; ++l_word_index)
        {
            uint32_t l_word = p_info.get_word(static_cast<u32_word_index_t>(l_word_index)) & ~l_bit_mask;
            int32_t l_fss = __ffs(l_word);
            if (l_fss != 0)
            {
                return l_fss + 32 * l_word_index;
            }
            l_bit_mask = 0;
        }
#endif // ENABLE_CUDA_CODE
        return 0;
    }
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_WIDE_H
// EOF
