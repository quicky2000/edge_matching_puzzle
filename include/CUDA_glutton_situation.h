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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H

#include "CUDA_common_struct_glutton.h"

namespace edge_matching_puzzle
{
    /**
     * Class representing an EMP situation used by CUDA_glutton_wide algorithm
     * It relies on CUDA_common_struct_glutton class to real store positions
     * info, positions infos computed from played step + shadowed steps due
     * to parallel computing
     * The evaluation of score is performed on additional position info member
     * computed only from played step
     */
    class CUDA_glutton_situation: public CUDA_common_struct_glutton
    {

#ifdef STRICT_CHECKING
        friend
        std::ostream & operator<<(std::ostream & p_stream, const CUDA_glutton_situation & p_situation);
#endif // STRICT_CHECKING

    public:

        inline explicit
        CUDA_glutton_situation(uint32_t p_level
                              ,uint32_t p_puzzle_size
                              );

        inline explicit
        CUDA_glutton_situation(uint32_t p_level
                              ,uint32_t p_puzzle_size
                              ,CUDA_piece_position_info2 * p_initial_capability
        );

        inline
        ~CUDA_glutton_situation();

        inline
        void
        print(unsigned int p_indent_level
             ,std::ostream & p_stream
             ,uint32_t p_level
             ,uint32_t p_puzzle_size
             ) const;

#ifdef STRICT_CHECKING
        [[nodiscard]]
        inline
        uint32_t
        get_level() const;
#endif // STRICT_CHECKING

        //-------------------------------------------------------------------------
        [[nodiscard]]
        inline
        __device__ __host__
        bool
        is_position_free(position_index_t p_position_index) const;

        /**
         * Fill destination situation with information coming from current
         * situation
         * No check on if this is possible to play according to piece borders
         * It only check if position is free and piece is available
         * @param p_position_index index of position where turn is played
         * @param p_piece_index
         * @param p_orientation_index
         * @param p_dest_situation destination situation
         * @param p_level level of current situation
         * @param p_puzzle_size size of puzzle
         */
        inline
        void
        play_to(position_index_t p_position_index
               ,unsigned int p_piece_index
               ,unsigned int p_orientation_index
               ,CUDA_glutton_situation & p_dest_situation
               ,unsigned int p_level
               ,unsigned int p_puzzle_size
               );

        /**
         * Make this step unavailable for play but it is still there for score
         * @param p_info_index
         * @param p_word_index
         * @param p_bit_index
         */
        inline
        __device__
        void
        shadow_step(uint32_t p_info_index
                   ,uint32_t p_word_index
                   ,uint32_t p_bit_index
                   );

    private:

        /**
         * Helper to compute information when define strict checking is not enabled
         * @param p_level
         * @param p_puzzle_size
         * @return
         */
        [[nodiscard]]
        inline static
        uint32_t
        compute_nb_info_index(uint32_t p_level
                             ,uint32_t p_puzzle_size
        );

        /**
         * Helper to compute information when define strict checking is not enabled
         * @param p_level
         * @param p_puzzle_size
         * @return
         */
        [[nodiscard]]
        inline static
        uint32_t
        compute_info_size(uint32_t p_level
                         ,uint32_t p_puzzle_size
                         );

        [[nodiscard]]
        inline
        __device__ __host__
        const CUDA_piece_position_info2 &
        get_theoric_position_info(uint32_t p_info_index) const;

        inline
        __device__ __host__
        void
        set_theoric_position_info(uint32_t p_info_index, const CUDA_piece_position_info2 & p_info);

        inline
        void
        apply_color_constraint(uint32_t p_color_id
                              ,info_index_t p_info_index
                              ,uint32_t p_mask_to_apply
                              );

        inline
        void
        copy_played_info_to(CUDA_glutton_situation & p_destination
                           ,unsigned int p_size
                           );

        inline
        void
        copy_available_pieces_to(CUDA_glutton_situation & p_destination);

        inline
        void
        copy_position_info_to(CUDA_glutton_situation & p_destination
                             ,position_index_t p_position_index
                             ,unsigned int p_nb_info_index
                             ,unsigned int p_puzzle_size
                             ,unsigned int p_info_size
                             );

        /**
         * Position info for each free position but they are computed only from
         * played step and do not take in account the shadowed steps
         */
        CUDA_piece_position_info2 * m_theoric_position_infos;

    };

    //-------------------------------------------------------------------------
    CUDA_glutton_situation::CUDA_glutton_situation(uint32_t p_level
                                                  ,uint32_t p_puzzle_size
                                                  )
    :CUDA_common_struct_glutton(compute_nb_info_index(p_level, p_puzzle_size)
                               , p_level
                               , p_puzzle_size
                               , compute_info_size(p_level, p_puzzle_size)
                               )
    ,m_theoric_position_infos{new CUDA_piece_position_info2[p_puzzle_size - p_level]}
    {
        for(unsigned int l_index = 0; l_index < p_puzzle_size; ++l_index)
        {
            set_position_info_relation(static_cast<info_index_t>(l_index), static_cast<position_index_t >(l_index));
        }
    }

    //-------------------------------------------------------------------------
    CUDA_glutton_situation::CUDA_glutton_situation(uint32_t p_level
                                                  ,uint32_t p_puzzle_size
                                                  ,CUDA_piece_position_info2 * p_initial_capability
                                                  )
    : CUDA_glutton_situation(p_level, p_puzzle_size)
    {
        for(unsigned int l_index = 0; l_index < p_puzzle_size; ++l_index)
        {
            set_theoric_position_info(l_index, p_initial_capability[l_index]);
            this->set_position_info(l_index, p_initial_capability[l_index]);
        }
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situation::compute_nb_info_index(uint32_t p_level
                                                 ,uint32_t p_puzzle_size
                                                 )
    {
        return p_puzzle_size - p_level;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situation::compute_info_size(uint32_t p_level
                                             ,uint32_t p_puzzle_size
                                             )
    {
        return p_puzzle_size - p_level;
    }

    //-------------------------------------------------------------------------
    CUDA_glutton_situation::~CUDA_glutton_situation()
    {
        delete[] m_theoric_position_infos;
    }

    //-------------------------------------------------------------------------
#ifdef STRICT_CHECKING
    uint32_t
    CUDA_glutton_situation::get_level() const
    {
        return CUDA_common_struct_glutton::get_nb_played_info();
    }
#endif // STRICT_CHECKING

    //-------------------------------------------------------------------------
    __device__ __host__
    bool
    CUDA_glutton_situation::is_position_free(position_index_t p_position_index) const
    {
        return CUDA_common_struct_glutton::_is_position_free(p_position_index);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situation::apply_color_constraint(uint32_t p_color_id
                                                  ,info_index_t p_info_index
                                                  ,uint32_t p_mask_to_apply
                                                  )
    {
#ifdef ENABLE_CUDA_CODE
        //uint32_t l_capability = p_stack.get_position_info(l_related_info_index).get_word(threadIdx.x);
        //uint32_t l_constraint_capability = p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x);
        //l_constraint_capability &= p_mask_to_apply;
        //uint32_t l_result_capability = l_capability & l_constraint_capability;
#else // ENABLE_CUDA_CODE
        //pseudo_CUDA_thread_variable<uint32_t> l_capability{[&](dim3 threadIdx) { return p_stack.get_position_info(l_related_info_index).get_word(threadIdx.x);}};
        //pseudo_CUDA_thread_variable<uint32_t> l_constraint_capability{[&](dim3 threadIdx) { return p_color_constraints.get_info(l_color_id - 1, l_orientation_index).get_word(threadIdx.x);}};
        //l_constraint_capability &= p_mask_to_apply;
        //pseudo_CUDA_thread_variable<uint32_t> l_result_capability{l_capability & l_constraint_capability};
#endif // ENABLE_CUDA_CODE

    }

    //-------------------------------------------------------------------------
    __device__ __host__
    const CUDA_piece_position_info2 &
    CUDA_glutton_situation::get_theoric_position_info(uint32_t p_info_index) const
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < this->get_info_size());
#endif // STRICT_CHECKING
        return m_theoric_position_infos[p_info_index];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    void
    CUDA_glutton_situation::set_theoric_position_info(uint32_t p_info_index, const CUDA_piece_position_info2 & p_info)
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < this->get_info_size());
#endif // STRICT_CHECKING
        m_theoric_position_infos[p_info_index] = p_info;
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situation::print(unsigned int p_indent_level
                                 ,std::ostream & p_stream
                                 ,uint32_t p_level
                                 ,uint32_t p_puzzle_size
                                 ) const
    {
        p_stream << std::string(p_indent_level, ' ') << "Situation Level " << p_level << ": " << std::endl;
        p_stream << std::string(p_indent_level, ' ')  << "===== Position index <-> Info index =====" << std::endl;
        for(position_index_t l_index{0u}; l_index < p_puzzle_size; ++l_index)
        {
            p_stream << std::string(p_indent_level, ' ') << "Position[" << l_index << "] -> Index " <<
            get_info_index(l_index) << std::endl;
        }
        uint32_t l_nb_info_index = compute_nb_info_index(p_level, p_puzzle_size);
        for(info_index_t l_index{0u}; l_index < l_nb_info_index; ++l_index)
        {
            p_stream << std::string(p_indent_level, ' ') << "Index[" << l_index << "] -> Position " <<
            get_position_index(l_index) << std::endl;
        }

        for(unsigned int l_index = 0; l_index < l_nb_info_index; ++l_index)
        {
            p_stream << "Info[" << l_index << "]:" << std::endl;
            p_stream << get_position_info(l_index) << std::endl;
        }
        for(unsigned int l_index = 0; l_index < l_nb_info_index; ++l_index)
        {
            p_stream << "Theoric info[" << l_index << "]:" << std::endl;
            p_stream << get_theoric_position_info(l_index) << std::endl;
        }
    }

#ifdef STRICT_CHECKING
    inline
    std::ostream & operator<<(std::ostream & p_stream, const CUDA_glutton_situation & p_situation)
    {
        p_situation.print(0, p_stream, p_situation.get_level(), p_situation.get_puzzle_size());
        return p_stream;
    }
#endif // STRICT_CHECKING

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situation::play_to(position_index_t p_position_index
                                   ,unsigned int p_piece_index
                                   ,unsigned int p_orientation_index
                                   ,CUDA_glutton_situation & p_dest_situation
                                   ,unsigned int p_level
                                   ,unsigned int p_puzzle_size
    )
    {
#ifdef STRICT_CHECKING
        assert(this->get_level() == p_level);
        assert(p_dest_situation.get_level() == (p_level - 1));
        assert(this->is_piece_available(p_piece_index));
        assert(this->is_position_free(p_position_index));
#endif // STRICT_CHECKING

        copy_played_info_to(p_dest_situation, p_level);
        p_dest_situation.set_played_info(p_level, generate_played_info(p_position_index, p_piece_index, p_orientation_index));

        copy_available_pieces_to(p_dest_situation);
        p_dest_situation.set_piece_unavailable(p_piece_index);

        copy_position_info_to(p_dest_situation, p_position_index
                , p_puzzle_size - p_level
                , p_puzzle_size
                , p_puzzle_size - p_level
        );

    }

    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_glutton_situation::shadow_step(uint32_t p_info_index
                                       ,uint32_t p_word_index
                                       ,uint32_t p_bit_index
                                       )
    {
        this->get_position_info(p_info_index).clear_bit(p_word_index, p_bit_index);
    }


    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situation::copy_available_pieces_to(CUDA_glutton_situation & p_destination)
    {
        for(unsigned int l_index = 0; l_index < 8; ++l_index)
        {
            p_destination.set_raw_available_piece(l_index, get_raw_available_piece(l_index));
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situation::copy_played_info_to(CUDA_glutton_situation & p_destination
                                               ,unsigned int p_size
    )
    {
#ifdef STRICT_CHECKING
        assert(p_size <= this->get_nb_played_info());
        assert(p_size <= p_destination.get_nb_played_info());
#endif // STRICT_CHECKING
        for(unsigned int l_index = 0; l_index < p_size; ++l_index)
        {
            p_destination.set_played_info(l_index, this->get_played_info(l_index));
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situation::copy_position_info_to(CUDA_glutton_situation & p_destination
                                                 ,position_index_t p_position_index
                                                 ,unsigned int p_nb_info_index
                                                 ,unsigned int p_puzzle_size
                                                 ,unsigned int p_info_size
                                                 )
    {
#ifdef STRICT_CHECKING
        assert(p_nb_info_index == this->get_nb_info_index());
        assert(p_puzzle_size == this->get_puzzle_size());
        assert(p_position_index < this->get_puzzle_size());
        assert(p_info_size == this->get_info_size());
#endif // STRICT_CHECKING
        info_index_t l_info_index = get_info_index(p_position_index);
#ifdef STRICT_CHECKING
        assert(l_info_index != std::numeric_limits<uint32_t>::max());
#endif // STRICT_CHECKING

        unsigned int l_new_index = 0;
        for(unsigned int l_index = 0; l_index < static_cast<uint32_t>(l_info_index); ++l_index)
        {
            set_position_info_relation(static_cast<info_index_t>(l_new_index), get_position_index(static_cast<info_index_t>(l_index)));
            p_destination.set_position_info(l_new_index, get_position_info(l_index));
            ++l_new_index;
        }
        for(unsigned int l_index = static_cast<uint32_t>(l_info_index) + 1; l_index < p_info_size ; ++l_index)
        {
            set_position_info_relation(static_cast<info_index_t>(l_new_index), get_position_index(static_cast<info_index_t>(l_index)));
            p_destination.set_position_info(l_new_index, get_position_info(l_index));
            ++l_new_index;
        }

    }

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H
//EOF
