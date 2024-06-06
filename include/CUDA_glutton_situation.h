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

#ifdef STRICT_CHECKING
    inline
    std::ostream & operator<<(std::ostream & p_stream, const CUDA_glutton_situation & p_situation)
    {
        p_stream << "Situation Level " << p_situation.get_level() << ": " << std::endl;
        p_stream <<  "====== Position index <-> Info index ======" << std::endl;
        for(position_index_t l_index{0u}; l_index < p_situation.get_puzzle_size(); ++l_index)
        {
            p_stream << "Position[" << l_index << "] -> Index " << p_situation.get_info_index(l_index) << std::endl;
        }
        for(info_index_t l_index{0u}; l_index < p_situation.get_nb_info_index(); ++l_index)
        {
            p_stream << "Index[" << l_index << "] -> Position " << p_situation.get_position_index(l_index) << std::endl;
        }

        for(unsigned int l_index = 0; l_index < p_situation.get_nb_info_index(); ++l_index)
        {
            p_stream << "Info[" << l_index << "]:" << std::endl;
            p_stream << p_situation.get_position_info(l_index) << std::endl;
        }
        for(unsigned int l_index = 0; l_index < p_situation.get_nb_info_index(); ++l_index)
        {
            p_stream << "Theoric info[" << l_index << "]:" << std::endl;
            p_stream << p_situation.get_theoric_position_info(l_index) << std::endl;
        }
        return p_stream;
    }
#endif // STRICT_CHECKING

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H
//EOF