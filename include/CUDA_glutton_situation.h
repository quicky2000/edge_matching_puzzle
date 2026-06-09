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

#include "CUDA_glutton_situations.h"

namespace edge_matching_puzzle
{
    /**
     * Class representing an EMP situation used by CUDA_glutton_wide algorithm
     * It relies on CUDA_glutton_situations that contain all the information in
     * a way optimised for GPU computation.
     * This class acts as a proxy to simplify access to single situation information
     */
    class CUDA_glutton_situation
    {

        friend
        std::ostream & operator<<(std::ostream & p_stream, const CUDA_glutton_situation & p_situation);

    public:

        /**
         * Pseudo Constructor on a situation stored in CUDA_glutton_situations
         * at index provided as parameter
         * @param p_situations
         * @param p_situation_index
         */
        inline explicit
        CUDA_glutton_situation(CUDA_glutton_situations & p_situations
                              ,situation_index_t p_situation_index
                              );

        inline
        void
        print(unsigned int p_indent_level
             ,std::ostream & p_stream
             ) const;

#if 0
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
#endif // 0

    private:

        CUDA_glutton_situations & m_situations;

        situation_index_t m_situation_index;

    };

    //-------------------------------------------------------------------------
    CUDA_glutton_situation::CUDA_glutton_situation(CUDA_glutton_situations & p_situations
                                                  ,situation_index_t p_situation_index
                                                  )
    :m_situations{p_situations}
    ,m_situation_index{p_situation_index}
    {
        assert(p_situation_index < m_situations.get_nb_situation());
    }

#if 0
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
        return m_situations.is_position_free(m_situation_index, p_position_index);
    }

#endif // 0

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situation::print(unsigned int p_indent_level
                                 ,std::ostream & p_stream
                                 ) const
    {
        p_stream << std::string(p_indent_level, ' ') << "Situation Level " << m_situations.get_level() << ": " << std::endl;
        p_stream << std::string(p_indent_level, ' ')  << "===== Position index <-> Info index =====" << std::endl;
        for(position_index_t l_index{0u}; l_index < m_situations.get_puzzle_size(); ++l_index)
        {
            p_stream << std::string(p_indent_level, ' ') << "Position[" << l_index << "] -> Index " <<
            m_situations.get_info_index(m_situation_index, l_index) << std::endl;
        }
        uint32_t l_nb_info_index = m_situations.get_situation_info_nb();
        for(info_index_t l_index{0u}; l_index < l_nb_info_index; ++l_index)
        {
            p_stream << std::string(p_indent_level, ' ') << "Index[" << l_index << "] -> Position " <<
            m_situations.get_position_index(m_situation_index, l_index) << std::endl;
        }

        for(info_index_t l_index{0u}; l_index < l_nb_info_index; ++l_index)
        {
            p_stream << "Info[" << l_index << "]:" << std::endl;
            p_stream << m_situations.get_position_info(m_situation_index, l_index) << std::endl;
        }
        for(info_index_t l_index{0u}; l_index < l_nb_info_index; ++l_index)
        {
            p_stream << "Theoric info[" << l_index << "]:" << std::endl;
            p_stream << m_situations.get_theoric_position_info(m_situation_index, l_index) << std::endl;
        }
    }

    inline
    std::ostream &
    operator<<(std::ostream & p_stream, const CUDA_glutton_situation & p_situation)
    {
        p_situation.print(0, p_stream);
        return p_stream;
    }

#if 0
    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_glutton_situation::shadow_step(uint32_t p_info_index
                                       ,uint32_t p_word_index
                                       ,uint32_t p_bit_index
                                       )
    {
        m_situations.get_position_info(m_situation_index, p_info_index).clear_bit(p_word_index, p_bit_index);
    }


#endif // 0
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATION_H
//EOF
