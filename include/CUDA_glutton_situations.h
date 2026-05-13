/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2026  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATIONS_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATIONS_H

#include "my_cuda.h"
#include "CUDA_memory_managed_item.h"
#include "CUDA_memory_managed_array.h"
#include "CUDA_piece_position_info2.h"
#include "CUDA_types.h"
#include <limits>
#include <cinttypes>

namespace edge_matching_puzzle
{
    /**
     * Class representing EMP situations related info used by CUDA_glutton_wide algorithm
     * It stores:
     * _ real positions info: positions infos computed from played step + shadowed steps due
     * to parallel computing
     * _ theoric positions info: positions infos computed from played step
     * The evaluation of global score is performed on additional position info member
     * computed only from played step
     */
    class CUDA_glutton_situations
#ifdef ENABLE_CUDA_CODE
    : public my_cuda::CUDA_memory_managed_item
#endif //ENABLE_CUDA_CODE
    {
        public:

        inline explicit
        CUDA_glutton_situations(uint32_t p_level
                               ,uint32_t p_puzzle_size
                               ,uint32_t p_nb_situation
                               );

        inline explicit
        CUDA_glutton_situations(uint32_t p_puzzle_size
                               ,CUDA_piece_position_info2 * p_initial_capacity
                               );

        inline
        ~CUDA_glutton_situations();

        [[nodiscard]]
        inline
        uint32_t
        get_nb_situation() const;

        [[nodiscard]]
        inline
        uint32_t
        get_level() const;

        [[nodiscard]]
        inline
        uint32_t
        get_puzzle_size() const;

        /**
         * Return the number of info for each situation
         * @return number of info for each situation
         */
        [[nodiscard]]
        inline
        uint32_t
        get_situation_info_nb() const;

        /**
         * Return true if position specified by p_position_index if free in situation
         * defined by p_situation_index
         * @param p_situation_index situation index
         * @param p_position_index position index
         * @return boolean indicating if position is free (true) or not (false)
         */
        [[nodiscard]]
        inline
        bool
        is_position_free(uint32_t p_situation_index
                        ,position_index_t p_position_index
                        ) const;

        /**
         * Get the theoric info for position corresponding to info index in
         * situation corresponding to situation index
         * @param p_situation_index
         * @param p_info_index
         * @return position information
         */
        inline
        const CUDA_piece_position_info2 &
        get_theoric_position_info(uint32_t p_situation_index
                                 ,info_index_t p_info_index
                                ) const;

        /**
         * Get info for position corresponding to info index in
         * situation corresponding to situation index
         * @param p_situation_index
         * @param p_info_index
         * @param p_info
         */
        inline
        const CUDA_piece_position_info2 &
        get_position_info(uint32_t p_situation_index
                         ,info_index_t p_info_index
                         ) const;

         /**
         * Get info for position corresponding to info index in
         * situation corresponding to situation index
         * @param p_situation_index
         * @param p_info_index
         * @param p_info
         */
        inline
        CUDA_piece_position_info2 &
        get_position_info(uint32_t p_situation_index
                         ,info_index_t p_info_index
                         );
        /**
         * Indicate which position corresponds to info stored at index
         * @param p_situation_index situation index
         * @param p_info_index index in info array
         * @return Position index whose info is stored
         */
        [[nodiscard]]
        inline
        position_index_t
        get_position_index(uint32_t p_situation_index
                          ,info_index_t p_info_index
                          ) const;

        /**
         * Indicate at which index information related to position index is stored
         * @param p_situation_index situation index
         * @param p_position_index
         * @return index in info array here information is stored
         */
        [[nodiscard]]
        inline
        info_index_t
        get_info_index(uint32_t p_situation_index
                      ,position_index_t p_position_index
                      ) const;

        /**
         * Mark position corresponding to position index as occupied
         * @param p_situation_index situation index
         * @param p_position_index position index
         */
        inline
        void
        invalidate_pos2info_index(uint32_t p_situation_index
                                 ,position_index_t p_position_index
                                 );

        /**
         * Help method to compute word index in a bitfield composed of 32 bits
         * words
         * @param p_raw_bit_index
         * @return word index
         */
        [[nodiscard]]
        inline static
        uint32_t compute_word_index(uint32_t p_raw_bit_index);

        /**
         * Help method to compute bit index in a word for a bitfield composed
         * of 32 bits words
         * @param p_raw_bit_index
         * @return bit index
         */
        [[nodiscard]]
        inline static
        uint32_t compute_bit_index(uint32_t p_raw_bit_index);

        [[nodiscard]]
        inline static
        uint32_t
        compute_raw_bit_index(uint32_t p_word_index
                             ,uint32_t p_bit_index
                             );
        /**
         * Type representing a step
         */
        typedef uint32_t played_info_t;

        protected:
        
        private:

        [[nodiscard]]
        inline
        played_info_t
        get_played_info(uint32_t p_situation_index
                       ,uint32_t p_index
                       ) const;

        inline
        void
        set_played_info(uint32_t p_situation_index
                       ,uint32_t p_index
                       ,played_info_t p_played_info
                       );

        /**
         * Set the theoric info for position corresponding to info index in
         * situation corresponding to situation index
         * @param p_situation_index
         * @param p_info_index
         * @param p_info
         */
        inline
        void
        set_theoric_position_info(uint32_t p_situation_index
                                 ,info_index_t p_info_index
                                 ,const CUDA_piece_position_info2 & p_info
                                );

        /**
         * Set info for position corresponding to info index in
         * situation corresponding to situation index
         * @param p_situation_index
         * @param p_info_index
         * @param p_info
         */
        inline
        void
        set_position_info(uint32_t p_situation_index
                         ,info_index_t p_info_index
                         ,const CUDA_piece_position_info2 & p_info
                         );

        /**
         * Indicate if piece designed by piece index is used or not
         * @param p_situation_index
         * @param p_piece_index
         * @return true if piece not used, false if used
         */
        [[nodiscard]]
        inline
        bool is_piece_available(uint32_t p_situation_index
                               ,uint32_t p_piece_index
                               )const;

        /**
         * Indicate that piece designed by piece index is not used
         * @param p_situation_index
         * @param p_piece_index
         */
        inline
        void set_piece_available(uint32_t p_situation_index
                                ,uint32_t p_piece_index
                                );

        /**
         * Indicate that piece designed by piece index is used
         * @param p_situation_index
         * @param p_piece_index
         */
        inline
        void set_piece_unavailable(uint32_t p_situation_index
                                  ,uint32_t p_piece_index
                                  );

        /**
         * Store relation between index of position info and position index
         * @param p_situation_index
         * @param p_info_index Index of position info
         * @param p_position_index Position index
         */
        inline
        void
        set_position_info_relation(uint32_t p_situation_index
                                  ,info_index_t p_info_index
                                  ,position_index_t p_position_index
                                  );

        [[nodiscard]]
        inline static
        uint32_t
        compute_situation_info_nb(uint32_t p_level
                                 ,uint32_t p_puzzle_size
                                 );

        [[nodiscard]]
        inline static
        uint32_t
        compute_info_nb(uint32_t p_level
                       ,uint32_t p_puzzle_size
                       ,uint32_t p_nb_situation
                       );

        [[nodiscard]]
        inline
        uint32_t
        compute_info2pos_index(uint32_t p_situation_index
                              ,info_index_t p_info_index
                              ) const;

        [[nodiscard]]
        inline
        uint32_t
        compute_pos2info_index(uint32_t p_situation_index
                              ,position_index_t p_position_index
                              ) const;

        [[nodiscard]]
        inline
        uint32_t
        compute_info_global_index(uint32_t p_situation_index
                                 ,info_index_t p_info_index
                                 ) const;
        /** 
         * @brief Compute played info index in m_played_info array
         * @param p_situation_index 
         * @param p_level_index 
         * @return played info index
         */
        [[nodiscard]]
        inline
        uint32_t
        compute_played_info_index(uint32_t p_situation_index
                                 ,uint32_t p_level_index
                                 ) const;

        [[nodiscard]]
        inline
        uint32_t
        compute_available_piece_index(uint32_t p_situation_index
                                     ,uint32_t p_piece_index
                                     ) const;

        /**
         * Position info for each free position.
         * It takes in account the shadowed steps
         */
        CUDA_piece_position_info2 * m_position_infos;

        /**
         * Position info for each free position but they are computed only from
         * played step and do not take in account the shadowed steps
         */
        CUDA_piece_position_info2 * m_theoric_position_infos;

        /**
         * Store correspondence between position index and info index
         */
        my_cuda::CUDA_memory_managed_array<position_index_t> m_info_index_to_position_index;

        /**
         * Store correspondence between info index and position index
         */
        my_cuda::CUDA_memory_managed_array<info_index_t> m_position_index_to_info_index;

        /**
         * Store position/piece/orientation selected at level
         */
        my_cuda::CUDA_memory_managed_array<played_info_t> m_played_info;

        /**
         * Store available pieces
         */
        my_cuda::CUDA_memory_managed_array<uint32_t> m_available_pieces;

        /**
         * Level
         */
        uint32_t m_level;

        /**
         * Puzzle size
         */
        uint32_t m_puzzle_size;

        /**
         * Number of situations
         */
        uint32_t m_nb_situation;
    };

    //-------------------------------------------------------------------------
    CUDA_glutton_situations::CUDA_glutton_situations(uint32_t p_level
                                                    ,uint32_t p_puzzle_size
                                                    ,uint32_t p_nb_situation
                                                    )
    :m_position_infos{new CUDA_piece_position_info2[compute_info_nb(p_level, p_puzzle_size, p_nb_situation)]}
    ,m_theoric_position_infos{new CUDA_piece_position_info2[compute_info_nb(p_level, p_puzzle_size, p_nb_situation)]}
    ,m_info_index_to_position_index{compute_info_nb(p_level, p_puzzle_size, p_nb_situation), position_index_t(std::numeric_limits<uint32_t>::max())}
    ,m_position_index_to_info_index{p_puzzle_size * p_nb_situation, info_index_t(std::numeric_limits<uint32_t>::max())}
    ,m_played_info{p_level * p_nb_situation, std::numeric_limits<uint32_t>::max()}
    ,m_available_pieces{8 * p_nb_situation, 0}
    ,m_level{p_level}
    ,m_puzzle_size(p_puzzle_size)
    ,m_nb_situation{p_nb_situation}
    {

    }

    //-------------------------------------------------------------------------
    CUDA_glutton_situations::CUDA_glutton_situations(uint32_t p_puzzle_size
                                                    ,CUDA_piece_position_info2 * p_initial_capability
                                                    )
    :CUDA_glutton_situations{0, p_puzzle_size, 1}
    {
        for(unsigned int l_index = 0; l_index < p_puzzle_size; ++l_index)
        {
            this->set_theoric_position_info(0, static_cast<info_index_t>(l_index), p_initial_capability[l_index]);
            this->set_position_info(0, static_cast<info_index_t>(l_index), p_initial_capability[l_index]);
            set_position_info_relation(0, static_cast<info_index_t>(l_index), static_cast<position_index_t >(l_index));
            set_piece_available(0, l_index);
        }
    }

    //-------------------------------------------------------------------------
    CUDA_glutton_situations::~CUDA_glutton_situations()
    {
        delete[] m_position_infos;
        delete[] m_theoric_position_infos;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::get_nb_situation() const
    {
        return m_nb_situation;
    }
        
    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::get_level() const
    {
        return m_level;
    }
        
    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::get_puzzle_size() const
    {
        return m_puzzle_size;
    }
        
    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::get_situation_info_nb() const
    {
        return compute_situation_info_nb(m_level, m_puzzle_size);
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_situation_info_nb(uint32_t p_level
                                                      ,uint32_t p_puzzle_size
                                                      )
    {
        return p_puzzle_size - p_level;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_info_nb(uint32_t p_level
                                            ,uint32_t p_puzzle_size
                                            ,uint32_t p_nb_situation
                                            )
    {
        return compute_situation_info_nb(p_level, p_puzzle_size) * p_nb_situation;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    bool
    CUDA_glutton_situations::is_piece_available(uint32_t p_situation_index
                                               ,uint32_t p_piece_index
                                               ) const
    {
        // Do not check with m_puzzle_size as array is designed to support up
        // to 256 piece. As CUDA implementation rely on warp, for small puzzle
        // it is possible that some threads check for piece whose id is greater
        // than puzzle size. As it is initialised with 0 pieces whose index is
        // greater than puzzle size will be unavailable
        assert(p_piece_index < 256);
        assert(p_situation_index < m_nb_situation);
        return m_available_pieces[compute_available_piece_index(p_situation_index, p_piece_index)] & (1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_piece_available(uint32_t p_situation_index
                                                ,uint32_t p_piece_index
                                                )
    {
        assert(p_piece_index < 256);
        assert(p_situation_index < m_nb_situation);
        assert(!is_piece_available(p_situation_index, p_piece_index));
#ifdef STRICT_CHECKING
        assert(p_piece_index < m_puzzle_size);
#endif // STRICT_CHECKING
        m_available_pieces[compute_available_piece_index(p_situation_index, p_piece_index)] |= (1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_piece_unavailable(uint32_t p_situation_index
                                                  ,uint32_t p_piece_index
                                                  )
    {
        assert(p_piece_index < 256);
        assert(p_situation_index < m_nb_situation);
        assert(is_piece_available(p_situation_index, p_piece_index));
#ifdef STRICT_CHECKING
        assert(p_piece_index < m_puzzle_size);
#endif // STRICT_CHECKING
        m_available_pieces[compute_available_piece_index(p_situation_index, p_piece_index)] &= ~(1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_played_info(uint32_t p_situation_index
                                            ,uint32_t p_index
                                            ,played_info_t p_played_info
                                            )
    {
#ifdef STRICT_CHECKING
        assert(p_index < m_level);
        assert(p_situation_index < m_nb_situation);
#endif
        m_played_info[compute_played_info_index(p_situation_index, p_index)] = p_played_info;
    }

    //-------------------------------------------------------------------------
    CUDA_glutton_situations::played_info_t
    CUDA_glutton_situations::get_played_info(uint32_t p_situation_index
                                            ,uint32_t p_index
                                            ) const
    {
#ifdef STRICT_CHECKING
        assert(p_index < m_level);
        assert(p_situation_index < m_nb_situation);
#endif
        return m_played_info[compute_played_info_index(p_situation_index, p_index)];
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    bool
    CUDA_glutton_situations::is_position_free(uint32_t p_situation_index
                                             ,position_index_t p_position_index
                                             ) const
    {
#ifdef ENABLE_CUDA_CODE
        return this->get_info_index(p_situation_index, p_position_index) != 0xFFFFFFFF;
#else // ENABLE_CUDA_CODE
        return this->get_info_index(p_situation_index, p_position_index) != std::numeric_limits<uint32_t>::max();
#endif //ENABLE_CUDA_CODE
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_position_info_relation(uint32_t p_situation_index
                                                       ,info_index_t p_info_index
                                                       ,position_index_t p_position_index
                                                       )
    {
        assert(p_info_index < compute_situation_info_nb(m_level, m_puzzle_size));
        assert(p_position_index < m_puzzle_size);
        assert(p_situation_index < m_nb_situation);

        m_position_index_to_info_index[compute_pos2info_index(p_situation_index, p_position_index)] = p_info_index;
        m_info_index_to_position_index[compute_info2pos_index(p_situation_index, p_info_index)] = p_position_index;
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_theoric_position_info(uint32_t p_situation_index
                                                      ,info_index_t p_info_index
                                                      ,const CUDA_piece_position_info2 & p_info
                                                      )
    {
        assert(p_situation_index < m_nb_situation);
        assert(p_info_index < compute_situation_info_nb(m_level, m_puzzle_size));
        m_theoric_position_infos[compute_info_global_index(p_situation_index, p_info_index)] = p_info;
    }

    //-------------------------------------------------------------------------
    const CUDA_piece_position_info2 &
    CUDA_glutton_situations::get_theoric_position_info(uint32_t p_situation_index
                                                      ,info_index_t p_info_index
                                                      ) const
    {
        // Boundary checking is done in index computation
        return m_theoric_position_infos[compute_info_global_index(p_situation_index, p_info_index)];
    }

    //-------------------------------------------------------------------------
    const CUDA_piece_position_info2 &
    CUDA_glutton_situations::get_position_info(uint32_t p_situation_index
                                              ,info_index_t p_info_index
                                              ) const
    {
        // Boundary checking is done in index computation
        return m_position_infos[compute_info_global_index(p_situation_index, p_info_index)];
    }

    //-------------------------------------------------------------------------
    CUDA_piece_position_info2 &
    CUDA_glutton_situations::get_position_info(uint32_t p_situation_index
                                              ,info_index_t p_info_index
                                              )
    {
        // Boundary checking is done in index computation
        return m_position_infos[compute_info_global_index(p_situation_index, p_info_index)];
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_position_info(uint32_t p_situation_index
                                              ,info_index_t p_info_index
                                              ,const CUDA_piece_position_info2 & p_info
                                              )
    {
        // Boundary checking is done in index computation
        m_position_infos[compute_info_global_index(p_situation_index, p_info_index)] = p_info;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    inline
    uint32_t
    CUDA_glutton_situations::compute_info_global_index(uint32_t p_situation_index
                                                      ,info_index_t p_info_index
                                                      ) const
    {
        assert(p_situation_index < m_nb_situation);
        assert(p_info_index < compute_situation_info_nb(m_level, m_puzzle_size));
        return p_situation_index * compute_situation_info_nb(m_level, m_puzzle_size) + static_cast<uint32_t>(p_info_index);
    }

    //-------------------------------------------------------------------------
    position_index_t
    CUDA_glutton_situations::get_position_index(uint32_t p_situation_index
                                               ,info_index_t p_info_index
                                               ) const
    {
        assert(p_info_index < compute_situation_info_nb(m_level, m_puzzle_size));
        assert(p_situation_index < m_nb_situation);
        uint32_t l_info2pos_index = compute_info2pos_index(p_situation_index, p_info_index);
        assert(l_info2pos_index < compute_info_nb(m_level, m_puzzle_size, m_nb_situation));
        return m_info_index_to_position_index[static_cast<uint32_t>(l_info2pos_index)];
    }

    //-------------------------------------------------------------------------
    info_index_t
    CUDA_glutton_situations::get_info_index(uint32_t p_situation_index
                                           ,position_index_t p_position_index
                                           ) const
    {
        assert(p_position_index < m_puzzle_size);
        assert(p_situation_index < m_nb_situation);
        uint32_t l_pos2info_index = compute_pos2info_index(p_situation_index, p_position_index);
        assert(l_pos2info_index < m_puzzle_size * m_nb_situation);
        return m_position_index_to_info_index[static_cast<uint32_t>(l_pos2info_index)];
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::invalidate_pos2info_index(uint32_t p_situation_index
                                                      ,position_index_t p_position_index
                                                      )
    {
        assert(p_position_index < m_puzzle_size);
        assert(p_situation_index < m_nb_situation);
        uint32_t l_pos2info_index = compute_pos2info_index(p_situation_index, p_position_index);
        assert(l_pos2info_index < m_puzzle_size * m_nb_situation);
        m_position_index_to_info_index[static_cast<uint32_t>(l_pos2info_index)] = std::numeric_limits<uint32_t>::max();
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    inline
    uint32_t
    CUDA_glutton_situations::compute_info2pos_index(uint32_t p_situation_index
                                                   ,info_index_t p_info_index
                                                   ) const
    {
        return p_situation_index * compute_situation_info_nb(m_level, m_puzzle_size) + static_cast<uint32_t>(p_info_index);
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    inline
    uint32_t
    CUDA_glutton_situations::compute_pos2info_index(uint32_t p_situation_index
                                                   ,position_index_t p_position_index
                                                   ) const
    {
        return p_situation_index * m_puzzle_size + static_cast<uint32_t>(p_position_index);
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_available_piece_index(uint32_t p_situation_index
                                                          ,uint32_t p_piece_index
                                                          ) const
    {
        assert(p_situation_index < m_nb_situation);
        assert(p_piece_index < m_puzzle_size);
        return 8 * p_situation_index + compute_word_index(p_piece_index);
    }
                                 
    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_played_info_index(uint32_t p_situation_index
                                                      ,uint32_t p_level_index
                                                      ) const
    {
        return m_level * p_situation_index + p_level_index;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_word_index(uint32_t p_raw_bit_index)
    {
        uint32_t l_word_index = p_raw_bit_index / 32;
        return l_word_index;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_bit_index(uint32_t p_raw_bit_index)
    {
        uint32_t l_bit_index = p_raw_bit_index % 32;
        return l_bit_index;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_raw_bit_index(uint32_t p_word_index
                                                  ,uint32_t p_bit_index
                                                  )
    {
        return p_word_index * 32 + p_bit_index;
    }

}

#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATIONS_H
//EOF
