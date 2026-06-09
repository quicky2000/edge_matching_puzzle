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
#include "CUDA_common_struct_glutton.h"
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
        is_position_free(situation_index_t p_situation_index
                        ,position_index_t p_position_index
                        ) const;

        inline
        __device__
        void
        play_from(situation_index_t p_dest_situation_index
                 ,const CUDA_glutton_situations & p_source
                 ,situation_index_t p_source_situation_index
                 ,info_index_t p_info_index
                 ,piece_index_t p_piece_index
                 ,emp_types::t_orientation p_orientation
                 ,const CUDA_color_constraints & p_color_constraints
                 );

        /**
         * Get the theoric info for position corresponding to info index in
         * situation corresponding to situation index
         * @param p_situation_index
         * @param p_info_index
         * @return position information
         */
        inline
        const CUDA_piece_position_info2 &
        get_theoric_position_info(situation_index_t p_situation_index
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
        get_position_info(situation_index_t p_situation_index
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
        get_position_info(situation_index_t p_situation_index
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
        get_position_index(situation_index_t p_situation_index
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
        get_info_index(situation_index_t p_situation_index
                      ,position_index_t p_position_index
                      ) const;

        /**
         * Mark position corresponding to position index as occupied
         * @param p_situation_index situation index
         * @param p_position_index position index
         */
        inline
        void
        invalidate_pos2info_index(situation_index_t p_situation_index
                                 ,position_index_t p_position_index
                                 );

        /**
         * Type representing a step
         */
        typedef uint32_t played_info_t;

        protected:
        
        private:

        /**
         * Apply color constraint corresponding to piece and orientation provided
         * as parameter from situation corresponding to source situation index
         * in source situations to situation corresponding to destination
         * situation index in current situations
         */
        inline
        __device__
        void
        apply_color_constraint_from(situation_index_t p_dest_situation_index
                                   ,const CUDA_glutton_situations & p_source
                                   ,situation_index_t p_source_situation_index
                                   ,info_index_t p_info_index
                                   ,piece_index_t p_piece_index
                                   ,emp_types::t_orientation p_orientation
                                   ,const CUDA_color_constraints & p_color_constraints
                                   );

        /**
         * Copy and recompute position/info relation from situation
         * corresponding to source situation index in source situations to
         * situation corresponding to destination situation index in current
         * situations
         */
        inline
        void
        copy_position_info_relation_from(situation_index_t p_dest_situation_index
                                        ,const CUDA_glutton_situations & p_source
                                        ,situation_index_t p_source_situation_index
                                        ,position_index_t p_info_index
                                        );

        /**
         * Copy played info from situation corresponding to source situation
         * index in source situations to situation corresponding to destination
         * situation index in current situations
         * @param p_dest_situation_index index of destination situation in current situations
         * @param p_source source situations
         * @param p_source_situation_index index of source situation in source situations
         */
        inline
        void
        copy_played_info_from(situation_index_t p_dest_situation_index
                             ,const CUDA_glutton_situations & p_source
                             ,situation_index_t p_source_situation_index
                             );

        /**
         * Copy available pieces from situation corresponding to source situation
         * index in source situations to situation corresponding to destination
         * situation index in current situations
         * @param p_dest_situation_index index of destination situation in current situations
         * @param p_source source situations
         * @param p_source_situation_index index of source situation in source situations
         */
        inline
        void
        copy_available_pieces_from(situation_index_t p_dest_situation_index
                                  ,const CUDA_glutton_situations & p_source
                                  ,situation_index_t p_source_situation_index
                                  );

        [[nodiscard]]
        inline
        played_info_t
        get_played_info(situation_index_t p_situation_index
                       ,uint32_t p_index
                       ) const;

        inline
        void
        set_played_info(situation_index_t p_situation_index
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
        set_theoric_position_info(situation_index_t p_situation_index
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
        set_position_info(situation_index_t p_situation_index
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
        bool is_piece_available(situation_index_t p_situation_index
                               ,piece_index_t p_piece_index
                               )const;

        /**
         * Indicate that piece designed by piece index is not used
         * @param p_situation_index
         * @param p_piece_index
         */
        inline
        void set_piece_available(situation_index_t p_situation_index
                                ,piece_index_t p_piece_index
                                );

        /**
         * Indicate that piece designed by piece index is used
         * @param p_situation_index
         * @param p_piece_index
         */
        inline
        void set_piece_unavailable(situation_index_t p_situation_index
                                  ,piece_index_t p_piece_index
                                  );

        /**
         * Store relation between index of position info and position index
         * @param p_situation_index
         * @param p_info_index Index of position info
         * @param p_position_index Position index
         */
        inline
        void
        set_position_info_relation(situation_index_t p_situation_index
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
        compute_info2pos_index(situation_index_t p_situation_index
                              ,info_index_t p_info_index
                              ) const;

        [[nodiscard]]
        inline
        uint32_t
        compute_pos2info_index(situation_index_t p_situation_index
                              ,position_index_t p_position_index
                              ) const;

        [[nodiscard]]
        inline
        uint32_t
        compute_info_global_index(situation_index_t p_situation_index
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
        compute_played_info_index(situation_index_t p_situation_index
                                 ,uint32_t p_level_index
                                 ) const;

        [[nodiscard]]
        inline
        uint32_t
        compute_available_piece_index(situation_index_t p_situation_index
                                     ,piece_index_t p_piece_index
                                     ) const;

        /**
         * Compute index in m_available_pieces array corresponding to situation
         * index and available piece index. This is use only to copy part of
         * the array when copying available pieces from one situation to another.
         * @param p_situation_index situation index
         * @param p_available_piece_index available piece index between 0 and 7
         * @return index in m_available_pieces array corresponding to situation index and available piece index
         */
        [[nodiscard]]
        inline
        uint32_t
        compute_raw_available_pieces_index(situation_index_t p_situation_index
                                          ,uint32_t p_available_pieces_index
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
            this->set_theoric_position_info(static_cast<situation_index_t>(0), static_cast<info_index_t>(l_index), p_initial_capability[l_index]);
            this->set_position_info(static_cast<situation_index_t>(0), static_cast<info_index_t>(l_index), p_initial_capability[l_index]);
            set_position_info_relation(static_cast<situation_index_t>(0), static_cast<info_index_t>(l_index), static_cast<position_index_t >(l_index));
            set_piece_available(static_cast<situation_index_t>(0), static_cast<piece_index_t>(l_index));
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
    CUDA_glutton_situations::is_piece_available(situation_index_t p_situation_index
                                               ,piece_index_t p_piece_index
                                               ) const
    {
        // Do not check with m_puzzle_size as array is designed to support up
        // to 256 piece. As CUDA implementation rely on warp, for small puzzle
        // it is possible that some threads check for piece whose id is greater
        // than puzzle size. As it is initialised with 0 pieces whose index is
        // greater than puzzle size will be unavailable
        assert(p_piece_index < 256);
        assert(p_situation_index < m_nb_situation);
        return m_available_pieces[compute_available_piece_index(p_situation_index, p_piece_index)] & (1u << CUDA_common_struct_glutton::compute_bit_index(static_cast<uint32_t>(p_piece_index)));
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_piece_available(situation_index_t p_situation_index
                                                ,piece_index_t p_piece_index
                                                )
    {
        assert(p_piece_index < 256);
        assert(p_situation_index < m_nb_situation);
        assert(!is_piece_available(p_situation_index, p_piece_index));
#ifdef STRICT_CHECKING
        assert(p_piece_index < m_puzzle_size);
#endif // STRICT_CHECKING
        m_available_pieces[compute_available_piece_index(p_situation_index, p_piece_index)] |= (1u << CUDA_common_struct_glutton::compute_bit_index(static_cast<uint32_t>(p_piece_index)));
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_piece_unavailable(situation_index_t p_situation_index
                                                  ,piece_index_t p_piece_index
                                                  )
    {
        assert(p_piece_index < 256);
        assert(p_situation_index < m_nb_situation);
        assert(is_piece_available(p_situation_index, p_piece_index));
#ifdef STRICT_CHECKING
        assert(p_piece_index < m_puzzle_size);
#endif // STRICT_CHECKING
        m_available_pieces[compute_available_piece_index(p_situation_index, p_piece_index)] &= ~(1u << CUDA_common_struct_glutton::compute_bit_index(static_cast<uint32_t>(p_piece_index)));
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_played_info(situation_index_t p_situation_index
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
    CUDA_glutton_situations::get_played_info(situation_index_t p_situation_index
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
    CUDA_glutton_situations::is_position_free(situation_index_t p_situation_index
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
    CUDA_glutton_situations::set_position_info_relation(situation_index_t p_situation_index
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
    CUDA_glutton_situations::set_theoric_position_info(situation_index_t p_situation_index
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
    CUDA_glutton_situations::get_theoric_position_info(situation_index_t p_situation_index
                                                      ,info_index_t p_info_index
                                                      ) const
    {
        // Boundary checking is done in index computation
        return m_theoric_position_infos[compute_info_global_index(p_situation_index, p_info_index)];
    }

    //-------------------------------------------------------------------------
    const CUDA_piece_position_info2 &
    CUDA_glutton_situations::get_position_info(situation_index_t p_situation_index
                                              ,info_index_t p_info_index
                                              ) const
    {
        // Boundary checking is done in index computation
        return m_position_infos[compute_info_global_index(p_situation_index, p_info_index)];
    }

    //-------------------------------------------------------------------------
    CUDA_piece_position_info2 &
    CUDA_glutton_situations::get_position_info(situation_index_t p_situation_index
                                              ,info_index_t p_info_index
                                              )
    {
        // Boundary checking is done in index computation
        return m_position_infos[compute_info_global_index(p_situation_index, p_info_index)];
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::set_position_info(situation_index_t p_situation_index
                                              ,info_index_t p_info_index
                                              ,const CUDA_piece_position_info2 & p_info
                                              )
    {
        // Boundary checking is done in index computation
        m_position_infos[compute_info_global_index(p_situation_index, p_info_index)] = p_info;
    }

    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_glutton_situations::play_from(situation_index_t p_dest_situation_index
                                      ,const CUDA_glutton_situations & p_source
                                      ,situation_index_t p_source_situation_index
                                      ,info_index_t p_info_index
                                      ,piece_index_t p_piece_index
                                      ,emp_types::t_orientation p_orientation
                                      ,const CUDA_color_constraints & p_color_constraints
                                      )
    {
        assert(this->m_level == (p_source.m_level + 1));
        assert(p_dest_situation_index < m_nb_situation);
        assert(p_source_situation_index < p_source.m_nb_situation);
        assert(p_piece_index < m_puzzle_size);

        // Need to look at source position because at this step destination situation is not filled
        auto l_position_index = p_source.get_position_index(p_source_situation_index, p_info_index);
        assert(p_source.is_position_free(p_source_situation_index, l_position_index));

        apply_color_constraint_from(p_dest_situation_index
                                   ,p_source
                                   ,p_source_situation_index
                                   ,p_info_index
                                   ,p_piece_index
                                   ,p_orientation
                                   ,p_color_constraints
                                   );
        copy_position_info_relation_from(p_dest_situation_index
                                        ,p_source
                                        ,p_source_situation_index
                                        ,l_position_index
                                        );
        copy_played_info_from(p_dest_situation_index
                             ,p_source
                             ,p_source_situation_index
                             );
        copy_available_pieces_from(p_dest_situation_index
                                  ,p_source
                                  ,p_source_situation_index
                                  );
        set_piece_unavailable(p_dest_situation_index, p_piece_index);
        set_played_info(p_dest_situation_index, m_level - 1, CUDA_common_struct_glutton::generate_played_info(l_position_index, p_piece_index, static_cast<uint32_t>(p_orientation)));
    }

    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_glutton_situations::apply_color_constraint_from(situation_index_t p_dest_situation_index
                                                        ,const CUDA_glutton_situations & p_source
                                                        ,situation_index_t p_source_situation_index
                                                        ,info_index_t p_info_index
                                                        ,piece_index_t p_piece_index
                                                        ,emp_types::t_orientation p_orientation
                                                        ,const CUDA_color_constraints & p_color_constraints
                                                        )
    {
        assert(this->m_level == (p_source.m_level + 1));
        assert(p_dest_situation_index < m_nb_situation);
        assert(p_source_situation_index < p_source.m_nb_situation);
        assert(p_piece_index < m_puzzle_size);

        // Need to look at source position because at this step destination situation is not filled
        auto l_position_index = p_source.get_position_index(p_source_situation_index, p_info_index);
        assert(p_source.is_position_free(p_source_situation_index, l_position_index));

#ifdef ENABLE_CUDA_CODE
        uint32_t l_mask_to_apply = threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(CUDA_piece_position_info2::compute_piece_bit_index(static_cast<piece_index_t>(p_piece_index), p_orientation))): 0xFFFFFFFFu;;
        info_index_t l_related_positions = p_info_index;
#else // ENABLE_CUDA_CODE
        pseudo_CUDA_thread_variable<uint32_t> l_mask_to_apply{[=](dim3 threadIdx){return CUDA_piece_position_info2::compute_piece_word_index(static_cast<piece_index_t>(p_piece_index)) == threadIdx.x ? (~CUDA_piece_position_info2::compute_piece_mask(CUDA_piece_position_info2::compute_piece_bit_index(static_cast<piece_index_t>(p_piece_index), p_orientation))): 0xFFFFFFFFu;}};
        pseudo_CUDA_thread_variable<info_index_t> l_related_positions{p_info_index};
#endif
        //---------------------------------------------------------------------
        //- WARNING !!!! THE CURRENT CODE DON'T MANAGE THEORIC INFO
        //---------------------------------------------------------------------

        // Appply the four colours constraints
        for(emp_types::t_orientation l_orientation : emp_types::get_orientations())
        {
            uint32_t l_color_id = g_pieces[static_cast<uint32_t>(p_piece_index)][(static_cast<uint32_t>(l_orientation) + static_cast<uint32_t>(p_orientation)) % 4];

            if (!l_color_id)
            {
                continue;
            }

            position_index_t l_related_position = l_position_index + g_position_offset[static_cast<uint32_t>(l_orientation)];
            if(!p_source.is_position_free(p_source_situation_index, l_related_position))
            {
                continue;
            }
            info_index_t l_related_info_index = p_source.get_info_index(p_source_situation_index, l_related_position);
            uint32_t l_offset = p_info_index < l_related_info_index;

#ifdef ENABLE_CUDA_CODE
            l_related_positions = threadIdx.x == static_cast<uint32_t>(l_orientation) ? l_related_info_index : l_related_positions;
            uint32_t l_constraint_capability = p_color_constraints.get_info(l_color_id - 1, static_cast<uint32_t>(l_orientation)).get_word(static_cast<u32_word_index_t>(threadIdx.x));
            l_constraint_capability &= l_mask_to_apply;

            uint32_t l_capability = p_source.get_position_info(p_source_situation_index, l_related_info_index).get_word(static_cast<u32_word_index_t>(threadIdx.x));
            uint32_t l_result_capability = l_capability & l_constraint_capability;
#else // ENABLE_CUDA_CODE
            l_related_positions[static_cast<uint32_t>(l_orientation)] = l_related_info_index;

            pseudo_CUDA_thread_variable<uint32_t> l_constraint_capability{[&](dim3 threadIdx) { return p_color_constraints.get_info(l_color_id - 1, static_cast<uint32_t>(l_orientation)).get_word(static_cast<u32_word_index_t>(threadIdx.x));}};
            l_constraint_capability &= l_mask_to_apply;

            pseudo_CUDA_thread_variable<uint32_t> l_capability{[&](dim3 threadIdx) { return p_source.get_position_info(p_source_situation_index, l_related_info_index).get_word(static_cast<u32_word_index_t>(threadIdx.x));}};
            pseudo_CUDA_thread_variable<uint32_t> l_result_capability{l_capability & l_constraint_capability};
#endif // ENABLE_CUDA_CODE

            if(!__any_sync(0xFFFFFFFFu, l_result_capability))
            {
                exit(-1);
            }
#ifdef ENABLE_CUDA_CODE
            this->get_position_info(p_dest_situation_index, l_related_info_index - l_offset).set_word(static_cast<u32_word_index_t>(threadIdx.x)
                                                                                                     ,l_result_capability
                                                                                                     );
#else // ENABLE_CUDA_CODE
            for (dim3 threadIdx{0, 1, 1}; threadIdx.x < 32; ++threadIdx.x)
            {
#if VERBOSITY_LEVEL >= 6
                    my_cuda::print_mask(5, l_print_mask, threadIdx, "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability[threadIdx.x], l_mask_to_apply[threadIdx.x], l_result[threadIdx.x]);
#endif // VERBOSITY_LEVEL >= 6
                    this->get_position_info(p_dest_situation_index, l_related_info_index - l_offset).set_word(static_cast<u32_word_index_t>(threadIdx.x)
                                                                                                             ,l_result_capability[threadIdx.x]
                                                                                                             );
            }
#endif // ENABLE_CUDA_CODE

        }

        // Apply the mask on other info
        for(info_index_t l_index{0}; l_index < info_index_t{p_source.get_situation_info_nb()}; ++l_index)
        {
#ifdef ENABLE_CUDA_CODE
            if(__any_sync(0x1Fu, l_related_positions == l_index))
            {
                continue;
            }
#else // ENABLE_CUDA_CODE
            pseudo_CUDA_thread_variable<uint32_t> l_matching_related_position{[&](dim3 threadIdx) { return l_related_positions[threadIdx.x] == l_index;}};
            if(__any_sync(0x1Fu, l_matching_related_position))
            {
                continue;
            }
#endif // ENABLE_CUDA_CODE
#ifdef ENABLE_CUDA_CODE
            uint32_t l_capability{p_source.get_position_info(p_source_situation_index, static_cast<info_index_t>(l_index)).get_word(static_cast<u32_word_index_t>(threadIdx.x))};
            uint32_t l_result_capability{l_capability & l_mask_to_apply};
#else // ENABLE_CUDA_CODE
            pseudo_CUDA_thread_variable<uint32_t> l_capability{[&](dim3 threadIdx) { return p_source.get_position_info(p_source_situation_index, static_cast<info_index_t>(l_index)).get_word(static_cast<u32_word_index_t>(threadIdx.x));}};
            pseudo_CUDA_thread_variable<uint32_t> l_result_capability{l_capability & l_mask_to_apply};
#endif // ENABLE_CUDA_CODE
            if(!__any_sync(0xFFFFFFFFu, l_result_capability))
            {
                exit(-1);
            }
            info_index_t l_offset{p_info_index < l_index};
#ifdef ENABLE_CUDA_CODE
            this->get_position_info(p_dest_situation_index, l_index - l_offset).set_word(static_cast<u32_word_index_t>(threadIdx.x)
                                                                                        ,l_result_capability
                                                                                        );
#else // ENABLE_CUDA_CODE
            for (dim3 threadIdx{0, 1, 1}; threadIdx.x < 32; ++threadIdx.x)
            {
#if VERBOSITY_LEVEL >= 6
                my_cuda::print_mask(5, l_print_mask, threadIdx, "Capability 0x%08" PRIx32 "\nConstraint 0x%08" PRIx32 "\nResult     0x%08" PRIx32 "\n", l_capability[threadIdx.x], l_mask_to_apply[threadIdx.x], l_result[threadIdx.x]);
#endif // VERBOSITY_LEVEL >= 6
                this->get_position_info(p_dest_situation_index, l_index - l_offset).set_word(static_cast<u32_word_index_t>(threadIdx.x)
                                                                                            ,l_result_capability[threadIdx.x]
                                                                                            );
            }
#endif // ENABLE_CUDA_CODE
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::copy_position_info_relation_from(situation_index_t p_dest_situation_index
                                                             ,const CUDA_glutton_situations & p_source
                                                             ,situation_index_t p_source_situation_index
                                                             ,position_index_t p_position_index
                                                             )
    {
        assert(this->m_level == (p_source.m_level + 1));
        assert(p_position_index < m_puzzle_size);
        assert(p_dest_situation_index < m_nb_situation);
        assert(p_source_situation_index < p_source.m_nb_situation);

        invalidate_pos2info_index(p_dest_situation_index, p_position_index);

        std::array<std::tuple<position_index_t, position_index_t, info_index_t>, 2> l_ranges
        {{{position_index_t{0}, (static_cast<uint32_t>(p_position_index) - 1 > m_puzzle_size) ? position_index_t{0} : p_position_index - 1, info_index_t(0)},
          {p_position_index + 1, position_index_t{m_puzzle_size}, info_index_t(1)}
         }
        };
        // We need to update the relation for all position info with index greater than info index as they are shifted by one position in new situation compared to source situation
        for(const auto & l_range: l_ranges)
        {
            for(position_index_t l_index{std::get<0>(l_range)}; l_index < std::get<1>(l_range); ++l_index)
            {
                if(get_info_index(p_source_situation_index, l_index) == std::numeric_limits<info_index_t>::max())
                {
                    auto l_pos2info_index = compute_pos2info_index(p_dest_situation_index, l_index);
                    m_position_index_to_info_index[static_cast<uint32_t>(l_pos2info_index)] = std::numeric_limits<info_index_t>::max();
                }
                else
                {
                    info_index_t l_info_index = p_source.get_info_index(p_source_situation_index, l_index);
                    set_position_info_relation(p_dest_situation_index, l_info_index - std::get<2>(l_range), l_index);
                }
            }
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::copy_played_info_from(situation_index_t p_dest_situation_index
                                                  ,const CUDA_glutton_situations & p_source
                                                  ,situation_index_t p_source_situation_index
                                                  )
    {
        assert(this->m_level == (p_source.m_level + 1));
        assert(p_dest_situation_index < m_nb_situation);
        assert(p_source_situation_index < p_source.m_nb_situation);
        for(unsigned int l_index = 0; l_index < p_source.m_level; ++l_index)
        {
            set_played_info(p_dest_situation_index, l_index, p_source.get_played_info(p_source_situation_index, l_index));
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_situations::copy_available_pieces_from(situation_index_t p_dest_situation_index
                                                       ,const CUDA_glutton_situations & p_source
                                                       ,situation_index_t p_source_situation_index
                                                       )
    {
        assert(this->m_level == (p_source.m_level + 1));
        assert(p_dest_situation_index < m_nb_situation);
        assert(p_source_situation_index < p_source.m_nb_situation);
        for(unsigned int l_index = 0; l_index < 8; ++l_index)
        {
            m_available_pieces[compute_raw_available_pieces_index(p_dest_situation_index, l_index)] = p_source.m_available_pieces[compute_raw_available_pieces_index(p_source_situation_index, l_index)];
        }
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_raw_available_pieces_index(situation_index_t p_situation_index
                                                               ,uint32_t p_available_pieces_index
                                                               ) const
    {
        assert(p_available_pieces_index < 8);
        assert(p_situation_index < m_nb_situation);
        return 8 * static_cast<uint32_t>(p_situation_index) + p_available_pieces_index;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    inline
    uint32_t
    CUDA_glutton_situations::compute_info_global_index(situation_index_t p_situation_index
                                                      ,info_index_t p_info_index
                                                      ) const
    {
        assert(p_situation_index < m_nb_situation);
        assert(p_info_index < compute_situation_info_nb(m_level, m_puzzle_size));
        return static_cast<uint32_t>(p_situation_index) * compute_situation_info_nb(m_level, m_puzzle_size) + static_cast<uint32_t>(p_info_index);
    }

    //-------------------------------------------------------------------------
    position_index_t
    CUDA_glutton_situations::get_position_index(situation_index_t p_situation_index
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
    CUDA_glutton_situations::get_info_index(situation_index_t p_situation_index
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
    CUDA_glutton_situations::invalidate_pos2info_index(situation_index_t p_situation_index
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
    CUDA_glutton_situations::compute_info2pos_index(situation_index_t p_situation_index
                                                   ,info_index_t p_info_index
                                                   ) const
    {
        return static_cast<uint32_t>(p_situation_index) * compute_situation_info_nb(m_level, m_puzzle_size) + static_cast<uint32_t>(p_info_index);
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    inline
    uint32_t
    CUDA_glutton_situations::compute_pos2info_index(situation_index_t p_situation_index
                                                   ,position_index_t p_position_index
                                                   ) const
    {
        return static_cast<uint32_t>(p_situation_index) * m_puzzle_size + static_cast<uint32_t>(p_position_index);
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_available_piece_index(situation_index_t p_situation_index
                                                          ,piece_index_t p_piece_index
                                                          ) const
    {
        assert(p_situation_index < m_nb_situation);
        assert(p_piece_index < m_puzzle_size);
        return 8 * static_cast<uint32_t>(p_situation_index) + CUDA_common_struct_glutton::compute_word_index(static_cast<uint32_t>(p_piece_index));
    }
                                 
    //-------------------------------------------------------------------------
    [[nodiscard]]
    uint32_t
    CUDA_glutton_situations::compute_played_info_index(situation_index_t p_situation_index
                                                      ,uint32_t p_level_index
                                                      ) const
    {
        return m_level * static_cast<uint32_t>(p_situation_index) + p_level_index;
    }

}

#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_SITUATIONS_H
//EOF
