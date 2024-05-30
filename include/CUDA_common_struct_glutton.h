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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_COMMON_STRUCT_GLUTTON_H
#define EDGE_MATCHING_PUZZLE_CUDA_COMMON_STRUCT_GLUTTON_H

#include "my_cuda.h"
#include "CUDA_memory_managed_item.h"
#include "CUDA_memory_managed_array.h"
#include "CUDA_piece_position_info2.h"
#include "CUDA_types.h"
#include <limits>
#include <cinttypes>

#define STRICT_CHECKING

namespace edge_matching_puzzle
{
    class CUDA_common_struct_glutton
#ifdef ENABLE_CUDA_CODE
    : public my_cuda::CUDA_memory_managed_item
#endif //ENABLE_CUDA_CODE
    {
    public:

        friend class CUDA_glutton_stack_XML_converter;

        inline explicit
        CUDA_common_struct_glutton(uint32_t p_nb_info_index
                                  ,uint32_t p_nb_played_info
                                  ,uint32_t p_puzzle_size
                                  ,uint32_t p_info_size
                                  );

        CUDA_common_struct_glutton(const CUDA_common_struct_glutton & ) = delete;

        inline
        ~CUDA_common_struct_glutton();

        /**
         * Indicate if piece designed by piece index is used or not
         * @param p_piece_index
         * @return true if piece not used, false if used
         */
        [[nodiscard]]
        inline
        __host__ __device__
        bool is_piece_available(unsigned int p_piece_index)const;

        /**
         * Indicate that piece designed by piece index is not used
         * @param p_piece_index
         */
        inline
        __host__ __device__
        void set_piece_available(unsigned int p_piece_index);

        /**
         * Indicate that piece designed by piece index is used
         * @param p_piece_index
         */
        inline
        __host__ __device__
        void set_piece_unavailable(unsigned int p_piece_index);

        /**
         * Store relation between index of position info and position index
         * @param p_info_index Index of position info
         * @param p_position_index Position index
         */
        inline
        void
        set_position_info_relation(info_index_t p_info_index
                ,position_index_t p_position_index
        );

        /**
         * Indicate at which index information related to position index is stored
         * @param p_position_index
         * @return index in info array here information is stored
         */
        [[nodiscard]]
        inline
        __device__ __host__
        info_index_t
        get_info_index(position_index_t p_position_index) const;

        /**
         * Indicate at which index information related to position index is stored
         * @param p_position_index
         * @return index in info array here information is stored
         */
        [[nodiscard]]
        inline
        __device__ __host__
        info_index_t &
        get_info_index(position_index_t p_position_index);

        /**
         * Indicate which position corresponds to info stored at index
         * @param p_info_index index in info array
         * @return Position index whose info is stored
         */
        [[nodiscard]]
        inline
        __device__ __host__
        position_index_t
        get_position_index(info_index_t p_info_index) const;

        /**
         * Indicate which position corresponds to info stored at index
         * @param p_info_index index in info array
         * @return Position index whose info is stored
         */
        [[nodiscard]]
        inline
        __device__ __host__
        position_index_t &
        get_position_index(info_index_t p_info_index);

        typedef uint32_t played_info_t;

        [[nodiscard]]
        inline
        __device__ __host__
        played_info_t
        get_played_info(uint32_t p_index) const;

        /**
         * Extract position index from played info
         * @param p_played_info
         * @return position index
         */
        [[nodiscard]]
        static inline
        __host__ __device__
        position_index_t
        decode_position_index(played_info_t p_played_info);

        /**
         * Extract piece index from played info
         * @param p_played_info
         * @return piece index
         */
        [[nodiscard]]
        static inline
        __host__ __device__
        unsigned int
        decode_piece_index(played_info_t p_played_info);

        /**
         * Extract orientation index from played info
         * @param p_played_info
         * @return  orientation index
         */
        [[nodiscard]]
        static inline
        __host__ __device__
        unsigned int
        decode_orientation_index(played_info_t p_played_info);

        /**
         * Encode information of piece position/id/orientation
         * @param p_position_index
         * @param p_piece_index
         * @param p_orientation_index
         * @return encoded info
         */
        [[nodiscard]]
        static inline
        __device__ __host__
        played_info_t
        generate_played_info(position_index_t p_position_index
                            ,unsigned int p_piece_index
                            ,unsigned int p_orientation_index
        );

    protected:

#ifdef STRICT_CHECKING
        [[nodiscard]]
        inline
        uint32_t
        get_nb_played_info() const;

        [[nodiscard]]
        inline
        uint32_t
        get_nb_info_index() const;

        [[nodiscard]]
        inline
        uint32_t
        get_puzzle_size() const;

        [[nodiscard]]
        inline
        uint32_t
        get_info_size() const;

#endif // STRICT_CHECKING

        [[nodiscard]]
        inline
        __device__ __host__
        const CUDA_piece_position_info2 &
        get_position_info(uint32_t p_info_index) const;

        [[nodiscard]]
        inline
        __device__ __host__
        CUDA_piece_position_info2 &
        get_position_info(uint32_t p_info_index);

        inline
        __device__ __host__
        void
        set_position_info(uint32_t p_info_index, const CUDA_piece_position_info2 & p_info);

        /**
         * Store which position corresponds to info stored at index
         * @param p_info_index index in info array
         * @param p_position_index position index
         */
        inline
        __device__ __host__
        void
        set_position_index(info_index_t p_info_index, position_index_t p_position_index);

        //-------------------------------------------------------------------------
        [[nodiscard]]
        inline
        __device__ __host__
        bool
        _is_position_free(position_index_t p_position_index) const;


        inline
        __device__ __host__
        void
        set_played_info(uint32_t p_index, played_info_t p_played_info);

        /**
         * Method only used to perform raw operations such as copy
         * @param p_index index of word
         * @return word value
         */
        [[nodiscard]]
        inline
        uint32_t
        get_raw_available_piece(uint32_t p_index);

        /**
         * Method only used to perform raw operations such as copy
         * @param p_index index of word
         * @param p_value new word value
         */
        inline
        void
        set_raw_available_piece(uint32_t p_index, uint32_t p_value);

    private:

        /**
         * Help method to compute word index in a bitfield composed of 32 bits
         * words
         * @param p_index
         * @return word index
         */
        inline static
        __device__ __host__
        uint32_t compute_word_index(uint32_t p_index);

        /**
         * Help method to compute bit index in a word for a bitfield composed
         * of 32 bits words
         * @param p_index
         * @return bit index
         */
        inline static
        __device__ __host__
        uint32_t compute_bit_index(uint32_t p_index);

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
        uint32_t m_available_pieces[8];

        /**
         * Position info for each free position
         */
        CUDA_piece_position_info2 * m_position_infos;
#ifdef STRICT_CHECKING
        uint32_t m_nb_info_index;

        uint32_t m_nb_played_info;

        uint32_t m_puzzle_size;

        uint32_t m_info_size;
#endif
    };

    //-------------------------------------------------------------------------
    CUDA_common_struct_glutton::CUDA_common_struct_glutton(uint32_t p_nb_info_index
                                                          ,uint32_t p_nb_played_info
                                                          ,uint32_t p_puzzle_size
                                                          ,uint32_t p_info_size)
    : m_info_index_to_position_index{p_nb_info_index, position_index_t(std::numeric_limits<uint32_t>::max())}
    , m_position_index_to_info_index{p_puzzle_size, info_index_t(std::numeric_limits<uint32_t>::max())}
    , m_played_info{p_nb_played_info, std::numeric_limits<uint32_t>::max()}
    , m_available_pieces{0, 0, 0, 0, 0, 0, 0, 0}
    , m_position_infos{new CUDA_piece_position_info2[p_info_size]}
#ifdef STRICT_CHECKING
    , m_nb_info_index{p_nb_info_index}
    , m_nb_played_info{p_nb_played_info}
    , m_puzzle_size{p_puzzle_size}
    ,m_info_size{p_info_size}
#endif
    {
        assert(p_puzzle_size <= 256);
    }

    //-------------------------------------------------------------------------
    CUDA_common_struct_glutton::~CUDA_common_struct_glutton()
    {
        delete[] m_position_infos;
    }

#ifdef STRICT_CHECKING
    //-------------------------------------------------------------------------
    uint32_t
    CUDA_common_struct_glutton::get_nb_played_info() const
    {
        return m_nb_played_info;
    }

    uint32_t
    CUDA_common_struct_glutton::get_nb_info_index() const
    {
        return m_nb_info_index;
    }

    uint32_t
    CUDA_common_struct_glutton::get_puzzle_size() const
    {
        return m_puzzle_size;
    }

    uint32_t
    CUDA_common_struct_glutton::get_info_size() const
    {
        return m_info_size;
    }

#endif

    //-------------------------------------------------------------------------
    __device__ __host__
    const CUDA_piece_position_info2 &
    CUDA_common_struct_glutton::get_position_info(uint32_t p_info_index) const
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < m_info_size);
#endif
        return m_position_infos[p_info_index];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    CUDA_piece_position_info2 &
    CUDA_common_struct_glutton::get_position_info(uint32_t p_info_index)
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < m_info_size);
#endif
        return m_position_infos[p_info_index];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    void
    CUDA_common_struct_glutton::set_position_info(uint32_t p_info_index, const CUDA_piece_position_info2 & p_info)
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < m_info_size);
#endif
        m_position_infos[p_info_index] = p_info;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    position_index_t
    CUDA_common_struct_glutton::get_position_index(info_index_t p_info_index) const
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < m_nb_info_index);
#endif
        return m_info_index_to_position_index[static_cast<uint32_t>(p_info_index)];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    position_index_t &
    CUDA_common_struct_glutton::get_position_index(info_index_t p_info_index)
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < m_nb_info_index);
#endif
        return m_info_index_to_position_index[static_cast<uint32_t>(p_info_index)];
    }

    //-------------------------------------------------------------------------
    void
    CUDA_common_struct_glutton::set_position_info_relation(info_index_t p_info_index
                                                          ,position_index_t p_position_index
                                                          )
    {
        // Should check m_position_index_to_info_index array size but consider that the
        // check is done by caller
#ifdef STRICT_CHECKING
        assert(p_info_index < m_nb_info_index);
        assert(p_position_index < m_puzzle_size);
#endif
        m_position_index_to_info_index[static_cast<uint32_t>(p_position_index)] = p_info_index;
        m_info_index_to_position_index[static_cast<uint32_t>(p_info_index)] = p_position_index;
    }


    //-------------------------------------------------------------------------
    __device__ __host__
    void
    CUDA_common_struct_glutton::set_position_index(info_index_t p_info_index, position_index_t p_position_index)
    {
#ifdef STRICT_CHECKING
        assert(p_info_index < m_nb_info_index);
#endif
        m_info_index_to_position_index[static_cast<uint32_t>(p_info_index)] = p_position_index;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    bool
    CUDA_common_struct_glutton::_is_position_free(position_index_t p_position_index) const
    {
#ifdef ENABLE_CUDA_CODE
        return this->get_info_index(p_position_index) != 0xFFFFFFFF;
#else // ENABLE_CUDA_CODE
        return this->get_info_index(p_position_index) != std::numeric_limits<uint32_t>::max();
#endif //ENABLE_CUDA_CODE
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    info_index_t
    CUDA_common_struct_glutton::get_info_index(position_index_t p_position_index) const
    {
#ifdef STRICT_CHECKING
        assert(p_position_index < m_puzzle_size);
#endif
        return m_position_index_to_info_index[static_cast<uint32_t>(p_position_index)];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    info_index_t &
    CUDA_common_struct_glutton::get_info_index(position_index_t p_position_index)
    {
#ifdef STRICT_CHECKING
        assert(p_position_index < m_puzzle_size);
#endif
        return m_position_index_to_info_index[static_cast<uint32_t>(p_position_index)];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    void
    CUDA_common_struct_glutton::set_played_info(uint32_t p_index, played_info_t p_played_info)
    {
#ifdef STRICT_CHECKING
        assert(p_index < m_nb_played_info);
#endif
        m_played_info[p_index] = p_played_info;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    CUDA_common_struct_glutton::played_info_t
    CUDA_common_struct_glutton::get_played_info(uint32_t p_index) const
    {
#ifdef STRICT_CHECKING
        assert(p_index < m_nb_played_info);
#endif
        return m_played_info[p_index];
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    bool
    CUDA_common_struct_glutton::is_piece_available(unsigned int p_piece_index) const
    {
        // Do not check with m_puzzle_size as array is designed to support up
        // to 256 piece. As CUDA implementation rely on warp, for small puzzle
        // it is possible that some threads check for piece whose id is greater
        // than puzzle size. As it is initialised with 0 pieces whose index is
        // greater than puzzle size will be unavailable
        assert(p_piece_index < 256);
        return m_available_pieces[compute_word_index(p_piece_index)] & (1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    void
    CUDA_common_struct_glutton::set_piece_available(unsigned int p_piece_index)
    {
        assert(p_piece_index < 256);
        assert(!is_piece_available(p_piece_index));
#ifdef STRICT_CHECKING
        assert(p_piece_index < m_puzzle_size);
#endif // STRICT_CHECKING
        m_available_pieces[compute_word_index(p_piece_index)] |= (1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    void
    CUDA_common_struct_glutton::set_piece_unavailable(unsigned int p_piece_index)
    {
        assert(p_piece_index < 256);
        assert(is_piece_available(p_piece_index));
#ifdef STRICT_CHECKING
        assert(p_piece_index < m_puzzle_size);
#endif // STRICT_CHECKING
        m_available_pieces[compute_word_index(p_piece_index)] &= ~(1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    uint32_t
    CUDA_common_struct_glutton::get_raw_available_piece(uint32_t p_index)
    {
        assert(p_index < 8);
        return m_available_pieces[p_index];
    }

    //-------------------------------------------------------------------------
    void
    CUDA_common_struct_glutton::set_raw_available_piece(uint32_t p_index, uint32_t p_value)
    {
        assert(p_index < 8);
        m_available_pieces[p_index] = p_value;
    }


    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_common_struct_glutton::compute_word_index(uint32_t p_index)
    {
        unsigned int l_word_index = p_index / 32;
        return l_word_index;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_common_struct_glutton::compute_bit_index(uint32_t p_index)
    {
        unsigned int l_bit_index = p_index % 32;
        return l_bit_index;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    CUDA_common_struct_glutton::played_info_t
    CUDA_common_struct_glutton::generate_played_info(position_index_t p_position_index,
                                                 unsigned int p_piece_index,
                                                 unsigned int p_orientation_index
    )
    {
        assert(p_position_index < 256);
        assert(p_piece_index < 256);
        assert(p_orientation_index < 4);
        return (p_orientation_index << 16u) | (p_piece_index << 8u) | static_cast<uint32_t>(p_position_index);
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    position_index_t
    CUDA_common_struct_glutton::decode_position_index(CUDA_common_struct_glutton::played_info_t p_played_info)
    {
        return position_index_t(p_played_info & 0xFFu);
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    unsigned int
    CUDA_common_struct_glutton::decode_piece_index(CUDA_common_struct_glutton::played_info_t p_played_info)
    {
        return (p_played_info >> 8u) & 0xFFu;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    unsigned int
    CUDA_common_struct_glutton::decode_orientation_index(CUDA_common_struct_glutton::played_info_t p_played_info)
    {
        return p_played_info >> 16u;
    }

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_COMMON_STRUCT_GLUTTON_H
// EOF
