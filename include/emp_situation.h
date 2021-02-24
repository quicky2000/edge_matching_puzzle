/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2021  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef EDGE_MATCHING_PUZZLE_EMP_SITUATION_H
#define EDGE_MATCHING_PUZZLE_EMP_SITUATION_H

#include "emp_situation_base.h"
#include "common.h"
#include <cinttypes>
#include <vector>

namespace edge_matching_puzzle
{
    /**
     * Implementation of situation with a dumped data structure more readable
     * with an hexdump and structure optimized for comparison: first member
     * used of comparison represents 64 positions, then 16 then 8
     * Everything is size considering eternity2 :
     * 256 pieces
     * 8 bits for piece id
     * 2 bĭts for piece orientation
     * 1 bit for position usage
     */
    class emp_situation: public emp_situation_base
    {
      public:

        /**
         * Constructor for empty situation
         */
        inline explicit
        emp_situation();

        [[maybe_unused]] /**
         * Return situation level ie number of pieces set
         * @return situation level
         */
        [[nodiscard]] [[maybe_unused]] inline
        unsigned int get_level() const;

        /**
         * Indicate if a piece has been put on position whose coordinates have
         * been provided as parameters
         * @param p_x X axis position coordinate
         * @param p_y Y axis position coordinate
         * @return true if a piece occupy this position
         */
        [[nodiscard]] inline
        bool contains_piece(unsigned int p_x
                           ,unsigned int p_y
                           ) const;

        /**
         * Return oriented piece set at position whose coordinates have been
         * provided as parameters
         * Valid on if contains_piece at same position return true
         * Checked by assertion
         * @param p_x X axis position coordinate
         * @param p_y Y axis position coordinate
         * @return Oriented piece set at this position
         */
        [[nodiscard]] inline
        emp_types::t_oriented_piece get_piece(unsigned int p_x
                                             ,unsigned int p_y
                                             ) const;

        /**
         * Set oriented piece at position whose coordinates have been provided
         * as parameters
         * Checked by assertion if position is empty
         * @param p_x X axis position coordinate
         * @param p_y Y axis position coordinate
         * @param p_piece Oriented piece
         */
        inline
        void set_piece(unsigned int p_x
                      ,unsigned int p_y
                      ,const emp_types::t_oriented_piece & p_piece
                      );

        inline
        bool operator<(const emp_situation & p_situation) const;

        /**
         * Indicate if a situation can be reached from this situation
         * @param p_situation situation to check if it can be reached
         * @return true if this is a predecessor
         */
         [[nodiscard]]
        inline
        bool is_predecessor(const emp_situation & p_situation) const;

         /**
          * Set situation as empty situation
          */
         [[maybe_unused]]
         inline
         void reset();

      private:

        /**
         * Indicate if a piece has been put on position whose index has been
         * provided as parameters
         * @param p_position_index Position index
         * @return true if a piece occupy this position
         */
        [[nodiscard]] inline
        bool contains_piece(unsigned int p_position_index) const;

        /**
         * Return oriented piece set at position whose index has been provided
         * as parameters
         * Valid on if contains_piece at same position return true
         * Checked by assertion
         * @param p_position_index Position indexx
         * @return Oriented piece set at this position
         */
        [[nodiscard]] inline
        emp_types::t_oriented_piece get_piece(unsigned int p_position_index) const;

        /**
         * Return orientation of piece set at position whose index has been
         * provided as parameter
         * Valid on if contains_piece at same position return true
         * Checked by assertion
         * @param p_position_index Position index
         * @return Orientation of piece set at this position
         */
        [[nodiscard]] inline
        emp_types::t_orientation get_orientation(unsigned int p_position_index) const;

        /**
         * Return index of piece set at position whose index has been
         * provided as parameter
         * Valid on if contains_piece at same position return true
         * Checked by assertion
         * @param p_position_index Position indexx
         * @return Indexx of piece set at this position
         */
        [[nodiscard]] inline
        emp_types::t_piece_id get_piece_id(unsigned int p_position_index) const;

        /**
         * Set oriented piece at position whose index has been provided as
         * parameter
         * Checked by assertion if position is empty
         * @param p_position_index Position index
         * @param p_piece Oriented piece
         */
        inline
        void set_piece(unsigned int p_position_index
                      ,const emp_types::t_oriented_piece & p_piece
                      );

        /**
         * Set piece orientation at position whose index has been provided as
         * parameter
         * Checked by assertion if position is empty
         * @param p_position_index Position index
         * @param p_orientation Piece orientation
         */
        inline
        void set_piece_orientation(unsigned int p_position_index
                                  , emp_types::t_orientation p_orientation
                                  );

        /**
         * Set piece id at position whose index has been provided as
         * parameter
         * Checked by assertion if position is empty
         * @param p_position_index Position index
         * @param p_piece_id Piece Id
         */
        inline
        void set_piece_id(unsigned int p_position_index
                         ,emp_types::t_piece_id p_piece_id
                         );

        /**
         * Set position whose index has been provided as parameter occupied
         * @param p_position_index
         */
        inline
        void set_position_used(unsigned int p_position_index);

        /**
         * Compute number of words required to stored piece fields
         * @tparam FIELD_SIZE field size in bits
         * @return number of word
         */
        template<unsigned int FIELD_SIZE>
        [[nodiscard]] static
        unsigned int nb_word();

        /**
         * Word type used for bitfields
         */
        typedef uint64_t t_word;

        /**
         * Compute field indexes in internal structure for position whose index
         * has been provided as parameter
         * @tparam FIELD_SIZE Field size in bits
         * @param p_position_index Position index
         * @param p_bitfield bitfield bifield belong to
         * @return a pair with first element representing word index in list of
         * word and with second element representing bitfield start bit index
         */
        template<unsigned int FIELD_SIZE>
        inline static
        std::pair<unsigned int, unsigned int> get_field_indexes(unsigned int p_position_index
                                                               ,const std::vector<t_word> & p_bitfield
                                                               );

        /**
         * Indicate if a vector is a predecessor of an other vector ie
         * p_a can be set equal to p_b by only adding pieces
         * Both vectors should have same size
         * @param p_a potential predecessor
         * @param p_b potential successor
         * @return true if p_a is a predecessor of p_b
         */
        inline static
        bool is_predecessor(const std::vector<t_word> & p_a
                           ,const std::vector<t_word> & p_b
                           );

        template <unsigned int FIELD_SIZE>
        [[nodiscard]]
        bool is_predecessor_fields(const std::vector<t_word> & p_a
                                  ,const std::vector<t_word> & p_b
                                  ) const;

        /**
         * Number of pieces set
         */
        unsigned char m_level;

        /**
         * Positions where a piece has been set
         */
        std::vector<t_word> m_used_positions;

        /**
         * Piece orientation
         */
         std::vector<t_word> m_pieces_orientation;

         /**
          * Pieces index
          */
         std::vector<t_word> m_pieces_index;
    };

    //-------------------------------------------------------------------------
    emp_situation::emp_situation()
    :m_level(0)
    ,m_used_positions(nb_word<1>())
    ,m_pieces_orientation(nb_word<2>())
    ,m_pieces_index(nb_word<8>())
    {

    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    unsigned int
    emp_situation::get_level() const
    {
        return m_level;
    }

    //-------------------------------------------------------------------------
    bool
    emp_situation::contains_piece(unsigned int p_x
                                 ,unsigned int p_y
                                 ) const
    {
        unsigned int l_global_index = get_info().get_position_index(p_x, p_y);
        return contains_piece(l_global_index);
    }

    //-------------------------------------------------------------------------
    emp_types::t_oriented_piece
    emp_situation::get_piece(unsigned int p_x,
                             unsigned int p_y
                            ) const
    {
        unsigned int l_position_index = get_info().get_position_index(p_x, p_y);
        return get_piece(l_position_index);
    }

    //-------------------------------------------------------------------------
    void
    emp_situation::set_piece(unsigned int p_x
                            ,unsigned int p_y
                            ,const emp_types::t_oriented_piece & p_piece
                            )
    {
        unsigned int l_position_index = get_info().get_position_index(p_x, p_y);
        set_piece(l_position_index, p_piece);
        ++m_level;
    }

    //-------------------------------------------------------------------------
    bool
    emp_situation::contains_piece(unsigned int p_position_index) const
    {
        unsigned int l_word_index = p_position_index / 64;
        unsigned int l_bit_index = p_position_index % 64;
        assert(l_word_index < m_used_positions.size());
        return m_used_positions[l_word_index] & ( 1ul << l_bit_index);
    }

    //-------------------------------------------------------------------------
    emp_types::t_oriented_piece
    emp_situation::get_piece(unsigned int p_position_index) const
    {
        return edge_matching_puzzle::emp_types::t_oriented_piece(get_piece_id(p_position_index)
                                                                , get_orientation(p_position_index)
                                                                );
    }

    //-------------------------------------------------------------------------
    emp_types::t_orientation
    emp_situation::get_orientation(unsigned int p_position_index) const
    {
        assert(contains_piece(p_position_index));
        unsigned int l_word_index;
        unsigned int l_bit_index;
        std::tie(l_word_index, l_bit_index) = get_field_indexes<2>(p_position_index, m_pieces_orientation);
        t_word l_orientation = (m_pieces_orientation[l_word_index] >> l_bit_index) & 0x3ul;
        assert(l_orientation < 4);
        return static_cast<emp_types::t_orientation>(l_orientation);
    }

    //-------------------------------------------------------------------------
    emp_types::t_piece_id
    emp_situation::get_piece_id(unsigned int p_position_index) const
    {
        assert(contains_piece(p_position_index));
        unsigned int l_word_index;
        unsigned int l_bit_index;
        std::tie(l_word_index, l_bit_index) = get_field_indexes<8>(p_position_index, m_pieces_index);
        emp_types::t_piece_id l_piece_id = 1 + ((m_pieces_index[l_word_index] >> l_bit_index ) & 0xFFul);
        return l_piece_id;
    }

    //-------------------------------------------------------------------------
    void
    emp_situation::set_piece(unsigned int p_position_index
                            ,const emp_types::t_oriented_piece & p_piece
                            )
    {
        set_piece_orientation(p_position_index, p_piece.second);
        set_piece_id(p_position_index, p_piece.first);
        set_position_used(p_position_index);
    }

    //-------------------------------------------------------------------------
    void
    emp_situation::set_piece_orientation(unsigned int p_position_index
                                        ,emp_types::t_orientation p_orientation
                                        )
    {
        assert(!contains_piece(p_position_index));
        unsigned int l_word_index;
        unsigned int l_bit_index;
        std::tie(l_word_index, l_bit_index) = get_field_indexes<2>(p_position_index, m_pieces_orientation);
        auto l_orientation = static_cast<t_word>(p_orientation);
        assert(l_orientation < 4);
        m_pieces_orientation[l_word_index] |= (l_orientation << l_bit_index);
    }

    //-------------------------------------------------------------------------
    void
    emp_situation::set_piece_id(unsigned int p_position_index
                               ,emp_types::t_piece_id p_piece_id
                               )
    {
        assert(!contains_piece(p_position_index));
        unsigned int l_word_index;
        unsigned int l_bit_index;
        std::tie(l_word_index, l_bit_index) = get_field_indexes<8>(p_position_index, m_pieces_index);
        auto l_piece_index = static_cast<t_word>(p_piece_id - 1);
        assert(l_piece_index < 256);
        m_pieces_index[l_word_index] |= (l_piece_index << l_bit_index);
    }

    //-------------------------------------------------------------------------
    void
    emp_situation::set_position_used(unsigned int p_position_index)
    {
        assert(!contains_piece(p_position_index));
        unsigned int l_word_index;
        unsigned int l_bit_index;
        std::tie(l_word_index, l_bit_index) = get_field_indexes<1>(p_position_index, m_used_positions);
        m_used_positions[l_word_index] |= (1ul << l_bit_index);
    }

    //-------------------------------------------------------------------------
    template<unsigned int FIELD_SIZE>
    unsigned int
    emp_situation::nb_word()
    {
        constexpr unsigned int l_nb_field = (8 * sizeof(t_word)) / FIELD_SIZE;
        return emp_situation_base::get_info().get_nb_pieces() / l_nb_field + ((emp_situation_base::get_info().get_nb_pieces() % l_nb_field) != 0);
    }

    //-------------------------------------------------------------------------
    template<unsigned int FIELD_SIZE>
    std::pair<unsigned int, unsigned int>
    emp_situation::get_field_indexes(unsigned int p_position_index
                                    ,const std::vector<t_word> & p_bitfield
                                    )
    {
        constexpr unsigned int l_nb_field = (8 * sizeof(t_word)) / FIELD_SIZE;
        unsigned int l_word_index = p_position_index / l_nb_field;
        unsigned int l_bit_index = FIELD_SIZE * (p_position_index % l_nb_field);
        assert(l_word_index < p_bitfield.size());
        return {l_word_index, l_bit_index};
    }

    //-------------------------------------------------------------------------
    bool
    emp_situation::operator<(const emp_situation & p_situation) const
    {
        if(m_level != p_situation.m_level)
        {
            return m_level < p_situation.m_level;
        }
        if(m_used_positions != p_situation.m_used_positions)
        {
            return m_used_positions < p_situation.m_used_positions;
        }
        if(m_pieces_orientation != p_situation.m_pieces_orientation)
        {
            return m_pieces_orientation < p_situation.m_pieces_orientation;
        }
        if(m_pieces_index != p_situation.m_pieces_index)
        {
            return m_pieces_index < p_situation.m_pieces_index;
        }
        return false;
    }

    //-------------------------------------------------------------------------
    bool
    emp_situation::is_predecessor(const emp_situation & p_situation) const
    {
        if(m_level >= p_situation.m_level)
        {
            return false;
        }

        if(!is_predecessor(m_used_positions, p_situation.m_used_positions))
        {
            return false;
        }

        if(!is_predecessor_fields<2>(m_pieces_orientation, p_situation.m_pieces_orientation))
        {
            return false;
        }
        return is_predecessor_fields<8>(m_pieces_index, p_situation.m_pieces_index);
    }

    //-------------------------------------------------------------------------
    bool
    emp_situation::is_predecessor(const std::vector<t_word> & p_a,
                                  const std::vector<t_word> & p_b
                                 )
    {
        std::vector<t_word> l_and(p_a.size());
        assert(p_a.size() == p_b.size());
        std::transform(p_a.begin()
                      ,p_a.end()
                      ,p_b.begin()
                      ,l_and.begin()
                      , [](t_word p_wa, t_word p_wb){ return p_wa & p_wb; }
                      );
        return l_and == p_a;
    }

    //-------------------------------------------------------------------------
    template <unsigned int FIELD_SIZE>
    bool
    emp_situation::is_predecessor_fields(const std::vector<t_word> & p_a,
                                         const std::vector<t_word> & p_b
                                        ) const
    {
        constexpr t_word l_field_mask = (1ul << FIELD_SIZE) - 1;
        constexpr unsigned int l_word_bit_size = 8 * sizeof(t_word);
        for(unsigned int l_index = 0; l_index < m_used_positions.size();++l_index)
        {
            t_word l_word = m_used_positions[l_index];
            while(unsigned int l_ffs = ffsll(l_word))
            {
                --l_ffs;
                l_word &= ~(1ul << l_ffs);
                unsigned int l_full_index = FIELD_SIZE * l_ffs + 8 * sizeof(t_word) * l_index ;
                unsigned int l_word_index = l_full_index / l_word_bit_size;
                unsigned int l_bit_index = l_full_index % l_word_bit_size;
                t_word l_mask = l_field_mask << l_bit_index;
                t_word l_xor = p_a[l_word_index] ^ p_b[l_word_index];
                t_word l_masked = l_xor & l_mask;
                if(l_masked)
                {
                    return false;
                }
            }
        }
        return true;
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    void
    emp_situation::reset()
    {
        m_level = 0;
        std::transform(m_used_positions.begin(), m_used_positions.end(), m_used_positions.begin(), [](unsigned int){return 0;});
        std::transform(m_pieces_orientation.begin(), m_pieces_orientation.end(), m_pieces_orientation.begin(), [](unsigned int){return 0;});
        std::transform(m_pieces_index.begin(), m_pieces_index.end(), m_pieces_index.begin(), [](unsigned int){return 0;});
    }

}
#endif //EDGE_MATCHING_PUZZLE_EMP_SITUATION_H
// EOF
