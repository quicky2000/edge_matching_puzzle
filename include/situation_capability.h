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
#ifndef EMP_SITUATION_CAPABILITY_H
#define EMP_SITUATION_CAPABILITY_H

#include "piece_position_info.h"
#include <array>
#include <vector>

#if __cplusplus < 201703L
#define [[maybe_unused]]
#define [[nodiscard]]
#endif // __cplusplus >= 201703L

namespace edge_matching_puzzle
{

    template <unsigned int SIZE>
    class situation_capability;

    template <unsigned int SIZE>
    std::ostream & operator<<(std::ostream & p_stream, const situation_capability<SIZE> & p_capability);

    /**
     * Represent capabilities for a situation:
     * _ for a position which oriented pieces are possible
     * _ for a piece which positions with orientation are possible
     * @tparam SIZE twice the number of pieces/positions as we have info for both
     */
    template <unsigned int SIZE>
    class situation_capability
    {

        friend
        std::ostream & operator<< <>(std::ostream & p_stream, const situation_capability<SIZE> & p_capability);

      public:

        situation_capability() = default;
        situation_capability(const situation_capability &) = default;
        situation_capability & operator=(const situation_capability &) = default;

        [[nodiscard]] [[maybe_unused]] inline
        const piece_position_info &
        get_capability(unsigned int p_index) const;

        [[maybe_unused]] inline
        piece_position_info &
        get_capability(unsigned int p_index);

        inline
        void apply_and( const situation_capability & p_a
                      , const situation_capability & p_b
                      );

        inline
        bool operator==(const situation_capability &) const;

        [[nodiscard]] inline
        std::vector<unsigned int> compute_profile() const;

        /**
         * Return true if profil is valid for that level (ie no prematured
         * locked piece or situation)
         * @param p_profile situation profile
         * @param p_level level index
         * @return true if situation profile is valid
         */
        [[maybe_unused]] inline
        bool is_profile_valid(const std::vector<unsigned int> & p_profile
                             , unsigned int p_level
                             );

        [[maybe_unused]] typedef piece_position_info info_t;

      private:
        std::array<piece_position_info, SIZE> m_capability;

        static_assert(!(SIZE % 2),"Situation capability size is odd whereas it should be even");
    };

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    [[maybe_unused]]
    const piece_position_info &
    situation_capability<SIZE>::get_capability(unsigned int p_index) const
    {
        assert(p_index < SIZE);
        return m_capability[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    [[maybe_unused]]
    piece_position_info &
    situation_capability<SIZE>::get_capability(unsigned int p_index)
    {
        assert(p_index < SIZE);
        return m_capability[p_index];
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    void
    situation_capability<SIZE>::apply_and( const situation_capability & p_a
                                         , const situation_capability & p_b
                                         )
    {
        std::transform( &(p_a.m_capability[0])
                      , &(p_a.m_capability[SIZE])
                      , &(p_b.m_capability[0])
                      , &(m_capability[0])
                      , [=](const piece_position_info & p_first, const piece_position_info & p_second)
                        {piece_position_info l_result;
                        l_result.apply_and(p_first, p_second);
                         return l_result;
                        }
                      );
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    bool
    situation_capability<SIZE>::operator==(const situation_capability & p_operator) const
    {
        return m_capability == p_operator.m_capability;
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    std::ostream & operator<<(std::ostream & p_stream, const situation_capability<SIZE> & p_capability)
    {
        for(unsigned int l_index = 0; l_index < SIZE; ++l_index)
        {
            p_stream << "[" << l_index << "] =>" << std::endl << p_capability.m_capability[l_index] << std::endl;
        }
        p_stream << std::endl;
        return p_stream;
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    std::vector<unsigned int>
    situation_capability<SIZE>::compute_profile() const
    {
        std::vector<unsigned int> l_result(SIZE);
        std::transform(m_capability.begin(), m_capability.end(), l_result.begin(), [](const piece_position_info & p_info)
                                                                                   {
                                                                                       unsigned int l_nb_bits = p_info.get_nb_bits_set();
                                                                                       return l_nb_bits;
                                                                                   }
                      );
        std::sort(l_result.begin(), l_result.end());
        return l_result;
    }

    //-------------------------------------------------------------------------
    template <unsigned int SIZE>
    [[maybe_unused]]
    bool
    situation_capability<SIZE>::is_profile_valid(const std::vector<unsigned int> & p_profile
                                                ,unsigned int p_level
                                                )
    {
        assert(2 * p_level + 1 < p_profile.size());
        return (p_level < (SIZE / 2) - 1) && p_profile[2 * (p_level + 1)];
    }

}
#endif //EMP_SITUATION_CAPABILITY_H
// EOF
