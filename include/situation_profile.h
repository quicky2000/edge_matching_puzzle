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

#ifndef EDGE_MATCHING_PUZZLE_SITUATION_PROFILE_H
#define EDGE_MATCHING_PUZZLE_SITUATION_PROFILE_H

#include <vector>
#include <cassert>
#include <algorithm>

#if __cplusplus < 201703L
#define [[maybe_unused]]
#define [[nodiscard]]
#endif // __cplusplus >= 201703L

namespace edge_matching_puzzle
{
    class situation_profile
    {
      public:

        /**
         * Create a profile from a unique value, typical use is for min/max
         * search
         * @param p_level level of situation this profile willl be compared with
         * @param p_size size of profile ( twice size of situation)
         * @param p_init_value value used to fill the profile
         */
        [[maybe_unused]] inline
        situation_profile(unsigned int p_level
                         ,unsigned int p_size
                         ,unsigned int p_init_value
                         );

        /**
         * Create a profile from situation values ( piece_position_info)
         * @param p_level level of situation
         * @param p_values values composing profile
         */
        inline
        situation_profile(unsigned int p_level
                         ,std::vector<unsigned int> && p_values
                         );

        [[nodiscard]] [[maybe_unused]] inline
        unsigned int get_level()const;

        [[nodiscard]] [[maybe_unused]] inline
        const std::vector<unsigned int> & get_values() const;

        inline
        std::vector<unsigned int> & get_values();

        /**
         * Compute the sum of profile values
         * @return sum of profile values
         */
        [[nodiscard]] inline
        unsigned int compute_total()const;

        /**
         * Indicate if corresponding situation is valid
         * This depend on level and values
         * @return true if situation is valid
         */
        [[nodiscard]] inline
        bool is_valid()const;

        [[nodiscard]] inline
        bool less_than_total(const situation_profile & p_profile)const;

        [[nodiscard]] inline
        bool less_than_max(const situation_profile & p_profile)const;

        [[nodiscard]] inline
        bool less_than_min(const situation_profile & p_profile)const;

        [[nodiscard]] inline
        bool less_than_vector(const situation_profile & p_profile)const;

        [[nodiscard]] inline
        bool less_than_rvector(const situation_profile & p_profile)const;

      private:

        [[nodiscard]] inline
        unsigned int get_max()const;

        [[nodiscard]] inline
        unsigned int get_min()const;

        unsigned int m_level;
        std::vector<unsigned int> m_values;
    };

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    situation_profile::situation_profile(unsigned int p_level
                                        ,unsigned int p_size
                                        ,unsigned int p_init_value
                                        )
    :m_level(p_level)
    ,m_values(p_size, p_init_value)
    {
        assert((2 * p_level) <= m_values.size());
    }

    //-------------------------------------------------------------------------
    situation_profile::situation_profile(unsigned int p_level
                                        ,std::vector<unsigned int> && p_values
                                        )
    :m_level(p_level)
    ,m_values(std::move(p_values))
    {
        assert(!(m_values.size() % 2));
        assert((2 * p_level) <= m_values.size());
        std::sort(m_values.begin(), m_values.end());
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    unsigned int
    situation_profile::get_level() const
    {
        return m_level;
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    const std::vector<unsigned int> &
    situation_profile::get_values() const
    {
        return m_values;
    }

    //-------------------------------------------------------------------------
    std::vector<unsigned int> &
    situation_profile::get_values()
    {
        return m_values;
    }

    //-------------------------------------------------------------------------
    unsigned int
    situation_profile::compute_total() const
    {
        return std::accumulate(m_values.begin()
                              ,m_values.end()
                              ,0
                              ,[](unsigned int p_a
                                 ,unsigned int p_b
                                 )
                                 { return p_a + p_b; }
                              );
    }

    //-------------------------------------------------------------------------
    bool
    situation_profile::is_valid() const
    {
        if(m_level < m_values.size() / 2)
        {
            return m_values[2 * m_level];
        }
        return true;
    }

    //-------------------------------------------------------------------------
    unsigned int
    situation_profile::get_max() const
    {
        return *m_values.rbegin();
    }

    //-------------------------------------------------------------------------
    unsigned int
    situation_profile::get_min() const
    {
        return 2 * m_level < m_values.size() ? m_values[2 * m_level] : 0;
    }

    //-------------------------------------------------------------------------
    bool
    situation_profile::less_than_total(const situation_profile & p_profile)const
    {
        return this->compute_total() < p_profile.compute_total();
    }

    //-------------------------------------------------------------------------
    bool
    situation_profile::less_than_max(const situation_profile & p_profile) const
    {
        return std::pair<unsigned int, unsigned int>(get_max(),compute_total()) < std::pair<unsigned int, unsigned int>(p_profile.get_max(), p_profile.compute_total());
    }

    //-------------------------------------------------------------------------
    bool
    situation_profile::less_than_min(const situation_profile & p_profile) const
    {
        return std::pair<unsigned int, unsigned int>(get_min(),compute_total()) < std::pair<unsigned int, unsigned int>(p_profile.get_min(), p_profile.compute_total());
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    bool
    situation_profile::less_than_vector(const situation_profile & p_profile) const
    {
        return m_values < p_profile.m_values;
    }

    //-------------------------------------------------------------------------
    [[nodiscard]]
    bool
    situation_profile::less_than_rvector(const situation_profile & p_profile) const
    {
        assert(m_values.size() == p_profile.m_values.size());
        unsigned int l_index = m_values.size() - 1;
        while(l_index && m_values[l_index] == p_profile.m_values[l_index])
        {
            --l_index;
        }
        return m_values[l_index] < p_profile.m_values[l_index];
    }
}
#endif //EDGE_MATCHING_PUZZLE_SITUATION_PROFILE_H
// EOF