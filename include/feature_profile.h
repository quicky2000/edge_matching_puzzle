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

#ifndef EDGE_MATCHING_PUZZLE_FEATURE_PROFILE_H
#define EDGE_MATCHING_PUZZLE_FEATURE_PROFILE_H

#include "feature_if.h"
#include "feature_sys_equa_CUDA_base.h"
#include "situation_string_formatter.h"
#include "emp_situation.h"
#include "VTK_histogram_dumper.h"

namespace edge_matching_puzzle
{
    /**
     * Class to compute extrema for unsigned int
     */
    class simple_stat
    {
      public:
        inline
        simple_stat();

        inline
        void update(unsigned int p_value);

        [[nodiscard]] [[maybe_unused]]

        inline
        unsigned int get_min() const;

        [[nodiscard]] [[maybe_unused]]

        inline
        unsigned int get_max() const;

      private:
        bool m_updated;
        unsigned int m_min;
        unsigned int m_max;
    };

    class profile_level_stats
    {
      public:

        profile_level_stats() = default;

        [[maybe_unused]]
        inline
        void update(const situation_profile & p_profile);

        [[nodiscard]] [[maybe_unused]]
        inline
        const simple_stat & get_low_stats() const;

        [[nodiscard]] [[maybe_unused]]
        inline
        const simple_stat & get_high_stats() const;

        [[nodiscard]] [[maybe_unused]]
        inline
        const simple_stat & get_total_stats() const;

      private:

        simple_stat m_low_stats;
        simple_stat m_high_stats;
        simple_stat m_total_stats;

    };

    template<unsigned int NB_PIECES>
    class profile_stats
    {
      public:

        profile_stats() = default;

        [[nodiscard]] [[maybe_unused]]
        inline
        const profile_level_stats & get_stats(unsigned int p_level) const;

        inline
        void update(const situation_profile & p_profile);

        [[nodiscard]]
        inline
        std::vector<unsigned int> get_low_stats_min() const;

        [[nodiscard]]
        inline
        std::vector<unsigned int> get_low_stats_max() const;

        [[nodiscard]]
        inline
        std::vector<unsigned int> get_high_stats_min() const;

        [[nodiscard]]
        inline
        std::vector<unsigned int> get_high_stats_max() const;

        [[nodiscard]]
        inline
        std::vector<unsigned int> get_total_stats_min() const;

        [[nodiscard]]
        inline
        std::vector<unsigned int> get_total_stats_max() const;

      private:

        std::array<profile_level_stats, NB_PIECES + 1> m_level_stats;
    };

    class feature_profile: public feature_if
                         , public feature_sys_equa_CUDA_base
    {
      public:

        inline
        feature_profile(const emp_piece_db & p_piece_db
                       ,const emp_FSM_info & p_info
                       ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                       );

        // Methods inherited from feature_if
        inline
        void run() override;

        // End of methods inherited from feature_if

      private:

        template<unsigned int NB_PIECES>
        void template_run();

        /**
         * Deep search alogirthm
         * @tparam NB_PIECES Number of pieces
         */
        template <unsigned int NB_PIECES>
        [[maybe_unused]]
        void deepest();

        template <unsigned int NB_PIECES>
        [[maybe_unused]]
        void widest(const emp_situation & p_situations
                   ,const transition_manager<NB_PIECES> & p_transition_manager
                   ,unsigned int p_start_level
                   ,unsigned int p_nb_level
                   ,const situation_capability<2 * NB_PIECES> & p_ref_situation_capability
                   ,std::set<emp_situation> & p_solutions
                   );

        template<unsigned int NB_PIECES>
        void wide(std::set<emp_situation> & p_solutions
                 ,profile_stats<NB_PIECES> & p_profile_stats
                 ,profile_stats<NB_PIECES> & p_profile_stats2
                 ,const std::set<emp_situation> & p_initial_situations
                 ,const transition_manager<NB_PIECES> & p_transition_manager
                 ,const situation_capability<2 * NB_PIECES> & p_ref_situation_capability
                 ,unsigned int p_level
                 );

        /**
         * Dump solution string representations in solutions.txt file
         * @param p_solutions Solutions string representation
         */
        inline static
        void dump_solutions(const std::set<std::string> & p_solutions);

        template <unsigned int NB_PIECES>
        inline static
        situation_capability<2 * NB_PIECES> compute_situation_capability(const emp_situation & p_situation
                                                                        ,const transition_manager<NB_PIECES> & p_transition_manager
                                                                        ,const situation_capability<2 * NB_PIECES> & p_ref_situation_capability
                                                                        ,const emp_situation & p_ref_situation
                                                                        );

        template <unsigned int NB_PIECES>
        static
        void set_piece(emp_situation & p_situation
                      ,const situation_capability<2 * NB_PIECES> & p_situation_capability
                      ,unsigned int p_x
                      ,unsigned int p_y
                      ,unsigned int p_piece_index
                      ,emp_types::t_orientation p_orientation
                      ,situation_capability<2 * NB_PIECES> & p_result_capability
                      ,const transition_manager<NB_PIECES> & p_transition_manager
                      );

        const emp_piece_db & m_piece_db;
    };

    //-------------------------------------------------------------------------
    simple_stat::simple_stat()
    : m_updated{false}
    , m_min{std::numeric_limits<unsigned int>::max()}
    , m_max{std::numeric_limits<unsigned int>::min()}
    {

    }

    //-------------------------------------------------------------------------
    void
    simple_stat::update(unsigned int p_value)
    {
        m_updated = true;
        if(p_value < m_min)
        {
            m_min = p_value;
        }
        if(p_value > m_max)
        {
            m_max = p_value;
        }
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    unsigned int
    simple_stat::get_min() const
    {
        return m_updated ? m_min : 0;
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    unsigned int
    simple_stat::get_max() const
    {
        return m_max;
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    void profile_level_stats::update(const situation_profile & p_profile)
    {
        m_low_stats.update(p_profile.get_min());
        m_high_stats.update(p_profile.get_max());
        m_total_stats.update(p_profile.compute_total());
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    const simple_stat & profile_level_stats::get_low_stats() const
    {
        return m_low_stats;
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    const simple_stat & profile_level_stats::get_high_stats() const
    {
        return m_high_stats;
    }

    //-------------------------------------------------------------------------
    [[maybe_unused]]
    const simple_stat & profile_level_stats::get_total_stats() const
    {
        return m_total_stats;
    }

    //-------------------------------------------------------------------------
    template<unsigned int NB_PIECES>
    [[maybe_unused]]
    void profile_stats<NB_PIECES>::update(const situation_profile & p_profile)
    {
        assert(p_profile.get_level() <= NB_PIECES);
        m_level_stats[p_profile.get_level()].update(p_profile);
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    [[maybe_unused]]
    const profile_level_stats &
    profile_stats<NB_PIECES>::get_stats(unsigned int p_level) const
    {
        assert(p_level <= NB_PIECES);
        return m_level_stats[p_level];
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::vector<unsigned int>
    profile_stats<NB_PIECES>::get_low_stats_min() const
    {
        std::vector<unsigned int> l_result;
        for(auto l_iter: m_level_stats)
        {
            l_result.emplace_back(l_iter.get_low_stats().get_min());
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::vector<unsigned int>
    profile_stats<NB_PIECES>::get_low_stats_max() const
    {
        std::vector<unsigned int> l_result;
        for(auto l_iter: m_level_stats)
        {
            l_result.emplace_back(l_iter.get_low_stats().get_max());
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::vector<unsigned int>
    profile_stats<NB_PIECES>::get_high_stats_min() const
    {
        std::vector<unsigned int> l_result;
        for(auto l_iter: m_level_stats)
        {
            l_result.emplace_back(l_iter.get_high_stats().get_min());
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::vector<unsigned int>
    profile_stats<NB_PIECES>::get_high_stats_max() const
    {
        std::vector<unsigned int> l_result;
        for(auto l_iter: m_level_stats)
        {
            l_result.emplace_back(l_iter.get_high_stats().get_max());
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::vector<unsigned int>
    profile_stats<NB_PIECES>::get_total_stats_min() const
    {
        std::vector<unsigned int> l_result;
        for(auto l_iter: m_level_stats)
        {
            l_result.emplace_back(l_iter.get_total_stats().get_min());
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::vector<unsigned int>
    profile_stats<NB_PIECES>::get_total_stats_max() const
    {
        std::vector<unsigned int> l_result;
        for(auto l_iter: m_level_stats)
        {
            l_result.emplace_back(l_iter.get_total_stats().get_max());
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    feature_profile::feature_profile(const emp_piece_db & p_piece_db
                                    ,const emp_FSM_info & p_info
                                    ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                    )
    :feature_sys_equa_CUDA_base(p_piece_db, p_info, p_strategy_generator, "")
    ,m_piece_db{p_piece_db}
    {

    }

    //-------------------------------------------------------------------------
    void feature_profile::run()
    {
       switch(get_info().get_nb_pieces())
        {
            case 9:
                template_run<9>();
                break;
            case 16:
                template_run<16>();
                break;
            case 25:
                template_run<25>();
                break;
            case 36:
                template_run<36>();
                break;
            case 72:
                template_run<72>();
                break;
            case 256:
                template_run<256>();
                break;
            default:
                throw quicky_exception::quicky_logic_exception("Unsupported size " + std::to_string(get_info().get_width()) + "x" + std::to_string(get_info().get_height()), __LINE__, __FILE__);
        }
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    situation_capability<2 * NB_PIECES> feature_profile::compute_situation_capability(const emp_situation & p_situation
                                                                                     ,const transition_manager<NB_PIECES> & p_transition_manager
                                                                                     ,const situation_capability<2 * NB_PIECES> & p_ref_situation_capability
                                                                                     ,const emp_situation & p_ref_situation
                                                                                     )
    {
        situation_capability<2 * NB_PIECES> l_situation_capability{p_ref_situation_capability};
        unsigned int l_level = p_situation.get_level();
        unsigned int l_nb_pieces = 0;
        for(unsigned int l_x = 0; l_x < p_situation.get_info().get_width() && l_nb_pieces < l_level; ++l_x)
        {
            for(unsigned int l_y = 0; l_y < p_situation.get_info().get_height() && l_nb_pieces < l_level; ++l_y)
            {
                if(p_situation.contains_piece(l_x, l_y) && !p_ref_situation.contains_piece(l_x, l_y))
                {
                    const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_x, l_y);
                    unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id(l_x
                            ,l_y
                            ,l_piece.first - 1
                            ,l_piece.second
                            ,p_situation.get_info()
                                                                                                      );
                    l_situation_capability.apply_and(l_situation_capability, p_transition_manager.get_transition(l_raw_variable_id));
                    ++l_nb_pieces;
                }
            }
        }
        return l_situation_capability;
    }
#define VERBOSE
    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    feature_profile::widest(const emp_situation & p_situation
                           ,const transition_manager<NB_PIECES> & p_transition_manager
                           ,unsigned int p_start_level
                           ,unsigned int p_nb_level
                           ,const situation_capability<2 * NB_PIECES> & p_ref_situation_capability
                           ,std::set<emp_situation> & p_solutions
                           )
    {
        unsigned int l_nb_level = p_start_level + p_nb_level <= NB_PIECES ? p_nb_level : NB_PIECES - p_start_level;
        unsigned int l_level = p_start_level;
        std::set<emp_situation> l_situations[2] = {{p_situation}};
        for(unsigned int l_level_index = 0; l_level_index < l_nb_level; ++l_level_index)
        {
            std::set<emp_situation> & l_current_situations = l_situations[l_level_index % 2];
            std::set<emp_situation> & l_next_situations = l_situations[(1 + l_level_index) % 2];
            l_level = l_level_index + p_start_level;
#ifdef VERBOSE
            std::cout << std::endl << "\rLevel " << l_level << " : " << l_current_situations.size() << " situations " << std::endl;
            unsigned int l_total_nb = l_current_situations.size();
            unsigned int l_percent = 101;
#endif // VERBOSE
            while(!l_current_situations.empty())
            {
#ifdef VERBOSE
                {
                    unsigned int l_new_percent = (100 * l_current_situations.size()) / l_total_nb;
                    if(l_new_percent != l_percent)
                    {
                        l_percent = l_new_percent;
                        std::cout << "\r" << l_percent << "%    ";
                        std::cout.flush();
                    }

                }
#endif // VERBOSE
                const emp_situation & l_current_situation = *l_current_situations.begin();
                // Rebuild situation capability
                situation_capability<2 * NB_PIECES> l_situation_capability = compute_situation_capability(l_current_situation, p_transition_manager, p_ref_situation_capability, p_situation);
#if 0
                situation_capability<2 * NB_PIECES> l_situation_capability{p_ref_situation_capability};
                unsigned int l_nb_pieces = 0;
                for(unsigned int l_x = 0; l_x < l_current_situation.get_info().get_width() && l_nb_pieces < l_level; ++l_x)
                {
                    for(unsigned int l_y = 0; l_y < l_current_situation.get_info().get_height() && l_nb_pieces < l_level; ++l_y)
                    {
                        if(l_current_situation.contains_piece(l_x, l_y))
                        {
                            const emp_types::t_oriented_piece & l_piece = l_current_situation.get_piece(l_x, l_y);
                            unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id(l_x
                                    ,l_y
                                    ,l_piece.first - 1
                                    ,l_piece.second
                                    ,l_current_situation.get_info()
                                                                                                              );
                            l_situation_capability.apply_and(l_situation_capability, p_transition_manager.get_transition(l_raw_variable_id));
                            ++l_nb_pieces;
                        }
                    }
                }
#endif // 0
//                situation_profile l_profile = l_situation_capability.compute_profile(l_level);

                // Search for transitions
                for(unsigned int l_position_index = 0; l_position_index < NB_PIECES; ++l_position_index)
                {
                    piece_position_info & l_piece_position_info{l_situation_capability.get_capability(l_position_index)};
                    if(l_piece_position_info.any_bit_set())
                    {
                        for (unsigned int l_word_index = 0; l_word_index < 32; ++l_word_index)
                        {
                            uint32_t l_word = l_piece_position_info.get_word(l_word_index);
                            while (l_word)
                            {
                                unsigned int l_bit_index = ffs(l_word) - 1;
                                unsigned int l_full_index = 32 * l_word_index + l_bit_index;
                                unsigned int l_x = l_position_index % get_info().get_width();
                                unsigned int l_y = l_position_index / get_info().get_width();
                                unsigned int l_piece_index = l_full_index % 256;
                                auto l_orientation = static_cast<emp_types::t_orientation>(l_full_index / 256);
                                emp_situation l_next_situation{l_current_situation};
                                situation_capability<2 * NB_PIECES> l_new_situation_capability{l_situation_capability};
                                set_piece<NB_PIECES>(l_next_situation, l_new_situation_capability, l_x, l_y, l_piece_index, l_orientation, l_new_situation_capability, p_transition_manager);

                                situation_profile l_profile = l_new_situation_capability.compute_profile(l_next_situation.get_level());
                                if(l_profile.is_valid() && l_next_situations.end() == l_next_situations.find(l_next_situation))
                                {
                                    l_next_situations.insert(l_next_situation);
                                }
                                l_word &= ~(1u << l_bit_index);
                            }
                        }
                    }
                }
                l_current_situations.erase(l_current_situations.begin());
            }
            if(l_next_situations.size() > 10000000)
            {
                l_nb_level = 0;
            }
        }
        if(l_level + 1 < NB_PIECES)
        {
            std::set<emp_situation> & l_list = l_situations[(l_level + 1) % 2];
#ifdef VERBOSE
            unsigned int l_size = l_list.size();
#endif // VERBOSE
            while(!l_list.empty())
            {
#ifdef VERBOSE
                std::cout << "\r" << std::string(2 * (p_start_level + 1) , '-') << " " << (100 * l_list.size()) / l_size << "%   ";
                std::cout.flush();
#endif // VERBOSE
                situation_capability<2 * NB_PIECES> l_situation_capability = compute_situation_capability(*l_list.begin(), p_transition_manager, p_ref_situation_capability, p_situation);
                widest<NB_PIECES>(*l_list.begin(), p_transition_manager, l_level + 1, p_nb_level, l_situation_capability, p_solutions);
//                widest<NB_PIECES>(*l_list.begin(), p_transition_manager, l_level + 1, p_nb_level, p_ref_situation_capability, p_solutions);
                l_list.erase(l_list.begin());
            }
        }
        else
        {
#ifdef VERBOSE
            std::cout << "\rLevel " << NB_PIECES << " : " << l_situations[l_nb_level % 2].size() << std::endl;
#endif // VERBOSE
            for (const auto & l_iter: l_situations[l_nb_level % 2])
            {
                p_solutions.insert(l_iter);
            }
        }
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    feature_profile::wide(std::set<emp_situation> & p_solutions
                         ,profile_stats<NB_PIECES> & p_profile_stats
                         ,profile_stats<NB_PIECES> & p_profile_stats2
                         ,const std::set<emp_situation> & p_initial_situations
                         ,const transition_manager<NB_PIECES> & p_transition_manager
                         ,const situation_capability<2 * NB_PIECES> & p_ref_situation_capability
                         ,unsigned int p_level
                         )
    {
        std::set<emp_situation> l_situations[2] = {p_initial_situations};

        for(unsigned int l_level = 0; l_level < p_level; ++l_level)
        {
            std::set<emp_situation> & l_current_situations = l_situations[l_level % 2];
            std::set<emp_situation> & l_next_situations = l_situations[(1 + l_level) % 2];

            std::cout << std::endl << "\rLevel " << l_level << " : " << l_current_situations.size() << " situations " << std::endl;
            unsigned int l_total_nb = l_current_situations.size();
            unsigned int l_percent = 101;
            while(!l_current_situations.empty())
            {
                {
                    unsigned int l_new_percent = (100 * l_current_situations.size()) / l_total_nb;
                    if(l_new_percent != l_percent)
                    {
                        l_percent = l_new_percent;
                        std::cout << "\r" << l_percent << "%    ";
                        std::cout.flush();
                    }

                }
                const emp_situation & l_current_situation = *l_current_situations.begin();
                // Rebuild situation capability
                situation_capability<2 * NB_PIECES> l_situation_capability{p_ref_situation_capability};
                unsigned int l_nb_pieces = 0;
                for(unsigned int l_x = 0; l_x < l_current_situation.get_info().get_width() && l_nb_pieces < l_current_situation.get_level(); ++l_x)
                {
                    for(unsigned int l_y = 0; l_y < l_current_situation.get_info().get_height() && l_nb_pieces < l_current_situation.get_level(); ++l_y)
                    {
                        if(l_current_situation.contains_piece(l_x, l_y))
                        {
                            const emp_types::t_oriented_piece & l_piece = l_current_situation.get_piece(l_x, l_y);
                            unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id(l_x
                                    ,l_y
                                    ,l_piece.first - 1
                                    ,l_piece.second
                                    ,l_current_situation.get_info()
                                                                                                              );
                            l_situation_capability.apply_and(l_situation_capability, p_transition_manager.get_transition(l_raw_variable_id));
                            ++l_nb_pieces;
                        }
                    }
                }

                // Search for transitions
                for(unsigned int l_position_index = 0; l_position_index < NB_PIECES; ++l_position_index)
                {
                    piece_position_info & l_piece_position_info{l_situation_capability.get_capability(l_position_index)};
                    if(l_piece_position_info.any_bit_set())
                    {
                        for (unsigned int l_word_index = 0; l_word_index < 32; ++l_word_index)
                        {
                            uint32_t l_word = l_piece_position_info.get_word(l_word_index);
                            while (l_word)
                            {
                                unsigned int l_bit_index = ffs(l_word) - 1;
                                unsigned int l_full_index = 32 * l_word_index + l_bit_index;
                                unsigned int l_x = l_position_index % get_info().get_width();
                                unsigned int l_y = l_position_index / get_info().get_width();
                                unsigned int l_piece_index = l_full_index % 256;
                                auto l_orientation = static_cast<emp_types::t_orientation>(l_full_index / 256);
                                emp_situation l_next_situation{l_current_situation};
                                situation_capability<2 * NB_PIECES> l_new_situation_capability{l_situation_capability};
                                set_piece<NB_PIECES>(l_next_situation, l_new_situation_capability, l_x, l_y, l_piece_index, l_orientation, l_new_situation_capability, p_transition_manager);

                                situation_profile l_profile = l_new_situation_capability.compute_profile(l_next_situation.get_level());
                                if(l_profile.is_valid() && l_next_situations.end() == l_next_situations.find(l_next_situation))
                                {
                                    l_next_situations.insert(l_next_situation);
                                    if(p_solutions.empty() ||
                                       NB_PIECES == l_next_situation.get_level() ||
                                       std::any_of(p_solutions.begin()
                                                  ,p_solutions.end()
                                                  ,[&](const emp_situation & p_solution)
                                                   {return l_next_situation.is_predecessor(p_solution);}
                                                  )
                                      )
                                    {
                                        p_profile_stats.update(l_profile);
                                    }
                                    else
                                    {
                                        p_profile_stats2.update(l_profile);
                                    }
                                }
                                l_word &= ~(1u << l_bit_index);
                            }
                        }
                    }
                }
                l_current_situations.erase(l_current_situations.begin());
            }
        }
        std::cout << "\rLevel " << p_level << " : " << l_situations[p_level % 2].size() << std::endl;
        if(p_solutions.empty())
        {
            p_solutions = l_situations[p_level % 2];
        }
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    feature_profile::template_run()
    {
        situation_capability<2 * NB_PIECES> l_ref_situation_capability;
        std::map<unsigned int, unsigned int> l_variable_translator;
        std::unique_ptr<const transition_manager<NB_PIECES>> l_transition_manager{prepare_run<NB_PIECES>(l_ref_situation_capability, l_variable_translator)};

        emp_situation l_situation;
        std::set<emp_situation> l_solutions;

        std::vector<std::pair<emp_situation,situation_capability<2 * NB_PIECES>>> l_to_treat;
        unsigned int l_nb_level = NB_PIECES;
        if(NB_PIECES != 25)
        {
            l_to_treat.push_back(std::make_pair(l_situation, l_ref_situation_capability));
        }
        else
        {
            l_nb_level = NB_PIECES - 4;
            auto l_set_corner = [&](emp_situation & p_situation
                                  ,situation_capability<2 * NB_PIECES> & p_capability
                                  ,unsigned int p_corner_id
                                  ,unsigned int p_x
                                  ,unsigned int p_y
                                  )
            {
                for (auto l_orientation_index = static_cast<unsigned int>(emp_types::t_orientation::NORTH);
                     l_orientation_index <= static_cast<unsigned int>(emp_types::t_orientation::WEST);
                     ++l_orientation_index
                    )
                {
                    if (p_capability.get_capability(p_situation.get_info().get_position_index(p_x, p_y)).is_bit(p_corner_id - 1, static_cast<emp_types::t_orientation>(l_orientation_index)))
                    {
                        set_piece(p_situation
                                 ,p_capability
                                 ,p_x
                                 ,p_y
                                 ,p_corner_id - 1
                                 ,static_cast<emp_types::t_orientation>(l_orientation_index)
                                 ,p_capability
                                 ,*l_transition_manager
                                 );
                        break;
                    }
                }

            };

            emp_situation l_corner_situation;
            situation_capability<2 * NB_PIECES> l_corner_capability{l_ref_situation_capability};
            auto l_corner_id = m_piece_db.get_corner(0).get_id();

            l_set_corner(l_corner_situation, l_corner_capability, l_corner_id, 0, 0);

            std::vector<unsigned int> l_corner_positions{l_situation.get_info().get_position_index(l_situation.get_info().get_width() - 1, 0)
                                                        ,l_situation.get_info().get_position_index(0 , l_situation.get_info().get_height() - 1)
                                                        ,l_situation.get_info().get_position_index(l_situation.get_info().get_width() - 1,l_situation.get_info().get_height() - 1)
                                                        };
            std::vector<unsigned int> l_other_corners{1,2,3};

            do
            {
                emp_situation l_all_corner_situation{l_corner_situation};
                situation_capability<2 * NB_PIECES> l_all_corner_capability{l_corner_capability};
                unsigned int l_pos_index = 0;
                for (auto l_iter: l_other_corners)
                {
                    l_set_corner(l_all_corner_situation
                                ,l_all_corner_capability
                                ,m_piece_db.get_corner(l_iter).get_id()
                                ,l_all_corner_situation.get_info().get_x(l_corner_positions[l_pos_index])
                                ,l_all_corner_situation.get_info().get_y(l_corner_positions[l_pos_index])
                                );
                    ++l_pos_index;
                }
                std::cout << situation_string_formatter<emp_situation>::to_string(l_all_corner_situation) << std::endl;
                l_to_treat.push_back(std::make_pair(l_all_corner_situation, l_all_corner_capability));
            }
            while (next_permutation(l_other_corners.begin(),l_other_corners.end()));
        }
        profile_stats<NB_PIECES> l_all_stats;
        l_all_stats.update(l_ref_situation_capability.compute_profile(0));

        for(const auto & l_iter: l_to_treat)
        {
            wide<NB_PIECES>(l_solutions, l_all_stats, l_all_stats, {l_iter.first}, *l_transition_manager, l_iter.second, l_nb_level);
        }

        std::set<std::string> l_solutions_string;
        for(const auto & l_iter: l_solutions)
        {
            l_solutions_string.insert(situation_string_formatter<emp_situation>::to_string(l_iter));
        }
        dump_solutions(l_solutions_string);
        profile_stats<NB_PIECES> l_solution_stats;
        profile_stats<NB_PIECES> l_non_solution_stats;
        l_solution_stats.update(l_ref_situation_capability.compute_profile(0));
        l_non_solution_stats.update(l_ref_situation_capability.compute_profile(0));

        for(const auto & l_iter: l_to_treat)
        {
            bool l_contains_solution = false;
            for(const auto & l_solution: l_solutions)
            {
                if(l_iter.first.is_predecessor(l_solution))
                {
                    l_contains_solution = true;
                    break;
                }
            }
            if(l_contains_solution)
            {
                wide<NB_PIECES>(l_solutions, l_solution_stats, l_non_solution_stats, {l_iter.first}, *l_transition_manager, l_iter.second, l_nb_level);
            }
        }
        VTK_histogram_dumper l_low_min_dumper("profile_low_min_vtk.txt"
                                             ,"Profile_Low_min"
                                             , "Level"
                                             , "Transition_number"
                                             , NB_PIECES + 1
                                             , 3
                                             , {"General", "Solution", "NoSolution"}
                                             );

        l_low_min_dumper.dump_serie(l_all_stats.get_low_stats_min());
        l_low_min_dumper.dump_serie(l_solution_stats.get_low_stats_min());
        l_low_min_dumper.dump_serie(l_non_solution_stats.get_low_stats_min());

        VTK_histogram_dumper l_low_max_dumper("profile_low_max_vtk.txt"
                                             ,"Profile_Low_max"
                                             , "Level"
                                             , "Transition_number"
                                             , NB_PIECES + 1
                                             , 3
                                             , {"General", "Solution", "NoSolution"}
                                             );

        l_low_max_dumper.dump_serie(l_all_stats.get_low_stats_max());
        l_low_max_dumper.dump_serie(l_solution_stats.get_low_stats_max());
        l_low_max_dumper.dump_serie(l_non_solution_stats.get_low_stats_max());

        VTK_histogram_dumper l_high_min_dumper("profile_high_min_vtk.txt"
                                              ,"Profile_High_min"
                                              , "Level"
                                              , "Transition_number"
                                              , NB_PIECES + 1
                                              , 3
                                              , {"General", "Solution", "NoSolution"}
                                             );
        l_high_min_dumper.dump_serie(l_all_stats.get_high_stats_min());
        l_high_min_dumper.dump_serie(l_solution_stats.get_high_stats_min());
        l_high_min_dumper.dump_serie(l_non_solution_stats.get_high_stats_min());

        VTK_histogram_dumper l_high_max_dumper("profile_high_max_vtk.txt"
                                              ,"Profile_High_max"
                                              , "Level"
                                              , "Transition_number"
                                              , NB_PIECES + 1
                                              , 3
                                              , {"General", "Solution", "NoSolution"}
                                              );
        l_high_max_dumper.dump_serie(l_all_stats.get_high_stats_max());
        l_high_max_dumper.dump_serie(l_solution_stats.get_high_stats_max());
        l_high_max_dumper.dump_serie(l_non_solution_stats.get_high_stats_max());

        VTK_histogram_dumper l_total_min_dumper("profile_total_min_vtk.txt"
                                               ,"Profile_Total_Min"
                                               , "Level"
                                               , "Transition_number"
                                               , NB_PIECES + 1
                                               , 3
                                               , {"General", "Solution", "NoSolution"}
                                               );

        l_total_min_dumper.dump_serie(l_all_stats.get_total_stats_min());
        l_total_min_dumper.dump_serie(l_solution_stats.get_total_stats_min());
        l_total_min_dumper.dump_serie(l_non_solution_stats.get_total_stats_min());

        VTK_histogram_dumper l_total_max_dumper("profile_total_max_vtk.txt"
                                               ,"Profile_Total_Max"
                                               , "Level"
                                               , "Transition_number"
                                               , NB_PIECES + 1
                                               , 3
                                               , {"General", "Solution", "NoSolution"}
                                               );

        l_total_max_dumper.dump_serie(l_all_stats.get_total_stats_max());
        l_total_max_dumper.dump_serie(l_solution_stats.get_total_stats_max());
        l_total_max_dumper.dump_serie(l_non_solution_stats.get_total_stats_max());

#if 0
        // Test algorithm to spli the wide search
        widest(l_situation, *l_transition_manager, 0, 13, l_ref_situation_capability, l_solutions);
        std::set<std::string> l_solutions_string;
        for(const auto & l_iter: l_solutions)
        {
            l_solutions_string.insert(situation_string_formatter<emp_situation>::to_string(l_iter));
        }

        dump_solutions(l_solutions_string);
        return;
#endif // 0

#if 0
        std::set<emp_situation> l_situations[2];

        l_situation.set_piece(0,0,emp_types::t_oriented_piece(1,emp_types::t_orientation::NORTH));
        unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id(0
                ,0
                ,1 - 1
                ,emp_types::t_orientation::NORTH
                ,l_situation.get_info()
                                                                                          );
        l_ref_situation_capability.apply_and(l_ref_situation_capability, l_transition_manager->get_transition(l_raw_variable_id));

        l_situations[1].insert(l_situation);
#endif // 0
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    [[maybe_unused]]
    void
    feature_profile::deepest()
    {
        typedef std::tuple<emp_situation, situation_capability<2 * NB_PIECES>> t_stack_element;
        std::array<t_stack_element, NB_PIECES + 1> l_stack;
        std::map<unsigned int, unsigned int> l_variable_translator;
        std::unique_ptr<const transition_manager<NB_PIECES>> l_transition_manager{prepare_run<NB_PIECES>(std::get<1>(l_stack[0]), l_variable_translator)};

        std::set<emp_situation> l_situations;
        std::set<std::string> l_solutions_string;

        std::function<bool(unsigned int)> l_treat_element = [&](unsigned int p_level)
        {
            t_stack_element & l_element = l_stack[p_level];
            situation_capability<2 * NB_PIECES> & l_situation_capability = std::get<1>(l_element);
            emp_situation & l_situation = std::get<0>(l_element);
            situation_profile l_profile = l_situation_capability.compute_profile(p_level);
            if(!l_profile.is_valid())
            {
                return false;
            }
            if(l_situations.end() != l_situations.find(l_situation))
            {
                return false;
            }
            l_situations.insert(l_situation);
            if(p_level == NB_PIECES)
            {
                std::string l_solution_string{situation_string_formatter<emp_situation>::to_string(l_situation)};
                std::cout << "Level=" << p_level << R"( ")" << l_solution_string << R"(" Min=)" << l_profile.get_min() << " Max=" << l_profile.get_max() << " Total=" << l_profile.compute_total() << std::endl;
                l_solutions_string.insert(l_solution_string);
            }
            for(unsigned int l_position_index = 0; l_position_index < NB_PIECES; ++l_position_index)
            {
                piece_position_info & l_piece_position_info{l_situation_capability.get_capability(l_position_index)};
                if(l_piece_position_info.any_bit_set())
                {
                    for (unsigned int l_word_index = 0; l_word_index < 32; ++l_word_index)
                    {
                        uint32_t l_word = l_piece_position_info.get_word(l_word_index);
                        while (l_word)
                        {
                            unsigned int l_bit_index = ffs(l_word) - 1;
                            unsigned int l_full_index = 32 * l_word_index + l_bit_index;
                            unsigned int l_x = l_position_index % get_info().get_width();
                            unsigned int l_y = l_position_index / get_info().get_width();
                            unsigned int l_piece_index = l_full_index % 256;
                            auto l_orientation = static_cast<emp_types::t_orientation>(l_full_index / 256);
                            t_stack_element & l_next_element = l_stack[p_level + 1];
                            std::get<0>(l_next_element) = std::get<0>(l_element);
                            set_piece<NB_PIECES>(std::get<0>(l_next_element), std::get<1>(l_element), l_x, l_y, l_piece_index, l_orientation, std::get<1>(l_next_element), *l_transition_manager);
                            l_treat_element(p_level + 1);
                            l_word &= ~(1u << l_bit_index);
                        }
                    }
                }
            }
            return false;
        };

        l_treat_element(0);
        dump_solutions(l_solutions_string);

    }

    //-------------------------------------------------------------------------
    void
    feature_profile::dump_solutions(const std::set<std::string> & p_solutions)
    {
        std::ofstream l_solutions_dump;
        l_solutions_dump.open("solutions.txt");
        for(const auto & l_iter:p_solutions)
        {
            std::cout << l_iter << std::endl;
            l_solutions_dump << l_iter << std::endl;
        }

        l_solutions_dump.close();
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    feature_profile::set_piece(emp_situation & p_situation
                              ,const situation_capability<2 * NB_PIECES> & p_situation_capability
                              ,unsigned int p_x
                              ,unsigned int p_y
                              ,unsigned int p_piece_index
                              ,emp_types::t_orientation p_orientation
                              ,situation_capability<2 * NB_PIECES> & p_result_capability
                              ,const transition_manager<NB_PIECES> & p_transition_manager
                              )
    {
        unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id(p_x
                                                                                          ,p_y
                                                                                          ,p_piece_index
                                                                                          ,p_orientation
                                                                                          ,p_situation.get_info()
                                                                                          );
        p_result_capability.apply_and(p_situation_capability, p_transition_manager.get_transition(l_raw_variable_id));
        p_situation.set_piece(p_x, p_y, emp_types::t_oriented_piece(p_piece_index + 1, p_orientation));
    }

}
#endif //EDGE_MATCHING_PUZZLE_FEATURE_PROFILE_H
// EOF
