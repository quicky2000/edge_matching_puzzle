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

#ifndef EDGE_MATCHING_PUZZLE_FEATURE_SITUATION_PROFILE_H
#define EDGE_MATCHING_PUZZLE_FEATURE_SITUATION_PROFILE_H

#include "feature_if.h"
#include "situation_capability.h"
#include "transition_manager.h"
#include "system_equation_for_CUDA.h"
#include "VTK_line_plot_dumper.h"
#include <fstream>
#include <functional>

namespace edge_matching_puzzle
{
    class feature_situation_profile: public feature_if
    {
      public:

        inline
        feature_situation_profile( const emp_piece_db & p_piece_db
                                 , const emp_FSM_info & p_info
                                 , std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                 , const std::string & p_initial_situation
                                 );

        void run() override;

        ~feature_situation_profile() override;

      private:

        template<unsigned int NB_PIECES>
        void treat_situation_capability( situation_capability<2 * NB_PIECES> & p_situation_capability
                                       , unsigned int p_level
                                       );

        template<unsigned int NB_PIECES>
        void template_run();

        /**
         * Method computing situation profiling by following a glutton algo-
         * -rithm saeching for min or max of a defined criteria
         * @tparam NB_PIECES Piece number of situation
         * @param p_situation Initial situation capability
         * @param p_transition_manager Transition manager
         * @param p_max indicate if alogirthm search min or max
         * @param p_less_than comparison function
         * @param p_name name of criteria that is compared
         * @return computed profile with total number of bits
         */
        template<unsigned int NB_PIECES>
        std::vector<uint32_t>
        compute_glutton(const situation_capability<2 * NB_PIECES> & p_situation
                       ,const transition_manager<NB_PIECES> & p_transition_manager
                       ,bool p_max
                       ,std::function<bool(const situation_profile & p_a, const situation_profile & p_b)> & p_less_than
                       ,const std::string & p_name
                       );

        template<unsigned int NB_PIECES>
        void compute_glutton(const situation_capability<2 * NB_PIECES> & p_situation
                            ,const transition_manager<NB_PIECES> & p_transition_manager
                            );

        std::string get_file_name(const std::string & p_name) const;

        const emp_piece_db & m_piece_db;

        const emp_FSM_info & m_info;

        /**
         * Contains initial situation
         * Should be declared before variable generator to be fully built
         */
        emp_FSM_situation m_initial_situation;

        /**
         * Generate variables of equation system representing the puzzle
         */
        emp_variable_generator m_variable_generator;

        const emp_strategy_generator & m_strategy_generator;

        std::ofstream m_vtk_surface_file;

        VTK_line_plot_dumper * m_vtk_line_plot_dumper;
    };

    //-------------------------------------------------------------------------
    feature_situation_profile::feature_situation_profile( const emp_piece_db & p_piece_db
                                                        , const emp_FSM_info & p_info
                                                        , std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                                        , const std::string & p_initial_situation
                                                        )
    : m_piece_db(p_piece_db)
    , m_info(p_info)
    , m_variable_generator(p_piece_db, *p_strategy_generator, p_info, "", m_initial_situation)
    , m_strategy_generator(*p_strategy_generator)
    , m_vtk_line_plot_dumper(nullptr)
    {
        m_initial_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));
        if(p_initial_situation.empty())
        {
            throw quicky_exception::quicky_logic_exception("Initial situation should be provided", __LINE__, __FILE__);
        }
        m_initial_situation.set(p_initial_situation);

        std::string l_root_file_name = std::to_string(p_info.get_width()) + "_" + std::to_string(p_info.get_height()) + "_";

        std::string l_file_name = l_root_file_name + "surface_vtk.txt";
        m_vtk_surface_file.open(l_file_name.c_str());
        if(!m_vtk_surface_file.is_open())
        {
            throw quicky_exception::quicky_logic_exception("Unable to create VTK surface file", __LINE__, __FILE__);
        }
        m_vtk_surface_file << "surface" << std::endl;
        m_vtk_surface_file << 2 * m_info.get_height() * m_info.get_width() << " " << m_initial_situation.get_level() + 1 << std::endl;

        m_vtk_line_plot_dumper = new VTK_line_plot_dumper(l_root_file_name + "line_plot_vtk.txt"
                                                         ,"Profile_evolution"
                                                         ,"Step"
                                                         ,"Possibilities"
                                                         ,2 * m_info.get_height() * m_info.get_width()
                                                         ,m_initial_situation.get_level() + 1
                                                         ,[](const emp_FSM_situation & p_situation)
                                                          {
                                                            std::vector<std::string> l_names;
                                                            for(unsigned int l_index = 0; l_index <= p_situation.get_level() ; ++l_index)
                                                            {
                                                                l_names.emplace_back("Level_" + std::to_string(l_index));
                                                            }
                                                            return l_names;
                                                          }(m_initial_situation)
                                                         );
    }

    //-------------------------------------------------------------------------
    template<unsigned int NB_PIECES>
    void feature_situation_profile::template_run()
    {
        situation_capability<2 * NB_PIECES> l_situation;

        system_equation_for_CUDA::prepare_initial<NB_PIECES, situation_capability<2 * NB_PIECES>>(m_variable_generator, m_strategy_generator, l_situation);
        std::map<unsigned int, unsigned int> l_variable_translator;
        const transition_manager<NB_PIECES> * l_transition_manager = system_equation_for_CUDA::prepare_transitions<NB_PIECES, situation_capability<2 * NB_PIECES>, transition_manager<NB_PIECES>>(m_info, m_variable_generator, m_strategy_generator, l_variable_translator);

        compute_glutton<NB_PIECES>(l_situation, *l_transition_manager);

        for(unsigned int l_position_index = 0; l_position_index < NB_PIECES; ++l_position_index)
        {
            unsigned int l_x, l_y;
            std::tie(l_x, l_y) = m_strategy_generator.get_position(l_position_index);
            if(m_initial_situation.contains_piece(l_x, l_y))
            {

                treat_situation_capability<NB_PIECES>(l_situation, l_position_index);

                const emp_types::t_oriented_piece l_oriented_piece  = m_initial_situation.get_piece(l_x, l_y);
                unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id( l_x, l_y
                                                                                                  , l_oriented_piece.first - 1
                                                                                                  , l_oriented_piece.second
                                                                                                  , m_info
                                                                                                  );

                l_situation.apply_and(l_situation, l_transition_manager->get_transition(l_raw_variable_id));
            }
        }
        treat_situation_capability<NB_PIECES>(l_situation, NB_PIECES);
        delete l_transition_manager;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    feature_situation_profile::treat_situation_capability( situation_capability<2 * NB_PIECES> & p_situation_capability
                                                         , unsigned int p_level
                                                         )
    {
        situation_profile l_profile{p_situation_capability.compute_profile(p_level)};
        for(auto l_iter: l_profile.get_values())
        {
            std::cout << std::setw(3) << l_iter << " ";
            m_vtk_surface_file << l_iter << " ";
        }
        m_vtk_surface_file << std::endl;
        std::transform(l_profile.get_values().begin(),  l_profile.get_values().end(), l_profile.get_values().begin(), [=](unsigned int p_value){return p_value + NB_PIECES - p_level;});
        m_vtk_line_plot_dumper->dump_serie(l_profile.get_values());
        std::cout << std::endl;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    feature_situation_profile::compute_glutton(const situation_capability<2 * NB_PIECES> & p_situation_capability
                                              ,const transition_manager<NB_PIECES> & p_transition_manager
                                              )
    {
        typedef std::function<bool(const situation_profile & p_a, const situation_profile & p_b)> t_comparator;
        std::vector<std::pair<std::string, t_comparator>> l_criterias =
        {
         {"total", [](const situation_profile & p_a, const situation_profile & p_b) -> bool { return p_a.less_than_total(p_b);}}
        };
        std::vector<std::string> l_serie_names;
        std::for_each(l_criterias.begin()
                     ,l_criterias.end()
                     ,[&](const std::pair<std::string, t_comparator> & p_pair)
                      {
                        std::string l_name{std::string(1, std::toupper(p_pair.first[0])) + p_pair.first.substr(1)};
                        l_serie_names.emplace_back(l_name + "_min");
                        l_serie_names.emplace_back(l_name + "_max");
                      }
                     );
        VTK_line_plot_dumper l_plot_dumper{get_file_name("level_glutton")
                                          ,"Level_glutton"
                                          ,"Step", "Total"
                                          ,m_info.get_height() * m_info.get_width() + 1
                                          ,static_cast<unsigned int>(l_serie_names.size())
                                          ,l_serie_names
                                          };

        for(auto l_iter:l_criterias)
        {
            l_plot_dumper.dump_serie(compute_glutton(p_situation_capability, p_transition_manager, false, l_iter.second, l_iter.first));
            l_plot_dumper.dump_serie(compute_glutton(p_situation_capability, p_transition_manager, true, l_iter.second, l_iter.first));
        }
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::vector<uint32_t>
    feature_situation_profile::compute_glutton(const situation_capability<2 * NB_PIECES> & p_situation_capability
                                              ,const transition_manager<NB_PIECES> & p_transition_manager
                                              ,bool p_max
                                              ,std::function<bool(const situation_profile & p_a, const situation_profile & p_b)> & p_less_than
                                              ,const std::string & p_name
                                              )
    {
        unsigned int l_serie_dim = m_info.get_height() * m_info.get_width() + 1;
        std::string l_title = std::string("Level_") + p_name + "_" + (p_max ? "max" : "min") + "_glutton";
        VTK_line_plot_dumper l_plot_dumper{get_file_name(l_title)
                                          ,l_title
                                          ,"Step" , std::string(1, std::toupper(p_name[0])) + p_name.substr(1)
                                          ,l_serie_dim
                                          ,1
                                          ,{p_max ? "Max" : "Min"}
                                          };
        emp_FSM_situation l_situation_min;
        l_situation_min.set_context(*(new emp_FSM_context(m_info.get_width() * m_info.get_height())));

        situation_capability<2 * NB_PIECES> l_situation_capability_min{p_situation_capability};
        situation_profile l_extrema{0, 2 * NB_PIECES, p_max ? std::numeric_limits<uint32_t>::min() : std::numeric_limits<uint32_t>::max()};
        unsigned l_extrema_score;

        std::vector<uint32_t> l_profile;
        {
            situation_profile l_current_profile{p_situation_capability.compute_profile(0)};
            uint32_t l_score = l_current_profile.compute_total();
            l_profile.push_back(l_score);
            l_plot_dumper.dump_value(l_score);
        }
        for(unsigned int l_level = 0; l_level < m_initial_situation.get_level(); ++l_level)
        {
            unsigned int l_x_min;
            unsigned int l_y_min;
            emp_types::t_oriented_piece l_oriented_piece_min;
            situation_capability<2 * NB_PIECES> l_situation_capability_new_min{l_situation_capability_min};

            for(unsigned int l_position_index = 0; l_position_index < NB_PIECES; ++l_position_index)
            {
                unsigned int l_x, l_y;
                std::tie(l_x, l_y) = m_strategy_generator.get_position(l_position_index);
                if(m_initial_situation.contains_piece(l_x, l_y) && !l_situation_min.contains_piece(l_x, l_y))
                {
                    situation_capability<2 * NB_PIECES> l_situation_capability_new{l_situation_capability_min};
                    const emp_types::t_oriented_piece l_oriented_piece  = m_initial_situation.get_piece(l_x, l_y);
                    unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id( l_x, l_y
                                                                                                      , l_oriented_piece.first - 1
                                                                                                      , l_oriented_piece.second
                                                                                                      , m_info
                                                                                                      );
                    l_situation_capability_new.apply_and(l_situation_capability_min, p_transition_manager.get_transition(l_raw_variable_id));
                    situation_profile l_current_profile{l_situation_capability_new.compute_profile(l_level + 1)};
                    bool l_ok = l_current_profile.is_valid();
                    uint32_t l_score = l_ok ? l_current_profile.compute_total() : 0;
                    if(!l_ok || (p_max != p_less_than(l_current_profile,l_extrema)))
                    {
                        l_extrema = l_current_profile;
                        l_extrema_score = l_score;
                        l_situation_capability_new_min = l_situation_capability_new;
                        l_x_min = l_x;
                        l_y_min = l_y;
                        l_oriented_piece_min = l_oriented_piece;
                        if(!l_ok)
                        {
                            break;
                        }
                    }
                }
            }
            //std::cout << "Level[" << l_level << "] set " << l_oriented_piece_min.first << emp_types::orientation2short_string(l_oriented_piece_min.second) << "(" << l_x_min << "," << l_y_min << ") => " << l_extrema_score << std::endl;
            l_situation_capability_min = l_situation_capability_new_min;
            l_situation_min.set_piece(l_x_min, l_y_min, l_oriented_piece_min);
            l_plot_dumper.dump_value(l_extrema_score);
            l_profile.push_back(l_extrema_score);
            if(p_max)
            {
                l_extrema = situation_profile(l_level, 2 * NB_PIECES, 0);
            }
        }
        // Complete incomplete profile in case of non final situation
        while(l_profile.size() < l_serie_dim)
        {
            l_plot_dumper.dump_value(0);
            l_profile.push_back(0);
        }
        l_plot_dumper.close_serie();
        return l_profile;
    }

}
#endif //EDGE_MATCHING_PUZZLE_FEATURE_SITUATION_PROFILE_H
