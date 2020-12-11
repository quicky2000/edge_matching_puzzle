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
#include <fstream>

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

        std::ofstream m_vtk_line_plot_file;
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

        l_file_name = l_root_file_name + "line_plot_vtk.txt";
        m_vtk_line_plot_file.open(l_file_name.c_str());
        if(!m_vtk_line_plot_file.is_open())
        {
            throw quicky_exception::quicky_logic_exception("Unable to create VTK line_plot file", __LINE__, __FILE__);
        }

        m_vtk_line_plot_file << "line_plot" << std::endl;
        m_vtk_line_plot_file << "Profile_evolution Step Possibilites " << 2 * m_info.get_height() * m_info.get_width() << " " << m_initial_situation.get_level() + 1 << std::endl;
        for(unsigned int l_index = 0; l_index <= m_initial_situation.get_level() ; ++l_index)
        {
            m_vtk_line_plot_file << "Level_" << std::to_string(l_index) << " ";
        }
        m_vtk_line_plot_file << std::endl;
    }

    //-------------------------------------------------------------------------
    template<unsigned int NB_PIECES>
    void feature_situation_profile::template_run()
    {
        situation_capability<2 * NB_PIECES> l_situation;

        system_equation_for_CUDA::prepare_initial<NB_PIECES, situation_capability<2 * NB_PIECES>>(m_variable_generator, m_strategy_generator, l_situation);
        std::map<unsigned int, unsigned int> l_variable_translator;
        const transition_manager<NB_PIECES> * l_transition_manager = system_equation_for_CUDA::prepare_transitions<NB_PIECES, situation_capability<2 * NB_PIECES>, transition_manager<NB_PIECES>>(m_info, m_variable_generator, m_strategy_generator, l_variable_translator);

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
        for(auto l_iter: p_situation_capability.compute_profile())
        {
            std::cout << std::setw(3) << l_iter << " ";
            m_vtk_surface_file << l_iter << " ";
            m_vtk_line_plot_file << l_iter + NB_PIECES - p_level << " ";
        }
        m_vtk_surface_file << std::endl;
        m_vtk_line_plot_file << std::endl;
        std::cout << std::endl;
    }

}
#endif //EDGE_MATCHING_PUZZLE_FEATURE_SITUATION_PROFILE_H
