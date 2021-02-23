/*
      This file is part of edge_matching_puzzle
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

#ifndef EDGE_MATCHING_PUZZLE_VTK_TABLE_BASED_DUMPER_H
#define EDGE_MATCHING_PUZZLE_VTK_TABLE_BASED_DUMPER_H

#include <quicky_exception.h>
#include <fstream>
#include <vector>

namespace edge_matching_puzzle
{
    class VTK_table_based_dumper
    {

      public:

        inline
        void dump_serie(const std::vector<unsigned int> & p_serie);

        inline
        void dump_value(unsigned int p_value);

        inline
        void close_serie();

      protected:

        inline
        VTK_table_based_dumper(const std::string & p_type
                              ,const std::string & p_file_name
                              ,const std::string & p_tile
                              ,const std::string & p_X_name
                              ,const std::string & p_Y_name
                              ,unsigned int p_serie_dim
                              ,unsigned int p_nb_series
                              ,const std::vector<std::string> & p_series_name
                              );

        inline
        ~VTK_table_based_dumper();

      private:

        std::ofstream m_vtk_table_based_file;

        /**
         * Number of values in a serie
         */
        unsigned int m_serie_dim;

        /**
         * Number of declared series
         */
        unsigned int m_nb_series;

        std::vector<std::string> m_series_name;

        /**
         * Number of dumped series
         */
        unsigned int m_serie_counter;

        /**
         * Number of value dumped in current serie
         */
        unsigned int m_serie_value_counter;

    };

    //-------------------------------------------------------------------------
    VTK_table_based_dumper::VTK_table_based_dumper(const std::string & p_type
                                                  ,const std::string & p_file_name
                                                  ,const std::string & p_tile
                                                  ,const std::string & p_X_name
                                                  ,const std::string & p_Y_name
                                                  ,unsigned int p_serie_dim
                                                  ,unsigned int p_nb_series
                                                  ,const std::vector<std::string> & p_series_name
                                                  )
    :m_serie_dim(p_serie_dim)
    ,m_nb_series(p_nb_series)
    ,m_series_name(p_series_name)
    ,m_serie_counter(0)
    ,m_serie_value_counter(0)
    {
        m_vtk_table_based_file.open(p_file_name.c_str());
        if(!m_vtk_table_based_file.is_open())
        {
            throw quicky_exception::quicky_logic_exception(R"(Unable to create VTK )" + p_type + R"( file ")" + p_file_name +R"(")", __LINE__, __FILE__);
        }
        m_vtk_table_based_file << p_type << std::endl;
        m_vtk_table_based_file << p_tile << " " << p_X_name << " " << p_Y_name << " " << p_serie_dim << " " << p_nb_series << std::endl;

         if (p_series_name.size() != p_nb_series)
         {
             throw quicky_exception::quicky_logic_exception("Number of serie name (" + std::to_string(p_series_name.size()) +") not coherent with serie number(" + std::to_string(p_nb_series) + ")"
                                                           ,__LINE__
                                                           ,__FILE__
                                                           );
         }

         for (auto l_iter = p_series_name.begin(); l_iter != p_series_name.end() - 1; ++l_iter)
         {
             m_vtk_table_based_file << *l_iter << " ";
         }
         m_vtk_table_based_file << *p_series_name.rbegin() << std::endl;
    }

    //-------------------------------------------------------------------------
    VTK_table_based_dumper::~VTK_table_based_dumper()
    {
        if(m_vtk_table_based_file.is_open())
        {
            m_vtk_table_based_file.close();
        }
    }

    //-------------------------------------------------------------------------
    void
    VTK_table_based_dumper::dump_serie(const std::vector<unsigned int> & p_serie)
    {
        if(p_serie.size() != m_serie_dim)
        {
            throw quicky_exception::quicky_logic_exception("Number of serie values (" + std::to_string(p_serie.size()) + ") not coherent with serie number(" + std::to_string(m_serie_dim) +")", __LINE__, __FILE__);
        }
        if(m_serie_counter >= m_nb_series)
        {
            throw quicky_exception::quicky_logic_exception("Too many series", __LINE__, __FILE__);
        }
        for(auto l_iter: p_serie)
        {
            m_vtk_table_based_file << l_iter << " ";
        }
        m_vtk_table_based_file << std::endl;
        ++m_serie_counter;
    }

    //-------------------------------------------------------------------------
    void
    VTK_table_based_dumper::dump_value(unsigned int p_value)
    {
        if(m_serie_value_counter >= m_serie_dim)
        {
            assert(m_serie_counter < m_series_name.size());
            throw quicky_exception::quicky_logic_exception(R"(Too many values in serie ")" + m_series_name[m_serie_counter] + R"(")", __LINE__ ,__FILE__);
        }
        m_vtk_table_based_file << p_value << " ";
        ++m_serie_value_counter;
    }

    //-------------------------------------------------------------------------
    void
    VTK_table_based_dumper::close_serie()
    {
        if(m_serie_value_counter != m_serie_dim)
        {
            assert(m_serie_counter < m_series_name.size());
            throw quicky_exception::quicky_logic_exception("(Not enough values (" + std::to_string(m_serie_value_counter) + R"() in serie ")" + m_series_name[m_serie_counter] + R"(")", __LINE__ ,__FILE__);
        }
        m_vtk_table_based_file << std::endl;
        ++m_serie_counter;
        m_serie_value_counter = 0;
    }
}
#endif //EDGE_MATCHING_PUZZLE_VTK_TABLE_BASED_DUMPER_H
// EOF