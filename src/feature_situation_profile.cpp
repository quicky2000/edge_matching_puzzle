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

#include "feature_situation_profile.h"

namespace edge_matching_puzzle
{
    //-------------------------------------------------------------------------
    void feature_situation_profile::run()
    {
        switch(m_info.get_height() * m_info.get_width())
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
                throw quicky_exception::quicky_logic_exception("Unsupported size " + std::to_string(m_info.get_width()) + "x" + std::to_string(m_info.get_height()), __LINE__, __FILE__);
        }
    }

    //-------------------------------------------------------------------------
    feature_situation_profile::~feature_situation_profile()
    {
        m_vtk_surface_file.close();
        delete m_vtk_line_plot_dumper;
    }

    //-------------------------------------------------------------------------
    std::string
    feature_situation_profile::get_file_name(const std::string & p_name) const
    {
        std::string l_root_file_name = std::to_string(m_info.get_width()) + "_" + std::to_string(m_info.get_height());
        return l_root_file_name + "_" + p_name + ".txt";
    }
}
// EOF

