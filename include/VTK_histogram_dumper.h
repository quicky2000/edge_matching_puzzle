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

#ifndef EDGE_MATCHING_PUZZLE_VTK_HISTOGRAM_DUMPER_H
#define EDGE_MATCHING_PUZZLE_VTK_HISTOGRAM_DUMPER_H

#include "VTK_table_based_dumper.h"

namespace edge_matching_puzzle
{
    class VTK_histogram_dumper: public VTK_table_based_dumper
    {
      public:

        inline
        VTK_histogram_dumper(const std::string & p_file_name
                            ,const std::string & p_title
                            ,const std::string & p_X_name
                            ,const std::string & p_Y_name
                            ,unsigned int p_serie_dim
                            ,unsigned int p_nb_series
                            ,const std::vector<std::string> & p_series_name
                            );

      private:

    };

    //-------------------------------------------------------------------------
    VTK_histogram_dumper::VTK_histogram_dumper(const std::string & p_file_name
                                              ,const std::string & p_title
                                              ,const std::string & p_X_name
                                              ,const std::string & p_Y_name
                                              ,unsigned int p_serie_dim
                                              ,unsigned int p_nb_series
                                              ,const std::vector<std::string> & p_series_name
                                              )
    :VTK_table_based_dumper("histogram", p_file_name, p_title, p_X_name, p_Y_name, p_serie_dim, p_nb_series,p_series_name)
    {
    }
}
#endif //EDGE_MATCHING_PUZZLE_VTK_HISTOGRAM_DUMPER_H
// EOF