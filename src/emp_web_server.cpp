/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2015  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#include "emp_web_server.h"
#include "emp_strategy.h"
#include <sstream>

namespace edge_matching_puzzle
{
  //----------------------------------------------------------------------------
  int emp_web_server::treat_request (struct MHD_Connection *p_connection,
                                     const char *p_url,
                                     const char *p_method,
                                     const char *p_http_version,
                                     const char *p_upload_data,
                                     size_t *p_upload_data_size,
                                     void **p_connection_ptr)
  {
    std::cout << "Method = \"" << p_method << "\"" <<std::endl ;
    std::cout << "URL = \"" << p_url << "\"" <<std::endl ;
    MHD_get_connection_values (p_connection, MHD_HEADER_KIND, print_out_key,NULL);

    if (0 != strcmp (p_method, "GET"))
      {
        return MHD_NO;              /* unexpected method */
      }
    if (&(m_connection_ptr) != *p_connection_ptr)
      {
        /* do never respond on first call */
        *p_connection_ptr = &(m_connection_ptr);
        return MHD_YES;
      }
    *p_connection_ptr = NULL; /* reset when done */

    struct MHD_Response * l_response = nullptr;
    if(!strcmp(p_url,"/"))
      {

        uint64_t l_nb_situations = 0;
        uint64_t l_nb_solutions = 0;
        unsigned int l_shift = 0;
#ifdef DEBUG_WEBSERVER
        std::cout << "Webserver asking strategy to pause" << std::endl ;
#endif
        m_strategy.pause();
#ifdef DEBUG_WEBSERVER
        std::cout << "Webserver wait for strategy to pause" << std::endl ;
#endif
        while(!m_strategy.is_paused())
          {
            usleep(1);
          }
        m_strategy.send_info(l_nb_situations,l_nb_solutions,l_shift,m_pieces,m_info);
        m_strategy.restart();
#ifdef DEBUG_WEBSERVER
        std::cout << "Webserver asking strategy to restart" << std::endl ;
#endif
    
        std::stringstream l_nb_situation_stream;
        l_nb_situation_stream << l_nb_situations;

        std::stringstream l_nb_solution_stream;
        l_nb_solution_stream << l_nb_solutions;

        std::string l_response_page_begin = "<html><head><title>Status</title></head><body><center><h1>Status</h1></center><br>\n<form name=\"Show\">\nNb situations<input type=\"text\" name=\"Nb_situations\" value=\"";
        std::string l_response_page = l_response_page_begin + l_nb_situation_stream.str() ;

        std::string l_inter = "\" size=\"20\"><br>\nNb solutions<input type=\"text\" name=\"Nb_solutions\" value=\"";
        l_response_page += l_inter;

        l_response_page += l_nb_solution_stream.str();

        std::string l_begin_table = "\" size=\"20\"><br>\n</form>\n\n\n<TABLE>\n<CAPTION>Current situation</CAPTION>\n";
        l_response_page += l_begin_table;

        for(unsigned int l_y = 0 ; l_y < m_info.get_height() ; ++l_y)
          {
            l_response_page += "<TR>";
            for(unsigned int l_x = 0 ; l_x < m_info.get_width() ; ++l_x)
              {
                emp_types::t_binary_piece l_binary_piece = m_pieces[m_info.get_width() * l_y + l_x];
                std::string l_piece_name = m_empty_file_name;
                if(l_binary_piece)
                  {
                    l_binary_piece = l_binary_piece >> l_shift;
                    std::stringstream l_stream_piece_id;
                    l_stream_piece_id << (1 + (l_binary_piece >> 2 ));
                    l_piece_name = l_stream_piece_id.str() + emp_types::orientation2short_string((emp_types::t_orientation)(l_binary_piece & 0x3));
                  }
                std::string l_cell_string = "<TD><IMG src=\"pieces/" + l_piece_name + ".bmp\"></TD>";
                l_response_page += l_cell_string;
              }
            l_response_page += "</TR>\n";
          }
        std::string l_end_table = "</TABLE>\n</body></html>";
        l_response_page += l_end_table;

        l_response = MHD_create_response_from_buffer (l_response_page.size(),
                                                      (void *)l_response_page.c_str() ,
                                                      MHD_RESPMEM_MUST_COPY);
      }
    else if(strlen(p_url) > m_picture_root.size() + 2 && !strncmp(p_url,(std::string("/")+m_picture_root).c_str(),m_picture_root.size()))
      {
        std::string l_picture_name = std::string(p_url).substr(m_picture_root.size() + 2);
        size_t l_pos = l_picture_name.find(".");
        assert(std::string::npos != l_pos);
        l_picture_name = l_picture_name.substr(0,l_pos);
 
        assert(l_picture_name.size() >= 2);
        std::string l_piece_id = l_picture_name.substr(0,l_picture_name.size() - 1);
        std::string l_orient_string = l_picture_name.substr(l_picture_name.size() - 1);
        unsigned int l_index = (atoi(l_piece_id.c_str()) - 1 )* 4 + (unsigned int) emp_types::short_string2orientation(l_orient_string[0]);
        l_response = MHD_create_response_from_buffer (m_picture_size,m_picture_data[l_index],MHD_RESPMEM_PERSISTENT);
        MHD_add_response_header(l_response,MHD_HTTP_HEADER_CONTENT_TYPE,"image/bmp");
      }
    else
      {
        std::string l_error_string = std::string("404 : ") + p_url + " not found";
        l_error_string = "<html><head>" + l_error_string + "</head><body>" + l_error_string + "</body></html>";
        l_response =
          MHD_create_response_from_buffer (l_error_string.size(),
                                           (void *)l_error_string.c_str(),
                                           MHD_RESPMEM_PERSISTENT);
        if (l_response)
          {
            int ret = MHD_queue_response (p_connection, MHD_HTTP_NOT_FOUND,l_response);
            MHD_destroy_response (l_response);
            return ret;
          }
        else
          {
            return MHD_NO;
          }
      }
    int ret = MHD_queue_response (p_connection, MHD_HTTP_OK, l_response);
    MHD_destroy_response (l_response);
    return ret;
  }

}
//EOF
