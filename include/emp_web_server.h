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

#ifndef EMP_WEB_SERVER_H
#define EMP_WEB_SERVER_H

#include "quicky_exception.h"
#include "emp_gui.h"
#include "emp_types.h"
#include "emp_FSM_info.h"
#include <unistd.h>
#include <limits.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <microhttpd.h>

#ifndef HOST_NAME_MAX
#if defined(__APPLE__)
#define HOST_NAME_MAX 255
#else
#define HOST_NAME_MAX 64
#endif // __APPLE__
#endif // HOST_NAME_MAX

namespace edge_matching_puzzle
{
    class emp_advanced_feature_base;

    class emp_web_server
    {
      public:
        inline
        emp_web_server(const unsigned int & p_port
                      ,emp_advanced_feature_base & p_strategy
                      ,const emp_gui & p_gui
                      ,const emp_FSM_info & p_FSM_info
                      );

        inline
        void start();

        inline
        ~emp_web_server();

      private:

#if MHD_VERSION >= 0x00097002
        typedef enum MHD_Result internal_result_t;
#else  // MHD_VERSION >= 0x00097002
        typedef int internal_result_t;
#endif // MHD_VERSION >= 0x00097002

        inline
        void dump_picture(const unsigned int & p_index
                         ,const emp_types::t_orientation & p_orientation
                         ,const lib_bmp::my_bmp & p_bmp
                         );

        internal_result_t treat_request (struct MHD_Connection * p_connection
                                        ,const char * p_url
                                        ,const char * p_method
                                        ,const char * p_http_version
                                        ,const char * p_upload_data
                                        ,size_t * p_upload_data_size
                                        ,void ** p_connection_ptr
                                        );

        // Microhttpd callbacks
        inline static
        internal_result_t s_treat_request(void *p_callback_data
                                         ,struct MHD_Connection *p_connection
                                         ,const char * p_url
                                         ,const char * p_method
                                         ,const char * p_http_version
                                         ,const char * p_upload_data
                                         ,size_t * p_upload_data_size
                                         ,void ** p_connection_ptr
                                         );

        inline static
        internal_result_t print_out_key (void * p_callback_data
                                        ,enum MHD_ValueKind p_kind
                                        ,const char * p_key
                                        ,const char * p_value
                                        );

        // End of Microhttpd callbacks

        std::string m_empty_file_name;
        std::string m_picture_root;
        emp_advanced_feature_base & m_strategy;
        const emp_FSM_info & m_info;
        emp_types::t_binary_piece * m_pieces;
        size_t m_picture_size;
        char ** m_picture_data;
        unsigned int m_port;
        struct MHD_Daemon * m_daemon;
        int m_connection_ptr;
    };

    //----------------------------------------------------------------------------
    emp_web_server::emp_web_server(const unsigned int & p_port
                                  ,emp_advanced_feature_base & p_strategy
                                  ,const emp_gui & p_gui
                                  ,const emp_FSM_info & p_FSM_info
                                  )
    : m_picture_root("pieces")
    , m_strategy(p_strategy)
    , m_info(p_FSM_info)
    , m_pieces(new emp_types::t_binary_piece[m_info.get_width() * m_info.get_height()])
    , m_picture_size(0)
    , m_picture_data(new char*[m_info.get_width() * m_info.get_height() * 4 + 1])
    , m_port(p_port)
    , m_daemon(nullptr)
    {
        std::stringstream l_empty_stream;
        l_empty_stream << (1 + m_info.get_width() * m_info.get_height()) << "N"; 
        m_empty_file_name = l_empty_stream.str();

        lib_bmp::my_bmp l_empty_picture(p_gui.get_picture(1).get_width(),p_gui.get_picture(1).get_height(),p_gui.get_picture(1).get_nb_bits_per_pixel());
        dump_picture(m_info.get_width() * m_info.get_height(),emp_types::t_orientation::NORTH,l_empty_picture);

        DIR * l_dir = opendir(m_picture_root.c_str());
        if(l_dir)
        {
            closedir(l_dir);
            std::vector<std::string> l_file_list;
            quicky_utils::quicky_files::list_content(m_picture_root.c_str(),l_file_list);
            for(auto l_iter:l_file_list)
            {
                remove((m_picture_root+"/"+l_iter).c_str());
            }
            rmdir(m_picture_root.c_str());
        }
        mode_t l_mode = S_IRWXU;
        mkdir(m_picture_root.c_str(),l_mode);

        for(unsigned int l_index = 0 ; l_index < m_info.get_width() * m_info.get_height() ; ++l_index)
        {
            const lib_bmp::my_bmp & l_picture = p_gui.get_picture(l_index + 1);
            dump_picture(l_index,emp_types::t_orientation::NORTH,l_picture);
            for(unsigned int l_orient_index = (unsigned int) emp_types::t_orientation::EAST ;
                l_orient_index <= (unsigned int)emp_types::t_orientation::WEST;
                ++l_orient_index
               )
            {
                lib_bmp::my_bmp l_oriented_picture(l_picture.get_width(),l_picture.get_height(),l_picture.get_nb_bits_per_pixel());
                for(unsigned int l_index_x = 0 ;
                    l_index_x < l_picture.get_width();
                    ++l_index_x
                   )
                {
                    for(unsigned int l_index_y = 0 ;
                        l_index_y < l_picture.get_height();
                        ++l_index_y
                       )
                    {
                        unsigned int l_oriented_x = 0;
                        unsigned int l_oriented_y = 0;
                        switch((emp_types::t_orientation)l_orient_index)
                         {
                            case emp_types::t_orientation::NORTH:
                                l_oriented_x = l_index_x;
                                l_oriented_y = l_index_y;
                                break;
                            case emp_types::t_orientation::EAST:
                                l_oriented_x = l_picture.get_height() - 1 - l_index_y;
                                l_oriented_y = l_index_x;
                                break;
                            case emp_types::t_orientation::SOUTH:
                                 l_oriented_x = l_picture.get_width() - 1 - l_index_x;
                                 l_oriented_y = l_picture.get_height() - 1 - l_index_y;
                                 break;
                            case emp_types::t_orientation::WEST:
                                l_oriented_x = l_index_y;
                                l_oriented_y = l_picture.get_width() - 1 - l_index_x;
                                break;
                            default:
                                throw quicky_exception::quicky_logic_exception("Bad orientation value when rotating pieces images",__LINE__,__FILE__);
                                break;
                         }
                         const lib_bmp::my_color_alpha & l_rgb_color = l_picture.get_pixel_color(l_oriented_x,l_oriented_y);
                        l_oriented_picture.set_pixel_color(l_index_x,l_index_y,l_rgb_color);
                    }
                }
                dump_picture(l_index,(emp_types::t_orientation)l_orient_index,l_oriented_picture);
            }
        }
    }

    //----------------------------------------------------------------------------
    void emp_web_server::dump_picture(const unsigned int & p_index
                                     ,const emp_types::t_orientation & p_orientation
                                     ,const lib_bmp::my_bmp & p_bmp
                                     )
    {
        std::stringstream l_piece_id_stream;
        l_piece_id_stream << (m_picture_root + "/") << (p_index + 1);

        std::string l_file_name = l_piece_id_stream.str() + emp_types::orientation2short_string(p_orientation)+".bmp";
        p_bmp.save(l_file_name);

        std::ifstream l_stream;
        l_stream.open(l_file_name.c_str());
        if(!l_stream.is_open())
        {
            throw quicky_exception::quicky_runtime_exception("Unable to open " + l_file_name + " file",__LINE__,__FILE__);
        }
        l_stream.seekg(0,l_stream.end);
        size_t l_size = l_stream.tellg();
        l_stream.seekg(0,l_stream.beg);
        if(m_picture_size && m_picture_size != l_size)
        {
            throw quicky_exception::quicky_logic_exception("Different image sizes",__LINE__,__FILE__);
        }
        else if(!m_picture_size)
        {
            m_picture_size = l_size;
        }
        unsigned int l_data_index = 4 * p_index + (unsigned int) p_orientation;
        m_picture_data[l_data_index] = new char[l_size];
        l_stream.read(m_picture_data[l_data_index],l_size);
        l_stream.close();
    }

    //----------------------------------------------------------------------------
    void emp_web_server::start()
    {
        m_daemon = MHD_start_daemon(// MHD_USE_SELECT_INTERNALLY | MHD_USE_DEBUG | MHD_USE_POLL,
                                    MHD_USE_SELECT_INTERNALLY | MHD_USE_DEBUG
                                   // MHD_USE_THREAD_PER_CONNECTION | MHD_USE_DEBUG | MHD_USE_POLL,
                                   // MHD_USE_THREAD_PER_CONNECTION | MHD_USE_DEBUG,
                                   ,m_port
                                   ,NULL, NULL, &s_treat_request, (void*)this
                                   ,MHD_OPTION_CONNECTION_TIMEOUT, (unsigned int) 120
                                   ,MHD_OPTION_END
                                   );
        if (m_daemon == NULL)
        {
            throw quicky_exception::quicky_logic_exception("Unable to start webserver",__LINE__,__FILE__);
        }

        char l_buffer[HOST_NAME_MAX + 1] = { 0 };
        if(!l_buffer[0])
        {
            char l_name[HOST_NAME_MAX + 1];
            memset(l_name, 0, sizeof( l_name));
            snprintf(l_name,sizeof( l_name), "127.0.0.1");
            strncpy(l_buffer,l_name,sizeof(l_buffer));
          
            gethostname(l_name,HOST_NAME_MAX);
            l_name[HOST_NAME_MAX] = 0;
            struct hostent *l_hostent = 0;
            l_hostent = gethostbyname(l_name);
            if (l_hostent)
            {
                int l_index = 0;
                struct in_addr l_addr;
                while (l_hostent->h_addr_list[l_index] != 0)
                {
                    l_addr.s_addr = *(uint32_t *)l_hostent->h_addr_list[l_index++];
                    strncpy( l_buffer, inet_ntoa(l_addr),sizeof(l_buffer));
                }
            }
        }
        std::cout << "Web server started on " << l_buffer << ":" << m_port << std::endl ;
    }

    //----------------------------------------------------------------------------

    emp_web_server::internal_result_t emp_web_server::s_treat_request (void * p_callback_data
                                                                      ,struct MHD_Connection * p_connection
                                                                      ,const char * p_url
                                                                      ,const char * p_method
                                                                      ,const char * p_http_version
                                                                      ,const char * p_upload_data
                                                                      ,size_t * p_upload_data_size
                                                                      ,void ** p_connection_ptr
                                                                      )
    {
        assert(p_callback_data);
        emp_web_server * l_server = (emp_web_server*)p_callback_data;
        return l_server->treat_request(p_connection
                                      ,p_url
                                      ,p_method
                                      ,p_http_version
                                      ,p_upload_data
                                      ,p_upload_data_size
                                      ,p_connection_ptr
                                      );
    }

    //----------------------------------------------------------------------------
    emp_web_server::internal_result_t emp_web_server::print_out_key (void * p_callback_data
                                                                    ,enum MHD_ValueKind p_kind
                                                                    ,const char *p_key
                                                                    ,const char *p_value
                                                                    )
    {
        std::cout << "\"" << p_key << "\" = \"" <<  p_value << "\"" << std::endl ;
        return MHD_YES;
    }

    //----------------------------------------------------------------------------
    emp_web_server::~emp_web_server()
    {
        for(unsigned int l_index = 0 ; l_index < 1 + 4 * m_info.get_width() * m_info.get_height() ; ++l_index)
        {
            delete[] m_picture_data[l_index];
        }
        delete[] m_picture_data;
        delete[] m_pieces;
        MHD_stop_daemon(m_daemon);
    }
}

#endif // EMP_STRATEGY_H
//EOF
