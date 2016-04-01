/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2014  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EMP_GUI_H
#define EMP_GUI_H

#include "simple_gui.h"
#include "quicky_files.h"
#include "emp_piece.h"
#include "emp_FSM_situation.h"
#include "my_bmp.h"
#include "quicky_exception.h"
#include <sstream>
#include <vector>
#include <string>
#include <map>

namespace edge_matching_puzzle
{
  class emp_gui: public simple_gui::simple_gui
  {
  public:
    inline emp_gui(const unsigned int & p_puzzle_width,
                   const unsigned int & p_puzzle_height,
                   const std::string & p_ressources,
                   const std::vector<emp_piece> & p_pieces);

    inline void set_piece(const unsigned int & p_x,
                          const unsigned int & p_y,
                          const emp_types::t_piece_id & p_id,
                          const emp_types::t_orientation & p_orientation);
    inline void display(const emp_FSM_situation & p_situation);
    inline ~emp_gui(void);
    inline const lib_bmp::my_bmp & get_picture(const unsigned int & p_piece_id)const;
  private:
    inline void set_piece_without_lock(const unsigned int & p_x,
                                       const unsigned int & p_y,
                                       const emp_types::t_piece_id & p_id,
                                       const emp_types::t_orientation & p_orientation);
    inline void clear_piece_without_lock(const unsigned int & p_x,
                                         const unsigned int & p_y);
    const unsigned int m_puzzle_width;
    const unsigned int m_puzzle_height;
    lib_bmp::my_bmp **m_pieces_pictures;
    unsigned int m_piece_size;
    const unsigned int m_separator_size;
    const unsigned int m_nb_pieces;
    
  };

  //----------------------------------------------------------------------------
  emp_gui::~emp_gui(void)
    {
      for(unsigned int l_index = 0 ; l_index < m_nb_pieces ; ++l_index)
	{
	  delete m_pieces_pictures[l_index];
	}
      delete[] m_pieces_pictures;
    }

  //----------------------------------------------------------------------------
  emp_gui::emp_gui(const unsigned int & p_puzzle_width,
                   const unsigned int & p_puzzle_height,
                   const std::string & p_ressources,
                   const std::vector<emp_piece> & p_pieces):
    simple_gui(),
    m_puzzle_width(p_puzzle_width),
    m_puzzle_height(p_puzzle_height),
    m_pieces_pictures(new lib_bmp::my_bmp*[p_pieces.size()]),
    m_piece_size(0),
    m_separator_size(2),
    m_nb_pieces(p_pieces.size())
    {
      // Get screen characteristics
      uint32_t l_screen_width;
      uint32_t l_screen_height;
      uint32_t l_screen_bits_per_pixel;
      
      this->get_screen_info(l_screen_width,l_screen_height,l_screen_bits_per_pixel);
      std::cout << "Resolution = " << l_screen_width << " * " << l_screen_height << " : " << l_screen_bits_per_pixel << " bits/pixel" << std::endl;
      if(!l_screen_width || !l_screen_height)
        {
          l_screen_width = 800;
          l_screen_height = 600;
        }
      // Get available ressources
      std::vector<std::string> l_file_list;
      quicky_utils::quicky_files::list_content(p_ressources,l_file_list);

      if(!l_file_list.size())
        {
	  throw quicky_exception::quicky_logic_exception("Unable to find any files in \"" + p_ressources + "\"",__LINE__,__FILE__);
        }

      for(std::vector<std::string>::const_iterator l_iter = l_file_list.begin();
          l_iter != l_file_list.end();
          ++l_iter)
        {
          uint32_t l_ressource_size = strtoul(l_iter->c_str(),NULL,10);
          if((l_ressource_size < ( l_screen_height - m_separator_size * (p_puzzle_height - 1))/ p_puzzle_height) && (l_ressource_size < (l_screen_width - m_separator_size * (p_puzzle_width - 1))/ p_puzzle_width) && l_ressource_size > m_piece_size)
            {
              m_piece_size = l_ressource_size;
            }
        }
      if(!m_piece_size)
	{
	  throw quicky_exception::quicky_logic_exception("Unable to find correct ressources for this resolution",__LINE__,__FILE__);
	}
      std::cout << "Best fit ressource is " << m_piece_size << std::endl ;
      for(unsigned int l_index = 0 ; l_index < p_pieces.size() ; ++l_index)
        {
          m_pieces_pictures[l_index] = new lib_bmp::my_bmp(m_piece_size,m_piece_size,24);
        }
      this->create_window(p_puzzle_width * m_piece_size + (p_puzzle_width - 1) * m_separator_size ,p_puzzle_height * m_piece_size + (p_puzzle_height - 1) * m_separator_size);

      std::map<emp_types::t_color_id,lib_bmp::my_bmp *> l_colors_pictures;

      for(auto l_piece : p_pieces)
        {
          for(unsigned int l_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
              l_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
              ++l_orient_index)
            {
              emp_types::t_orientation l_orientation = (emp_types::t_orientation)l_orient_index;
              emp_types::t_color_id l_color_id = l_piece.get_color(l_orientation);
              std::map<emp_types::t_color_id,lib_bmp::my_bmp*>::const_iterator l_iter = l_colors_pictures.find(l_color_id);
              if(l_colors_pictures.end() == l_iter)
                {
                  std::stringstream l_stream;
                  l_stream << l_color_id;
                  std::stringstream l_stream2;
                  l_stream2 << m_piece_size;
                  std::string l_file_name = p_ressources+"/"+l_stream2.str()+"/"+(l_color_id ? l_stream.str():"Border")+".bmp";
                  std::cout << "Loading color " << l_color_id << " from file \"" << l_file_name << "\"" << std::endl ;
                  l_iter = l_colors_pictures.insert(std::map<emp_types::t_color_id,lib_bmp::my_bmp *>::value_type(l_color_id, new lib_bmp::my_bmp(l_file_name))).first;
                }

              for(unsigned int l_index_height = 0 ; l_index_height < m_piece_size / 2 ; ++l_index_height)
                {
                  for(unsigned int l_index_width = l_index_height ; l_index_width < m_piece_size -l_index_height ; ++l_index_width)
                    {
                      switch(l_orientation)
                        {
                        case emp_types::t_orientation::NORTH:
                          m_pieces_pictures[l_piece.get_id() - 1]->set_pixel_color(l_index_width,l_index_height,l_iter->second->get_pixel_color(l_index_width,l_index_height));
                          break;
                        case emp_types::t_orientation::EAST:
                          m_pieces_pictures[l_piece.get_id() - 1]->set_pixel_color(m_piece_size - 1 - l_index_height,l_index_width,l_iter->second->get_pixel_color(l_index_width,l_index_height));
                          break;
                        case emp_types::t_orientation::SOUTH:
                          m_pieces_pictures[l_piece.get_id() - 1]->set_pixel_color(m_piece_size - 1 - l_index_width, m_piece_size - 1 - l_index_height,l_iter->second->get_pixel_color(l_index_width,l_index_height));
                          break;
                        case emp_types::t_orientation::WEST:
                          m_pieces_pictures[l_piece.get_id() - 1]->set_pixel_color(l_index_height, m_piece_size - 1 - l_index_width,l_iter->second->get_pixel_color(l_index_width,l_index_height));
                          break;
                        default:
                          throw quicky_exception::quicky_logic_exception("Unsupported orientation \""+emp_types::orientation2string(l_orientation)+"\"",__LINE__,__FILE__);
                          break;
                        }
                    }
                }
            }
        }

      // Free memory used by color pictures
      for(auto l_color_picture: l_colors_pictures)
        {
          delete l_color_picture.second;
        }
      
   }

    //--------------------------------------------------------------------------
    void emp_gui::set_piece(const unsigned int & p_x,
                            const unsigned int & p_y,
                            const emp_types::t_piece_id & p_id,
                            const emp_types::t_orientation & p_orientation)
    {
      lock();
      set_piece_without_lock(p_x,p_y,p_id,p_orientation);
      unlock();
    }

    //--------------------------------------------------------------------------
    void emp_gui::set_piece_without_lock(const unsigned int & p_x,
                                         const unsigned int & p_y,
                                         const emp_types::t_piece_id & p_id,
                                         const emp_types::t_orientation & p_orientation)
    {
      assert(p_id <= m_nb_pieces);
      for(unsigned int l_index_x = 0 ;
          l_index_x < m_piece_size;
          ++l_index_x)
        {
          const unsigned int l_x = p_x * m_piece_size + p_x * m_separator_size + l_index_x;
          for(unsigned int l_index_y = 0 ;
              l_index_y < m_piece_size;
              ++l_index_y)
            {
              const unsigned int l_y = p_y * m_piece_size + p_y * m_separator_size + l_index_y;
              unsigned int l_oriented_x = 0;
              unsigned int l_oriented_y = 0;
              switch(p_orientation)
                {
                case emp_types::t_orientation::NORTH:
                  l_oriented_x = l_index_x;
                  l_oriented_y = l_index_y;
                  break;
                case emp_types::t_orientation::EAST:
                  l_oriented_x = m_piece_size - 1 - l_index_y;
                  l_oriented_y = l_index_x;
                  break;
                case emp_types::t_orientation::SOUTH:
                  l_oriented_x = m_piece_size - 1 - l_index_x;
                  l_oriented_y = m_piece_size - 1 - l_index_y;
                  break;
                case emp_types::t_orientation::WEST:
                  l_oriented_x = l_index_y;
                  l_oriented_y = m_piece_size - 1 - l_index_x;
                  break;
                default:
                  assert(false);
                  break;
                  
                }
              const lib_bmp::my_color_alpha & l_rgb_color = m_pieces_pictures[p_id - 1]->get_pixel_color(l_oriented_x,l_oriented_y);
              uint32_t l_color_code = get_color_code(l_rgb_color.get_red(),l_rgb_color.get_green(),l_rgb_color.get_blue());
              set_pixel_without_lock(l_x,l_y,l_color_code);
            }
        }
    }
   //--------------------------------------------------------------------------
    void emp_gui::clear_piece_without_lock(const unsigned int & p_x,
                                           const unsigned int & p_y)
    {
      uint32_t l_color_code = get_color_code(0,0,0);
      for(unsigned int l_index_x = 0 ;
          l_index_x < m_piece_size;
          ++l_index_x)
        {
          const unsigned int l_x = p_x * m_piece_size + p_x * m_separator_size + l_index_x;
          for(unsigned int l_index_y = 0 ;
              l_index_y < m_piece_size;
              ++l_index_y)
            {
              const unsigned int l_y = p_y * m_piece_size + p_y * m_separator_size + l_index_y;
              set_pixel_without_lock(l_x,l_y,l_color_code);
            }
        }
    }
    //--------------------------------------------------------------------------
    void emp_gui::display(const emp_FSM_situation & p_situation)
    {
      lock();
      for(unsigned int l_x = 0 ; l_x < m_puzzle_width ; ++l_x)
        {
          for(unsigned int l_y = 0 ; l_y < m_puzzle_height ; ++l_y)
            {
              if(p_situation.contains_piece(l_x,l_y))
                {
                  const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_x,l_y);
                  set_piece_without_lock(l_x,l_y,l_piece.first,l_piece.second);
                }
              else
                {
                  clear_piece_without_lock(l_x,l_y);
                }
            }
        } 
      unlock();
    }

    //--------------------------------------------------------------------------
    const lib_bmp::my_bmp & emp_gui::get_picture(const unsigned int & p_piece_id)const
      {
	assert(p_piece_id);
	assert(p_piece_id <= m_nb_pieces);
	return *(m_pieces_pictures[p_piece_id - 1]);
      }

}
#endif //EMP_TYPES_H
//EOF
