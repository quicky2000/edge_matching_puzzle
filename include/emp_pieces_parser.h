/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching puzzles
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
#ifndef EMP_PIECES_PARSER_H
#define EMP_PIECES_PARSER_H
#include "emp_piece.h"
#include "quicky_exception.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

//#define EMP_PIECES_PARSER_VERBOSE

namespace edge_matching_puzzle
{
  class emp_pieces_parser
  {
  public:
    inline emp_pieces_parser(const std::string & p_name);
    inline void parse(uint32_t & p_width,
                      uint32_t & p_height,
                      std::vector<emp_piece> & p_pieces);
    inline ~emp_pieces_parser(void);
  private:
    std::ifstream m_file;
  };

  //----------------------------------------------------------------------------
  emp_pieces_parser::emp_pieces_parser(const std::string & p_file_name)
    {
      m_file.open(p_file_name.c_str());
      if(!m_file.is_open()) throw quicky_exception::quicky_runtime_exception("Unable to open file \""+p_file_name+"\" for reading",__LINE__,__FILE__);
    }


    //----------------------------------------------------------------------------
    void emp_pieces_parser::parse(uint32_t & p_width,
                                  uint32_t & p_height,
                                  std::vector<emp_piece> & p_pieces)
    {
      std::string l_line;
      std::getline(m_file,l_line);
      const std::string l_hreader_ref("Width,Height");
      if(l_hreader_ref != l_line)
	{
	  throw quicky_exception::quicky_logic_exception("Unsupported puzzle header \""+l_line+"\" : should be \""+l_hreader_ref+"\"",__LINE__,__FILE__);
	}
      if(m_file.eof()) throw quicky_exception::quicky_logic_exception("Truncated file after puzzle header",__LINE__,__FILE__);
      std::getline(m_file,l_line);
      sscanf(l_line.c_str(), "%" SCNu32 ",%" SCNu32 ,&p_width,&p_height);
      if(m_file.eof()) throw quicky_exception::quicky_logic_exception("Truncated file after puzzle dimensions",__LINE__,__FILE__);
      std::getline(m_file,l_line);
      const std::string l_piece_hreader_ref("I,N,E,S,O");
      if(l_piece_hreader_ref != l_line)
	{
	  throw quicky_exception::quicky_logic_exception("Unsupported pieces header \""+l_line+"\" : should be \""+l_piece_hreader_ref+"\"",__LINE__,__FILE__);
	}
      if(m_file.eof()) throw quicky_exception::quicky_logic_exception("Truncated file after piece header",__LINE__,__FILE__);
      while(!m_file.eof())
        {
          std::getline(m_file,l_line);
#ifdef EMP_PIECES_PARSER_VERBOSE
          std::cout << "Line = \"" << l_line << "\"" << std::endl ;
#endif // EMP_PIECES_PARSER_VERBOSE
	  if("" != l_line)
	    {
	      uint32_t l_id = 0 ;
	      uint32_t l_north = 0;
	      uint32_t l_east = 0;
	      uint32_t l_south = 0;
	      uint32_t l_west = 0;
	      sscanf(l_line.c_str(), "%" SCNu32 ",%" SCNu32 ",%" SCNu32 ",%" SCNu32 ",%" SCNu32 ,&l_id,&l_north,&l_east,&l_south,&l_west);
#ifdef EMP_PIECES_PARSER_VERBOSE
	      std::cout << (uint32_t) l_id << "\t" << (uint32_t) l_north << "\t" << (uint32_t)l_east << "\t" << (uint32_t)l_south << "\t" << (uint32_t)l_west <<  std::endl;
#endif // EMP_PIECES_PARSER_VERBOSE
#if GCC_VERSION > 40702
	      p_pieces.push_back(emp_piece( l_id,{ l_north, l_east, l_south, l_west}));
#else
	      const std::array<unsigned int,((unsigned int)(emp_types::t_orientation::WEST))+1> l_colors = { l_north, l_east, l_south, l_west };
	      p_pieces.push_back(emp_piece( l_id,  l_colors));
	      //p_pieces.push_back(emp_piece( l_id,  l_north, l_east, l_south, l_west));
#endif // GCC_VERSION > 40702
	    }
        }
    }

    //----------------------------------------------------------------------------
    emp_pieces_parser::~emp_pieces_parser(void)
      {
        m_file.close();
      }

}

#endif // EMP_PIECES_PARSER_H
//EOF
