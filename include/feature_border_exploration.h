/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
      Copyright (C) 2017  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef FEATURE_BORDER_EXPLORATION_H
#define FEATURE_BORDER_EXPLORATION_H

#include "feature_if.h"
#include "border_exploration.h"
#include "border_color_constraint.h"
#include "light_border_pieces_db.h"

#include <map>

namespace edge_matching_puzzle
{
  class feature_border_exploration:public feature_if
  {
  public:
    inline feature_border_exploration(const emp_piece_db & p_db,
				      const emp_FSM_info & p_info,
				      const std::string & p_initial_situation
				      );
    // Virtual methods inherited from feature_if
    inline void run(void);
    // End of virtual methods inherited from feature_if
    inline ~feature_border_exploration(void);
  private:
    border_exploration *m_border_exploration;
    unsigned int m_border_edges[60];
  };
 
  //----------------------------------------------------------------------------
  feature_border_exploration::feature_border_exploration(const emp_piece_db & p_db,
							 const emp_FSM_info & p_info,
							 const std::string & p_initial_situation
							 ):
    m_border_exploration(nullptr)
  {

    // Compute reorganised colors
    unsigned int l_unaffected_B2C_color = 1;
    std::map<unsigned int, unsigned int> l_reorganised_all_colors;
    for(auto l_iter: p_db.get_border2center_colors())
	{
	  l_reorganised_all_colors.insert(std::map<unsigned int, unsigned int>::value_type(l_iter,l_unaffected_B2C_color));
	  std::cout << "Reorganised border2center colors : " << l_iter << " <=> " << l_unaffected_B2C_color << std::endl ;
	  ++l_unaffected_B2C_color;
	}
    for(auto l_iter: p_db.get_border_colors())
	{
	  l_reorganised_all_colors.insert(std::map<unsigned int, unsigned int>::value_type(l_iter,l_unaffected_B2C_color));
	  std::cout << "Reorganised all colors : " << l_iter << " <=> " << l_unaffected_B2C_color << std::endl ;
	  ++l_unaffected_B2C_color;
	}

    // Border pieces constraint summary. Index 0 correspond to no pieces
    border_color_constraint l_border_constraints[23];
    l_border_constraints[0].fill(true);

    // Binary representation of border_pieces
    light_border_pieces_db l_border_pieces;
    unsigned int l_nb_corners = p_db.get_nb_pieces(emp_types::t_kind::CORNER);
    for(unsigned int l_index = 0;
	l_index < l_nb_corners;
	++l_index
	)
      {
	const emp_piece_corner & l_corner = p_db.get_corner(l_index);

	// Search for border edges to normalise corner representation
	unsigned int l_border_edge_index = 0;
	while(l_corner.get_color((emp_types::t_orientation)l_border_edge_index) || l_corner.get_color((emp_types::t_orientation)((1 + l_border_edge_index) % 4)))
	  {
	    ++l_border_edge_index;
	  }
	emp_types::t_color_id l_left_color = l_corner.get_color((emp_types::t_orientation)(( 3 + l_border_edge_index) % 4));
	unsigned int l_piece_id = l_corner.get_id() - 1;
	m_border_edges[l_piece_id] = l_border_edge_index;
	l_border_pieces.set_colors(l_piece_id,
				   l_left_color,
				   0,
				   l_corner.get_color((emp_types::t_orientation)(( 2 + l_border_edge_index) % 4))
				   );
	l_border_constraints[l_left_color].set_bit(l_piece_id);
      }

    unsigned int l_nb_borders = p_db.get_nb_pieces(emp_types::t_kind::BORDER);
    for(unsigned int l_index = 0;
	l_index < l_nb_borders;
	++l_index
	)
      {
	const emp_piece_border & l_border = p_db.get_border(l_index);
	// No need to normalise border representation because it is done by constructor of emp_piece_border
	emp_types::t_color_id l_left_color = l_border.get_border_colors().first;
	emp_types::t_color_id l_center_color = l_border.get_center_color();
	unsigned int l_piece_id = l_border.get_id() - 1;
	m_border_edges[l_piece_id] = (unsigned int)l_border.get_border_orientation();
	l_border_pieces.set_colors(l_piece_id,
				   l_left_color,
				   l_center_color,
				   l_border.get_border_colors().second
				   );
	l_border_constraints[l_left_color].set_bit(l_piece_id);
	l_border_constraints[l_center_color].set_bit(l_piece_id);
      }
    m_border_exploration = new border_exploration(p_db.get_border2center_colors_nb(),
						  l_reorganised_all_colors,
						  l_border_constraints,
						  l_border_pieces,
						  p_initial_situation
						  );
  }

  //----------------------------------------------------------------------------
  void feature_border_exploration::run(void)
  {
    m_border_exploration->run(m_border_edges);
  }

  //----------------------------------------------------------------------------
  feature_border_exploration::~feature_border_exploration(void)
  {
    delete m_border_exploration;
  }
 }
#endif // FEATURE_BORDER_EXPLORATION_H
//EOF
