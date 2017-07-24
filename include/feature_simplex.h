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
#ifndef FEATURE_SIMPLEX_H
#define FEATURE_SIMPLEX_H

#include "feature_if.h"
#include "emp_piece_db.h"
#include "emp_FSM_info.h"
#include <string>

namespace edge_matching_puzzle
{
  class feature_simplex:public feature_if
  {
  public:
    inline feature_simplex(const emp_piece_db & p_db,
			   const emp_FSM_info & p_info,
			   const std::string & p_initial_situation
			   );
    // Virtual methods inherited from feature_if
    inline void run(void);
    // End of virtual methods inherited from feature_if
    inline ~feature_simplex(void);
  private:
    inline unsigned int get_nb_piece_possibility(const emp_types::t_kind & p_kind,
						 bool p_minor = false
						 );
    inline emp_types::t_kind get_position_kind(const unsigned int & p_x,
					       const unsigned int & p_y
					       );
    inline unsigned int compute_combination(const emp_types::t_kind & p_kind1,
					    const emp_types::t_kind & p_kind2
					    );

    inline void determine_simplex_parameters(const emp_piece_db & p_db);

    const emp_piece_db & m_db;
    const emp_FSM_info & m_info;
  };
 
  //----------------------------------------------------------------------------
  feature_simplex::feature_simplex(const emp_piece_db & p_db,
				   const emp_FSM_info & p_info,
				   const std::string & p_initial_situation
				   ):
    m_db(p_db),
    m_info(p_info)
  {
    determine_simplex_parameters(p_db);
  }

  //----------------------------------------------------------------------------
  void feature_simplex::determine_simplex_parameters(const emp_piece_db & p_db)
  {
    // Compute simplex variable number
    // Corner and border pieces can have only one orientation per position
    // Center pieces can have 4 for orientation per position
    unsigned int l_nb_variable = p_db.get_nb_pieces(emp_types::t_kind::CORNER) * p_db.get_nb_pieces(emp_types::t_kind::CORNER) + p_db.get_nb_pieces(emp_types::t_kind::BORDER) * p_db.get_nb_pieces(emp_types::t_kind::BORDER) + 4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER) * p_db.get_nb_pieces(emp_types::t_kind::CENTER);

    // Compute number of equations
    unsigned int l_nb_equation = 0;

    l_nb_equation += 2 * m_info.get_height() * m_info.get_width();

    for(unsigned int l_y = 0;
	l_y < m_info.get_height();
	++l_y
	)
      {
	for(unsigned int l_x = 0;
	    l_x < m_info.get_width();
	    ++l_x
	    )
	  {
	    emp_types::kind l_kind = get_position_kind(l_x,l_y);
	    std::cout << "Pos[" << l_x << "," << l_y << "] = " << emp_types::kind2string(l_kind) << " | " << get_nb_piece_possibility(l_kind) << std::endl;
	    if(l_x < m_info.get_width() - 1)
	      {
		emp_types::kind l_kind_bis = get_position_kind(l_x + 1,l_y);
		l_nb_equation += compute_combination(l_kind, l_kind_bis);
	      }
	    if(l_y < m_info.get_height() - 1)
	      {
		emp_types::kind l_kind_bis = get_position_kind(l_x,l_y + 1);
		l_nb_equation += compute_combination(l_kind, l_kind_bis);
	      }
	  }
      }
  
    std::cout << "== SImplex characteristics ==" << std::endl;
    std::cout << "Nb variables : " << l_nb_variable << std::endl;
    std::cout << "Nb equations : " << l_nb_equation << std::endl;
  }

  //----------------------------------------------------------------------------
  unsigned int feature_simplex::compute_combination(const emp_types::t_kind & p_kind1,
						    const emp_types::t_kind & p_kind2
						    )
  {
    return get_nb_piece_possibility(p_kind1) * get_nb_piece_possibility(p_kind2, p_kind1 == p_kind2);
  }

  //----------------------------------------------------------------------------
  unsigned int feature_simplex::get_nb_piece_possibility(const emp_types::t_kind & p_kind,
							 bool p_minor
							 )
  {
    assert(p_kind < emp_types::t_kind::UNDEFINED);
    unsigned int l_coef = emp_types::t_kind::CENTER == p_kind ? 4 : 1;
    return l_coef * (m_db.get_nb_pieces(p_kind) - p_minor);
  }

  //----------------------------------------------------------------------------
  emp_types::t_kind feature_simplex::get_position_kind(const unsigned int & p_x,
						       const unsigned int & p_y
						       )
    {
      assert(p_x < m_info.get_width());
      assert(p_y < m_info.get_height());
      emp_types::t_kind l_type = emp_types::t_kind::CENTER;
      if(!p_x || !p_y || m_info.get_width() - 1 == p_x || p_y == m_info.get_height() - 1)
	{
	  l_type = emp_types::t_kind::BORDER;
	  if((!p_x && !p_y) ||
	     (!p_x && p_y == m_info.get_height() - 1) ||
	     (!p_y && p_x == m_info.get_width() - 1) ||
	     (p_y == m_info.get_height() - 1 && p_x == m_info.get_width() - 1)
	     )
	    {
	      l_type = emp_types::t_kind::CORNER;
	    }
	}
      return l_type;
    }

  //----------------------------------------------------------------------------
  void feature_simplex::run(void)
  {
  }

  //----------------------------------------------------------------------------
  feature_simplex::~feature_simplex(void)
  {
  }
 }
#endif // FEATURE_SIMPLEX_H
//EOF
