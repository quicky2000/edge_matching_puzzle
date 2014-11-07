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
#ifndef EMP_PIECE_DB_H
#define EMP_PIECE_DB_H

#include "emp_piece.h"
#include "emp_piece_border.h"
#include "emp_piece_corner.h"
#include "emp_piece_constraint.h"
#include "emp_link.h"
#include <map>
#include <set>
#include <cmath>

namespace edge_matching_puzzle
{
  class emp_piece_db
  {
  public:
    inline emp_piece_db(const std::vector<emp_piece> & p_pieces,
			const unsigned int & p_width,
			const unsigned int & p_height);
    inline const emp_piece & get_piece(const unsigned int & p_id)const;
    inline void get_pieces(const emp_types::t_kind & p_kind,
                           const std::set<emp_constraint> & p_constraints,
                           std::vector<emp_types::t_oriented_piece> & p_pieces)const;
    inline ~emp_piece_db(void);
    inline const emp_piece_corner & get_corner(const unsigned int & p_index)const;
    inline const std::set<emp_types::t_piece_id> * const get_identical_pieces(const emp_types::t_piece_id & p_id)const;
  private:
    inline void compute_constraints(const emp_piece & p_piece);
    static inline void print_list(const std::string & p_name,
				  const std::set<unsigned int> & p_list);
    static inline void print_auto_similarities(const emp_piece::t_auto_similarity & p_similarity,
                                               const std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id> & p_auto_similarities);

   typedef std::map<emp_types::t_color_id,std::set<emp_types::t_oriented_piece> > t_color2oriented_pieces;

    static inline void add_piece(t_color2oriented_pieces& p_color2pieces, 
				 const emp_types::t_color_id & p_color_id,
				 const emp_types::t_oriented_piece & p_piece);
    inline void record_identical_pieces(const emp_types::t_piece_id & p_id1,
					const emp_types::t_piece_id & p_id2);

   typedef std::pair<std::pair<emp_types::t_color_id,emp_types::t_color_id>,std::pair<emp_types::t_color_id,emp_types::t_color_id> > t_constraint;
   typedef std::map<emp_types::t_piece_id,std::set<emp_types::t_piece_id> > t_identical_pieces_db;
   const std::vector<emp_piece> & m_pieces;
    emp_piece_corner* m_corners[4];


    typedef std::map<emp_piece_constraint,std::set<emp_types::t_oriented_piece> > t_constraint_db;
    t_constraint_db ** m_constraint_db;
    typedef std::map<emp_types::t_oriented_piece,std::set<emp_piece_constraint> > t_piece2constraint_db;
    t_piece2constraint_db m_piece2constraint_db;
    std::set<emp_constraint> m_single_constraints;
    t_identical_pieces_db m_identical_pieces_db;
  };
  //----------------------------------------------------------------------------
  const std::set<emp_types::t_piece_id> * const emp_piece_db::get_identical_pieces(const emp_types::t_piece_id & p_id)const
    {
      t_identical_pieces_db::const_iterator l_iter = m_identical_pieces_db.find(p_id);
      return (m_identical_pieces_db.end() == l_iter ? NULL : &(l_iter->second));
    }

  //----------------------------------------------------------------------------
  const emp_piece_corner & emp_piece_db::get_corner(const unsigned int & p_index)const
    {
      assert (p_index < 4);
      assert(m_corners[p_index]);
      return *(m_corners[p_index]);
    }

  //----------------------------------------------------------------------------
  emp_piece_db::emp_piece_db(const std::vector<emp_piece> & p_pieces,
			     const unsigned int & p_width,
			     const unsigned int & p_height):
    m_pieces(p_pieces),
    m_constraint_db(new t_constraint_db*[3])
    {
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << "Building piece database" << std::endl;

      unsigned int l_nb_edge = (p_width - 1) * p_height + (p_height - 1) * p_width;
      std::cout << "Width = " << p_width << std::endl ;
      std::cout << "Heigth = " << p_height << std::endl ;
      std::cout << "Nb edges : " << l_nb_edge << std::endl ;

      // Creating constraints database
      for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
	  l_index <= (unsigned int)emp_types::t_kind::CORNER;
	  ++l_index)
	{
	  m_constraint_db[l_index] = new t_constraint_db[4 - l_index];
	}

      // Creating corner array
      for(unsigned int l_index = 0 ; l_index < 4 ; ++l_index)
        {
          m_corners[l_index] = NULL;
        }

      // Pieces counters
      unsigned int l_nb[((uint32_t)emp_types::t_kind::CORNER) + 1] = {0,0,0};
      // Auto similarity counters
      unsigned int l_nb_auto_similarity[((uint32_t)emp_piece::t_auto_similarity::SIMILAR) + 1] = {0,0,0};

      typedef std::set<emp_types::t_color_id> t_color_list;
      // List of colors composing center of puzzle
      t_color_list l_center_colors;
      // All colors
      t_color_list l_colors;
      // List of colors composing border of puzzle
      t_color_list l_border_colors;
      // Colors of edge pieces that are related to center pieces
      t_color_list l_border2center_colors;
      // Colors of corners
      t_color_list l_corner_colors;

      // Store which piece contains color
      t_color2oriented_pieces l_color2pieces;

      std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id> l_auto_similarities;
 
      // Examining pieces
      for(auto l_iter : p_pieces)
	{
	  ++(l_nb[((unsigned int)l_iter.get_kind())]);
          ++(l_nb_auto_similarity[((unsigned int)l_iter.get_auto_similarity())]);
          l_auto_similarities.insert(std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::value_type(l_iter.get_auto_similarity(),l_iter.get_id()));

          for(auto l_iter_bis : p_pieces)
            {
              if(l_iter_bis.get_id() == l_iter.get_id()) break;
	      if(l_iter == l_iter_bis)
		{
		  record_identical_pieces(l_iter.get_id(),l_iter_bis.get_id());
		  record_identical_pieces(l_iter_bis.get_id(),l_iter.get_id());
		}
            }

          switch(l_iter.get_kind())
            {
            case emp_types::t_kind::CENTER:
              for(unsigned int l_index = (unsigned int)emp_types::t_orientation::NORTH ;
                  l_index <= (unsigned int)emp_types::t_orientation::WEST; 
                  ++l_index)
                {
		  emp_types::t_color_id l_color = l_iter.get_color((emp_types::t_orientation )l_index);
                  l_center_colors.insert(l_color);
                  l_colors.insert(l_color);
                  add_piece(l_color2pieces,l_color,emp_types::t_oriented_piece(l_iter.get_id(),(emp_types::t_orientation )l_index));
                }
              break;
            case emp_types::t_kind::BORDER:
              {
                emp_piece_border l_border(l_iter);
                std::pair<emp_types::t_color_id,emp_types::t_color_id> l_piece_border_colors = l_border.get_border_colors();
		l_colors.insert(l_piece_border_colors.first);
		l_colors.insert(l_piece_border_colors.second);
		l_border_colors.insert(l_piece_border_colors.first);
		l_border_colors.insert(l_piece_border_colors.second);
		l_center_colors.insert(l_border.get_center_color());
		l_border2center_colors.insert(l_border.get_center_color());
                
                const std::pair<emp_types::t_orientation,emp_types::t_orientation> & l_colors_orientations = l_border.get_colors_orientations();
                add_piece(l_color2pieces,l_piece_border_colors.first,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.first));
                add_piece(l_color2pieces,l_piece_border_colors.second,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.second));
                add_piece(l_color2pieces,l_border.get_center_color(),emp_types::t_oriented_piece(l_iter.get_id(),l_border.get_center_orientation()));                
              }
              break;
	    case emp_types::t_kind::CORNER:
              {
                emp_piece_corner* l_corner = new emp_piece_corner(l_iter);
                m_corners[l_nb[(unsigned int)emp_types::t_kind::CORNER] - 1] = l_corner;

		std::pair<emp_types::t_color_id,emp_types::t_color_id> l_piece_border_colors = l_corner->get_border_colors();
		l_colors.insert(l_piece_border_colors.first);
		l_colors.insert(l_piece_border_colors.second);
		l_border_colors.insert(l_piece_border_colors.first);
		l_border_colors.insert(l_piece_border_colors.second);
		l_corner_colors.insert(l_piece_border_colors.first);
		l_corner_colors.insert(l_piece_border_colors.second);

                const std::pair<emp_types::t_orientation,emp_types::t_orientation> & l_colors_orientations = l_corner->get_colors_orientations();
                add_piece(l_color2pieces,l_piece_border_colors.first,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.first));
                add_piece(l_color2pieces,l_piece_border_colors.second,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.second));
	      }
	      break;
            default:
              throw quicky_exception::quicky_logic_exception("Unsupported kind of piece \""+emp_types::kind2string(l_iter.get_kind())+"\"",__LINE__,__FILE__);
            }
	  compute_constraints(l_iter);
        }

      // Display number of pieces
      for(unsigned int l_index = ((uint32_t)emp_types::t_kind::CENTER) ;
	  l_index <= ((uint32_t)emp_types::t_kind::CORNER);
	  ++l_index)
	{
	  std::cout << "\t" << emp_types::kind2string((emp_types::t_kind)l_index) << "\t: " << l_nb[l_index] << std::endl ;
	}

      // Display number of auto similar pieces
      std::cout << "Number of pieces depending on auto_similarity :" << std::endl ;
      for(unsigned int l_index = ((uint32_t)emp_piece::t_auto_similarity::NONE) ;
	  l_index <= ((uint32_t)emp_piece::t_auto_similarity::SIMILAR);
	  ++l_index)
	{
	  std::cout << "\t" << emp_piece::auto_similarity2string((emp_piece::t_auto_similarity)l_index) << "\t: " << l_nb_auto_similarity[l_index] << std::endl ;
	}

      // Display auto similar pieces
      print_auto_similarities(emp_piece::t_auto_similarity::HALF_SIMILAR,l_auto_similarities);
      print_auto_similarities(emp_piece::t_auto_similarity::SIMILAR,l_auto_similarities);

      if(m_identical_pieces_db.size())
	{
	  std::cout << "Identical pieces:" << std::endl ;
	  for(auto l_iter: m_identical_pieces_db)
	    {
	      std::cout << l_iter.first << " <==> { " ;
	      bool l_first = true;
	      for(auto l_iter_second : l_iter.second)
		{
		  if(!l_first) 
		    {
		      std::cout << ", " ;
		    }
		  else
		    {
		      l_first = false;
		    }
		  std::cout << l_iter_second;
		}
	      std::cout << "}" << std::endl ;
	    }
	}

      print_list("Colors",l_colors);
      print_list("Center colors",l_center_colors);
      print_list("Border colors",l_border_colors);
      print_list("Corner colors",l_corner_colors);
      print_list("Border to center colors",l_border2center_colors);

      std::cout << "Color repartition on pieces :" <<std::endl ;
      unsigned int l_total = 0;
      bool l_error = false;
      for(auto l_iter: l_colors)
        {
          unsigned int l_nb = l_color2pieces.find(l_iter)->second.size();
          if(l_nb % 2) 
	    {
	      l_error = true;
	      std::cout << "ERROR : " ;
	    }
          std::cout << "Color = " << l_iter << " appears on " << l_nb << " pieces edge" << std::endl ;
          l_total += l_nb;
        }
      if(l_error)
	{
	  throw quicky_exception::quicky_logic_exception("Number of pieces edge with color should be multiple of 2 or the puzzle cannot be solved",__LINE__,__FILE__);
	}
      if(2 * l_nb_edge != l_total)
        {
          std::stringstream l_stream1 ;
          l_stream1 << l_nb_edge;
          std::stringstream l_stream2 ;
          l_stream2 << l_total;
          throw quicky_exception::quicky_logic_exception("Incoherency between number of edges ("+l_stream1.str()+") and color on edges ("+l_stream2.str()+")",__LINE__,__FILE__);
        }

      std::vector<emp_link> l_links;
      typedef std::multimap<emp_types::t_piece_id,emp_link> t_piece2links;
      t_piece2links l_piece2links;
      typedef std::multimap<emp_types::t_oriented_piece,emp_link> t_piece_edge_2links;
      t_piece_edge_2links l_piece_edge_2links;
      for(unsigned int l_index1 = 0 ; l_index1 < p_pieces.size() ; ++l_index1)
        {
          for(unsigned int l_orient_index1 = (unsigned int)emp_types::t_orientation::NORTH ;
              l_orient_index1 <= (unsigned int)emp_types::t_orientation::WEST; 
              ++l_orient_index1)
            {
              emp_types::t_piece_id l_id = p_pieces[l_index1].get_id();
              emp_types::t_orientation l_orientation = (emp_types::t_orientation)l_orient_index1;
              emp_types::t_color_id l_color = p_pieces[l_index1].get_color(l_orientation);
              if(l_color)
                {
                  t_color2oriented_pieces::const_iterator l_iter_color = l_color2pieces.find(l_color);
                  for (auto l_iter : l_iter_color->second)
                    {
                      if(l_iter.first < l_id)
                        {
                          emp_link l_link(l_id,l_orientation,l_iter.first,l_iter.second);
                          l_links.push_back(l_link);
                          //                          std::cout << "Link : " << l_link << std::endl ;
                          l_piece2links.insert(t_piece2links::value_type(l_id,l_link));
                          l_piece2links.insert(t_piece2links::value_type(l_iter.first,l_link));
                          l_piece_edge_2links.insert(t_piece_edge_2links::value_type(emp_types::t_oriented_piece(l_id,l_orientation),l_link));
                          l_piece_edge_2links.insert(t_piece_edge_2links::value_type(emp_types::t_oriented_piece(l_iter.first,l_iter.second),l_link));
                        }
                    }
                }
            }
        }
      std::cout << "Nb links = " << l_links.size() << std::endl ;
      unsigned int l_total_pieces_links = 0;
      for(auto l_iter : p_pieces)
        {
          unsigned int l_nb = l_piece2links.count(l_iter.get_id());
          std::cout << emp_types::kind2string(l_iter.get_kind()) << "\t: Piece = " << l_iter.get_id() << " has " << l_nb << " links" << std::endl ;
          l_total_pieces_links += l_nb;
        }
      if(l_total_pieces_links != 2 * l_links.size())
        {
          std::stringstream l_stream1 ;
          l_stream1 << l_total_pieces_links;
          std::stringstream l_stream2 ;
          l_stream2 << l_links.size();
          throw quicky_exception::quicky_logic_exception("Incoherency between number of links ("+l_stream1.str()+") and links per pieces ("+l_stream2.str()+")",__LINE__,__FILE__);
        }
      l_total_pieces_links = 0;
      for(auto l_iter : p_pieces)
        {
          for(unsigned int l_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
              l_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
              ++l_orient_index)
            {
              emp_types::t_piece_id l_id = l_iter.get_id();
              emp_types::t_orientation l_orientation = (emp_types::t_orientation)l_orient_index;
              if(l_iter.get_color(l_orientation))
                {
                  unsigned int l_nb = l_piece_edge_2links.count(emp_types::t_oriented_piece(l_id,l_orientation));
                  std::cout << emp_types::kind2string(l_iter.get_kind()) << "\t: Piece = " << l_id << " Edge = " << emp_types::orientation2string(l_orientation) << " has " << l_nb << " links" << std::endl ;
                  l_total_pieces_links += l_nb;
                }
            }
        }
      if(l_total_pieces_links != 2 * l_links.size())
        {
          std::stringstream l_stream1 ;
          l_stream1 << l_total_pieces_links;
          std::stringstream l_stream2 ;
          l_stream2 << l_links.size();
          throw quicky_exception::quicky_logic_exception("Incoherency between number of links ("+l_stream1.str()+") and links per pieces edge ("+l_stream2.str()+")",__LINE__,__FILE__);
        }

      for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
	  l_index <= (unsigned int)emp_types::t_kind::CORNER;
	  ++l_index)
	{
	  for(unsigned int l_index2 = 0 ; l_index2 < 4 - l_index ; ++l_index2)
            {
              for(auto l_iter : m_constraint_db[l_index][l_index2])
                {
                  std::cout << emp_types::kind2string((emp_types::t_kind)l_index) << " pieces matching size " << l_index2 + l_index + 1 << " constraint : " << l_iter.first << ": " << l_iter.second.size() << std::endl ;         
                  for(auto l_iter2 : l_iter.second)
                    {
                      std::cout << "\t" << emp_types::orientation2string(l_iter2.second) << " oriented piece " << get_piece(l_iter2.first) << std::endl ;
                    }
                }
            }
	}

      for(auto l_iter : m_piece2constraint_db)
        {
          std::cout << "Computed constraints for " << emp_types::orientation2string(l_iter.first.second) << " oriented piece " << get_piece(l_iter.first.first) << std::endl ;         
          for(auto l_iter2 : l_iter.second)
            {
              std::cout << "\t" << l_iter2 << std::endl ;
            }
        }

      std::cout << "Piece database builded" << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
    }

  //----------------------------------------------------------------------------
  const emp_piece & emp_piece_db::get_piece(const unsigned int & p_id)const
    {
      assert(p_id && p_id <= m_pieces.size());
      return m_pieces[p_id - 1];
    }

  //----------------------------------------------------------------------------
  void emp_piece_db::get_pieces(const emp_types::t_kind & p_kind,
                                const std::set<emp_constraint> & p_constraints,
                                std::vector<emp_types::t_oriented_piece> & p_pieces)const
  {
    unsigned int l_index = (unsigned int) p_kind;
    assert(l_index < 3);
    unsigned int l_index2 = p_constraints.size() - l_index - 1;
    assert(l_index2 < 4);
    emp_piece_constraint l_constraints(p_constraints);
    t_constraint_db::const_iterator l_iter = m_constraint_db[l_index][l_index2].find(l_constraints);
    if(m_constraint_db[l_index][l_index2].end() == l_iter) return;
    for(auto l_piece_iter : l_iter->second)
      {
        p_pieces.push_back(l_piece_iter);
      }
  }

  //----------------------------------------------------------------------------
  void emp_piece_db::compute_constraints(const emp_piece & p_piece)
  {
    unsigned int l_increment = pow(2,(unsigned int) p_piece.get_auto_similarity());
    // Compute for different orientations of piece
    for(unsigned int l_piece_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
	l_piece_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
	l_piece_orient_index += l_increment)
      {
	emp_types::t_oriented_piece l_oriented_piece(p_piece.get_id(),(emp_types::t_orientation)l_piece_orient_index);
	std::array<const emp_constraint*,4> l_oriented_piece_constraints;
	// Fill array with oriented colors
	for(unsigned int l_border_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
	    l_border_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
	    ++l_border_orient_index)
	  {
            emp_constraint l_constraint(p_piece.get_color((emp_types::t_orientation)l_border_orient_index,(emp_types::t_orientation)l_piece_orient_index),(emp_types::t_orientation)l_border_orient_index);
            std::set<emp_constraint>::const_iterator l_iter_constraint = m_single_constraints.find(l_constraint);
            if(m_single_constraints.end() == l_iter_constraint)
              {
                l_iter_constraint = m_single_constraints.insert(l_constraint).first;
              }
	    l_oriented_piece_constraints[l_border_orient_index] = &(*l_iter_constraint);
	  }

	// Compute the various constraints
	for(unsigned int l_index = 1 ; l_index <= 15 ; ++l_index)
	  {
	    std::set<emp_constraint> l_constraint;
	    for(unsigned int l_border_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
		l_border_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
		++l_border_orient_index)
	      {
		if(!l_oriented_piece_constraints[l_border_orient_index]->get_color() || (( 1 << l_border_orient_index) & l_index))
		  {
		    l_constraint.insert(*(l_oriented_piece_constraints[l_border_orient_index]));
		  }
	      }
	    if(l_constraint.size() >  (unsigned int)p_piece.get_kind())
	      {

                emp_piece_constraint l_piece_constraint(l_constraint);
		t_constraint_db & l_constraint_db = m_constraint_db[(unsigned int)p_piece.get_kind()][l_constraint.size() - (unsigned int)p_piece.get_kind() - 1];
		t_constraint_db::iterator l_iter = l_constraint_db.find(l_constraint);
		if(l_constraint_db.end() == l_iter)
		  {
		    l_iter = l_constraint_db.insert(t_constraint_db::value_type(l_constraint,std::set<emp_types::t_oriented_piece>())).first;
		  }
		l_iter->second.insert(l_oriented_piece);

                t_piece2constraint_db::iterator l_iter_piece2constraint = m_piece2constraint_db.find(l_oriented_piece);
                if(m_piece2constraint_db.end() == l_iter_piece2constraint)
                  {
                    l_iter_piece2constraint = m_piece2constraint_db.insert(t_piece2constraint_db::value_type(l_oriented_piece,std::set<emp_piece_constraint>())).first;
                  }

                l_iter_piece2constraint->second.insert(l_piece_constraint);
	      }
	  }

      }
  }

  //------------------------------------------------------------------------------
  void emp_piece_db::print_list(const std::string & p_name,
				const std::set<unsigned int> & p_list)
  {
    std::cout << p_name << " : " << p_list.size() ;
    if(p_list.size())
      {
	std::cout << " = { " << *(p_list.begin());
	for(std::set<unsigned int>::const_iterator l_iter = ++(p_list.begin());
	    l_iter != p_list.end();
	    ++l_iter)
	  {
	    std::cout << ", " << *l_iter ;
	}
	std::cout << "}" ;
      }
    std::cout << std::endl ;
  }

  //------------------------------------------------------------------------------
  void emp_piece_db::print_auto_similarities(const emp_piece::t_auto_similarity & p_similarity,
                                             const std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id> & p_auto_similarities)
  {
    std::pair<std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::const_iterator,std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::const_iterator> l_values = p_auto_similarities.equal_range(p_similarity);
    if(l_values.first != l_values.second)
      {
        std::cout << "List of pieces which are " << emp_piece::auto_similarity2string(p_similarity) << ":" << std::endl ;
        std::cout  << "{ " << l_values.first->second ;
        for(std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::const_iterator l_iter = ++(l_values.first) ;
            l_values.first != l_values.second;
            ++l_values.first)
          {
	    std::cout << ", " << l_iter->second ;
          }
	std::cout << "}"  << std::endl;
      }
  }

  //------------------------------------------------------------------------------
  void emp_piece_db::add_piece(t_color2oriented_pieces& p_color2pieces, 
			       const emp_types::t_color_id & p_color_id,
			       const emp_types::t_oriented_piece & p_piece)
  {
    t_color2oriented_pieces::iterator l_iter_color = p_color2pieces.find(p_color_id);
    if(p_color2pieces.end() == l_iter_color)
      {
	l_iter_color = p_color2pieces.insert(t_color2oriented_pieces::value_type(p_color_id,std::set<emp_types::t_oriented_piece>())).first;
      }
    l_iter_color->second.insert(p_piece);
  }

  //------------------------------------------------------------------------------
  void emp_piece_db::record_identical_pieces(const emp_types::t_piece_id & p_id1,
			       const emp_types::t_piece_id & p_id2)
  {
    t_identical_pieces_db::iterator l_iter = m_identical_pieces_db.find(p_id1);
    if(m_identical_pieces_db.end() == l_iter)
      {
	l_iter = m_identical_pieces_db.insert(t_identical_pieces_db::value_type(p_id1,std::set<emp_types::t_piece_id>())).first;
      }
    l_iter->second.insert(p_id2);
  }


  //----------------------------------------------------------------------------
  emp_piece_db::~emp_piece_db(void)
    {
      for(unsigned int l_index = 0 ; l_index < 4 ; ++l_index)
        {
          delete m_corners[l_index];
        }
      for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
	  l_index <= (unsigned int)emp_types::t_kind::CORNER;
	  ++l_index)
	{
	  delete[] m_constraint_db[l_index];
	}
      delete[] m_constraint_db;
    }
}
#endif // EMP_PIECE_DB_H
