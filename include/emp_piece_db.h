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
#include "emp_types.h"
#include <map>
#include <set>
#include <cmath>
#include <vector>
#include <sstream>

//#define HANDLE_IDENTICAL_PIECES

namespace edge_matching_puzzle
{
  class emp_piece_db
  {
  public:
    inline emp_piece_db(const std::vector<emp_piece> & p_pieces,
			const unsigned int & p_width,
			const unsigned int & p_height
			);

    /**
       Accessor to number of pieces of each kind
    **/
    inline const unsigned int & get_nb_pieces(const emp_types::t_kind & p_kind)const;

    /**
       Return the number of bits necessary to code color id including a special
       code for border color
    **/
    inline const unsigned int & get_color_id_size(void)const;

    /**
       Return the number of bits necessary to code piece id with ids starting at 0
    **/
    inline const unsigned int & get_piece_id_size(void)const;

    /**
       Return the number of bits necessary to code piece id with id == 0 mean no piece
    **/
    inline const unsigned int & get_dumped_piece_id_size(void)const;

    /**
       Return the id used internally to represent border color
    **/
    inline const unsigned int & get_border_color_id(void)const;

    /**
       Search Piece by Id
    **/
    inline const emp_piece & get_piece(const unsigned int & p_id)const;

    /**
       Return binary representation of piece described by its kind and its
       index in list of piece of same kind ( kind ID)
    **/
    inline const emp_types::t_binary_piece & get_piece(const emp_types::t_kind & p_kind,
                                                       const unsigned int & p_id
						       ) const;

    /**
       Search pieces which specified kind ( corner, border, center ) and matching
       constraint
    **/
    inline void get_pieces(const emp_types::t_kind & p_kind,
                           const std::set<emp_constraint> & p_constraints,
                           std::vector<emp_types::t_oriented_piece> & p_pieces
			   ) const;

    /**
       Search pieces matching constraint.
       The result is a bitfield in which each bit represent a couple
       ( Piece kind id, orientation)
    **/
    inline const emp_types::bitfield & get_pieces(const emp_types::t_binary_piece & p_constraint)const;


    inline const emp_types::bitfield & get_get_binary_identical_pieces(const emp_types::t_kind & p_kind,
								       const emp_types::t_piece_id & p_kind_id
								       ) const;

    /**
       Get Corner by index
    **/
    inline const emp_piece_corner & get_corner(const unsigned int & p_index)const;

    /**
       Get pieces identical to a given piece
    **/
    inline const std::set<emp_types::t_piece_id> * const get_identical_pieces(const emp_types::t_piece_id & p_id)const;

    /**
       Return index of piece in its kind category
    **/
    inline const unsigned int get_kind_index(const emp_types::t_piece_id & p_id)const;

    /**
       Return index of color in its kind category knowing that category is the more specific one,
       for example is color is both on corner and border method with return color index in corner kind
    **/
    inline const unsigned int get_color_kind_index(const emp_types::t_color_id & p_id)const;

    /**
       Return index of color in the specified kind category
       @param p_id Color id
       @param p_kind color category
     */
    inline const unsigned int get_color_kind_index(const emp_types::t_color_id & p_id,
						   const emp_types::t_kind & p_kind
						   )const;

    /**
       Return color kind
    **/
    inline const emp_types::t_kind get_color_kind(const emp_types::t_color_id & p_id)const;

    /**
       Return color id from its kind and kind index
     **/
    inline const emp_types::t_color_id get_color_id(const emp_types::t_kind & p_kind,
						    const unsigned int & p_index
						    ) const;

    /**
       Return a map containing for earch center color of border pieces the number of occurence of this colors
    */
    inline const std::map<emp_types::t_color_id,unsigned int> & get_border2center_colors_nb(void)const;

    /**
       Type used to store list of colors
    **/
    typedef std::set<emp_types::t_color_id> t_color_list;

    /**
       Return All colors
    **/
    inline const t_color_list & get_colors(void) const;

    /**
       Return List of colors composing center of puzzle
    **/
    inline const t_color_list & get_center_colors(void) const;

    /**
       Return List of colors composing border of puzzle
    **/
    inline const t_color_list & get_border_colors(void) const;

    /**
       Return Colors of edge pieces that are related to center pieces
    **/
    inline const t_color_list & get_border2center_colors(void) const;

    /**
       Return Colors of corners
    **/
    inline const t_color_list & get_corner_colors(void) const;

    inline ~emp_piece_db(void);

  private:
    /**
       Compute nb bits needed to code the value passed as parameter
    **/
    static inline unsigned int compute_nb_bits(unsigned int p_value);

    /**
       Fill binary constraint db
       For constraint matched by this piece it it will be indicated that this
       piece match by setting the corresponding bit to one
    **/
    inline void compute_binary_constraints(const emp_piece & p_piece);


    /**
       Fill constraint db
       For constraint matched by this piece it it will be indicated that this
       piece match by inserting its id in a list
    **/
    inline void compute_constraints(const emp_piece & p_piece);

    static inline void print_list(const std::string & p_name,
				  const std::set<unsigned int> & p_list);
    static inline void print_auto_similarities(const emp_piece::t_auto_similarity & p_similarity,
                                               const std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id> & p_auto_similarities
					       );

    typedef std::map<emp_types::t_color_id,std::set<emp_types::t_oriented_piece> > t_color2oriented_pieces;

    static inline void store_color2piece(t_color2oriented_pieces& p_color2pieces, 
                                         const emp_types::t_color_id & p_color_id,
                                         const emp_types::t_oriented_piece & p_piece
					 );

    inline void record_identical_pieces(const emp_types::t_piece_id & p_id1,
					const emp_types::t_piece_id & p_id2
					);

    typedef std::pair<std::pair<emp_types::t_color_id,emp_types::t_color_id>,std::pair<emp_types::t_color_id,emp_types::t_color_id> > t_constraint;
    typedef std::map<emp_types::t_piece_id,std::set<emp_types::t_piece_id> > t_identical_pieces_db;

    /**
       List of pieces
    **/
    const std::vector<emp_piece> & m_pieces;

    /**
       Pieces counters for each kind of piece
    **/
    unsigned int m_nb_pieces[((uint32_t)emp_types::t_kind::CORNER) + 1];

    /**
       Necessary size in bits to represent piece ids
     **/
    unsigned int m_coded_piece_id_size;

    /**
       Necessary size in bits to represent piece ids and missing piece
     **/
    unsigned int m_dumped_piece_id_size;

    /**
       Color Id used to represent border color in the application : max color id + 1
    **/
    unsigned int m_border_color_id;

    /**
       Necessary size in bits to represent color ids including border color ( max color id + 1) and lack of color ( 0 )
     **/
    unsigned int m_color_id_size;

    /**
       Store the max constraint that can be done with available color ids
    **/
    emp_types::t_binary_piece m_max_constraint;

    /**
       List of corner pieces
    **/
    emp_piece_corner* m_corners[4];

    /**
       List of border pieces
    **/
    emp_piece_border ** m_border_pieces;

    /**
       List of center pieces
    **/
    emp_piece ** m_center_pieces;

    /**
       Array storing correspondance from piece id to its index in its kind array
    **/
    unsigned int * m_piece_id2kind_index;

    /**
       Pieces counters for each kind of piece
    **/
    unsigned int m_nb_color_kinds[((uint32_t)emp_types::t_kind::CORNER) + 1];

    /**
       Array storing correspondance from color id to its index in its kind array
    **/
    unsigned int * m_color_id2kind_index;

    /**
       Array storing correspondance from color id to its index in a specified kind array
    **/
    unsigned int ** m_color_id2specific_kind_index;

    /**
       Array storing correspondance from kind color id to color id
     **/
    emp_types::t_color_id ** m_color_kind_index2color_id;

    /**
       Array storing the kind of each color
     **/
    emp_types::t_kind * m_color_kind;

    /**
       Bitfield representation for each kind of piece
     **/
    emp_types::t_binary_piece ** m_binary_pieces;


    /**
	Bitfield whose each bit represents an oriented piece matching the constraint
	used as index
     **/
    emp_types::bitfield * m_binary_constraint_db;

    /**
	Bitfield whose each bit represents an oriented piece identical to the one specified as index
     **/
    emp_types::bitfield ** m_binary_identical_pieces;


    typedef std::map<emp_piece_constraint,std::set<emp_types::t_oriented_piece> > t_constraint_db;
    t_constraint_db ** m_constraint_db;
    typedef std::map<emp_types::t_oriented_piece,std::set<emp_piece_constraint> > t_piece2constraint_db;
    t_piece2constraint_db m_piece2constraint_db;
    std::set<emp_constraint> m_single_constraints;
    t_identical_pieces_db m_identical_pieces_db;

    /**
       Number of occurence for border2center colors
    **/
    std::map<emp_types::t_color_id,unsigned int> m_border2center_colors_nb;

    /**
       All colors
    **/
    t_color_list m_colors;

    /**
       List of colors composing center of puzzle
    **/
    t_color_list m_center_colors;

    /**
       List of colors composing border of puzzle
    **/
    t_color_list m_border_colors;

    /**
       Colors of edge pieces that are related to center pieces
    **/
    t_color_list m_border2center_colors;

    /**
       Colors of corners
    **/
    t_color_list m_corner_colors;
  };

  //----------------------------------------------------------------------------
  inline const unsigned int & emp_piece_db::get_nb_pieces(const emp_types::t_kind & p_kind)const
    {
      assert(p_kind <= emp_types::t_kind::CORNER);
      return m_nb_pieces[(unsigned int)p_kind];
    }

  //----------------------------------------------------------------------------
  emp_piece_db::emp_piece_db(const std::vector<emp_piece> & p_pieces,
			     const unsigned int & p_width,
			     const unsigned int & p_height
			     ):
    m_pieces(p_pieces),
    m_nb_pieces{(p_width - 2) * (p_height - 2),2 * (p_width - 2 + p_height - 2),4},
    m_coded_piece_id_size(0),
    m_dumped_piece_id_size(0),
    m_border_color_id(0),
    m_color_id_size(0),
    m_max_constraint(0),
    m_border_pieces(nullptr),
    m_center_pieces(nullptr),
    m_piece_id2kind_index(new unsigned int[p_pieces.size()]),
    m_nb_color_kinds{0,0,0},
    m_color_id2kind_index(nullptr),
    m_color_id2specific_kind_index(new unsigned int*[3]),
    m_color_kind_index2color_id(new emp_types::t_color_id*[3]),
    m_color_kind(nullptr),
    m_binary_pieces(new emp_types::t_binary_piece*[3]),
    m_binary_constraint_db(nullptr),
    m_binary_identical_pieces(new emp_types::bitfield*[3]),
    m_constraint_db(new t_constraint_db*[3])
      {
        std::cout << "----------------------------------------------" << std::endl;
        std::cout << "Building piece database" << std::endl;

        unsigned int l_nb_edge = (p_width - 1) * p_height + (p_height - 1) * p_width;
        unsigned int l_nb_pieces = p_pieces.size();

        // Creating constraints database
        for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
            l_index <= (unsigned int)emp_types::t_kind::CORNER;
            ++l_index
	    )
          {
            m_constraint_db[l_index] = new t_constraint_db[4 - l_index];
          }

        // Creating corner array
        for(unsigned int l_index = 0;
	    l_index < 4 ;
	    ++l_index
	    )
          {
            m_corners[l_index] = NULL;
          }

	// Create arrays to store binary representation of each piece
	for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
	    l_index <= (unsigned int) emp_types::t_kind::CORNER;
	    ++l_index
	    )
	  {
	    m_binary_pieces[l_index] = new emp_types::t_binary_piece[4 * m_nb_pieces[l_index]];
	    m_color_id2specific_kind_index[l_index] = nullptr;
	    m_color_kind_index2color_id[l_index] = nullptr;
	  }
	m_border_pieces = new emp_piece_border*[m_nb_pieces[(unsigned int)emp_types::t_kind::BORDER]];
	m_center_pieces = new emp_piece*[m_nb_pieces[(unsigned int)emp_types::t_kind::CENTER]];

	// Auto similarity counters
        unsigned int l_nb_auto_similarity[((uint32_t)emp_piece::t_auto_similarity::SIMILAR) + 1] = {0,0,0};
        std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id> l_auto_similarities;

        // Counting auto similarities
        for(auto l_iter : p_pieces)
          {
            ++(l_nb_auto_similarity[((unsigned int)l_iter.get_auto_similarity())]);
            l_auto_similarities.insert(std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::value_type(l_iter.get_auto_similarity(),l_iter.get_id()));
          }

        // Looking for identical pieces
        // Examining pieces
        for(auto l_iter : p_pieces)
          {
            for(auto l_iter_bis : p_pieces)
              {
                if(l_iter_bis.get_id() == l_iter.get_id()) break;
                if(l_iter == l_iter_bis)
                  {
                    record_identical_pieces(l_iter.get_id(),l_iter_bis.get_id());
                    record_identical_pieces(l_iter_bis.get_id(),l_iter.get_id());
                  }
              }
          }

        // Store for each color which pieces contains this color
        t_color2oriented_pieces l_color2pieces;

        // Examining pieces:
        // _ Fill color lists
        // _ Fill database which store correspondancie from color to pieces
        // _ Compute constraints related to each piece
        unsigned int l_corner_index = 0;
        unsigned int l_border_index = 0;
        unsigned int l_center_index = 0;
        for(auto l_iter : p_pieces)
          {
            switch(l_iter.get_kind())
              {
              case emp_types::t_kind::CENTER:
		m_center_pieces[l_center_index] = new emp_piece(l_iter);
		m_piece_id2kind_index[l_iter.get_id() - 1] = l_center_index;
		++l_center_index;
                for(unsigned int l_index = (unsigned int)emp_types::t_orientation::NORTH ;
                    l_index <= (unsigned int)emp_types::t_orientation::WEST; 
                    ++l_index
		    )
                  {
                    // Fill color lists
                    emp_types::t_color_id l_color = l_iter.get_color((emp_types::t_orientation )l_index);
                    m_center_colors.insert(l_color);
                    m_colors.insert(l_color);

                    // Fill database which store correspondancie from color to pieces
                    store_color2piece(l_color2pieces,l_color,emp_types::t_oriented_piece(l_iter.get_id(),(emp_types::t_orientation )l_index));
                  }
                break;
              case emp_types::t_kind::BORDER:
                {
                  emp_piece_border * l_border = new emp_piece_border(l_iter);
                  m_border_pieces[l_border_index] = l_border;
		  m_piece_id2kind_index[l_iter.get_id() - 1] = l_border_index;
		  ++l_border_index;

		  std::map<emp_types::t_color_id,unsigned int>::iterator l_border_color_counter_iter = m_border2center_colors_nb.find(l_border->get_center_color());
		  if(m_border2center_colors_nb.end() == l_border_color_counter_iter)
		    {
		      m_border2center_colors_nb.insert(std::map<emp_types::t_color_id,unsigned int>::value_type(l_border->get_center_color(),1));
		    }
		  else
		    {
		      ++(l_border_color_counter_iter->second);
		    }
                  // Fill color lists
                  std::pair<emp_types::t_color_id,emp_types::t_color_id> l_piece_border_colors = l_border->get_border_colors();
                  m_colors.insert(l_piece_border_colors.first);
                  m_colors.insert(l_piece_border_colors.second);
                  m_border_colors.insert(l_piece_border_colors.first);
                  m_border_colors.insert(l_piece_border_colors.second);
                  m_center_colors.insert(l_border->get_center_color());
                  m_border2center_colors.insert(l_border->get_center_color());
                
                  // Fill database which store correspondancie from color to pieces
                  const std::pair<emp_types::t_orientation,emp_types::t_orientation> & l_colors_orientations = l_border->get_colors_orientations();
                  store_color2piece(l_color2pieces,l_piece_border_colors.first,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.first));
                  store_color2piece(l_color2pieces,l_piece_border_colors.second,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.second));
                  store_color2piece(l_color2pieces,l_border->get_center_color(),emp_types::t_oriented_piece(l_iter.get_id(),l_border->get_center_orientation()));
                }
                break;
              case emp_types::t_kind::CORNER:
                {
                  emp_piece_corner* l_corner = new emp_piece_corner(l_iter);
                  m_corners[l_corner_index] = l_corner;
		  m_piece_id2kind_index[l_iter.get_id() - 1] = l_corner_index;
                  ++l_corner_index;

                  // Fill color lists
                  std::pair<emp_types::t_color_id,emp_types::t_color_id> l_piece_border_colors = l_corner->get_border_colors();
                  m_colors.insert(l_piece_border_colors.first);
                  m_colors.insert(l_piece_border_colors.second);
                  m_border_colors.insert(l_piece_border_colors.first);
                  m_border_colors.insert(l_piece_border_colors.second);
                  m_corner_colors.insert(l_piece_border_colors.first);
                  m_corner_colors.insert(l_piece_border_colors.second);

                  // Fill database which store correspondancie from color to pieces
                  const std::pair<emp_types::t_orientation,emp_types::t_orientation> & l_colors_orientations = l_corner->get_colors_orientations();
                  store_color2piece(l_color2pieces,l_piece_border_colors.first,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.first));
                  store_color2piece(l_color2pieces,l_piece_border_colors.second,emp_types::t_oriented_piece(l_iter.get_id(),l_colors_orientations.second));
                }
                break;
              default:
                throw quicky_exception::quicky_logic_exception("Unsupported kind of piece \""+emp_types::kind2string(l_iter.get_kind())+"\"",__LINE__,__FILE__);
              }
          }

        // Pieces ID are coded as ID - 1 so first piece Id = 1 is coded 0
        m_coded_piece_id_size = compute_nb_bits(l_nb_pieces - 1);
        m_dumped_piece_id_size = compute_nb_bits(l_nb_pieces);

        unsigned int l_max_color_id = 0;
        for(auto l_iter: m_colors)
          {
            if(l_iter > l_max_color_id)
              {
                l_max_color_id = l_iter;
              }
          }
        // Color Id do not start a zero which is reserved for no color and l_max_color_id + 1 is used to represent border
	m_border_color_id = l_max_color_id + 1;
        m_color_id_size = compute_nb_bits(m_border_color_id);

	// Alocate array to store for each color the corresponding index in color kind
	assert(m_border_color_id);
	m_color_id2kind_index = new unsigned int[m_border_color_id + 1];
	for(unsigned int l_index = 0;
	    l_index <= m_border_color_id;
	    ++l_index
	    )
	  {
	    m_color_id2kind_index[l_index] = 0;
	  }

	assert(m_center_colors.size() == m_border2center_colors.size());
	m_nb_color_kinds[(unsigned int)emp_types::t_kind::CORNER] = m_corner_colors.size();
	m_nb_color_kinds[(unsigned int)emp_types::t_kind::BORDER] = m_border_colors.size();
	m_nb_color_kinds[(unsigned int)emp_types::t_kind::CENTER] = m_border2center_colors.size();


	m_color_kind = new emp_types::t_kind[m_border_color_id];


	    for(unsigned int l_kind_index = (unsigned int)emp_types::t_kind::CENTER;
		l_kind_index <= (unsigned int) emp_types::t_kind::CORNER;
		++l_kind_index
		)
	      {
		m_color_id2specific_kind_index[l_kind_index] = new unsigned int[m_border_color_id];
	      }

	for(unsigned int l_index = 0;
	    l_index < m_border_color_id;
	    ++l_index)
	  {
	    m_color_kind[l_index] = emp_types::t_kind::UNDEFINED;
	    m_color_id2kind_index[l_index] = 0xDEAD;
	    for(unsigned int l_kind_index = (unsigned int)emp_types::t_kind::CENTER;
		l_kind_index <= (unsigned int) emp_types::t_kind::CORNER;
		++l_kind_index
		)
	      {
		m_color_id2specific_kind_index[l_kind_index][l_index] = 0xDEAD;
	      }
	  }

	unsigned int l_corner_color_index = 0;
	m_color_kind_index2color_id[(unsigned int)emp_types::t_kind::CORNER] = new emp_types::t_color_id[m_corner_colors.size()];
	for(auto l_iter: m_corner_colors)
	  {
	    m_color_id2kind_index[l_iter] = l_corner_color_index;
	    m_color_id2specific_kind_index[(unsigned int)emp_types::t_kind::CORNER][l_iter] = l_corner_color_index;
	    m_color_kind_index2color_id[(unsigned int)emp_types::t_kind::CORNER][l_corner_color_index] = l_iter;
	    m_color_kind[l_iter] = emp_types::t_kind::CORNER;
	    ++l_corner_color_index;
	  }

	unsigned int l_border_color_index = 0;
	m_color_kind_index2color_id[(unsigned int)emp_types::t_kind::BORDER] = new emp_types::t_color_id[m_border_colors.size()];
	for(auto l_iter: m_border_colors)
	  {
	    m_color_kind_index2color_id[(unsigned int)emp_types::t_kind::BORDER][l_border_color_index] = l_iter;
	    m_color_id2specific_kind_index[(unsigned int)emp_types::t_kind::BORDER][l_iter] = l_border_color_index;
	    if(emp_types::t_kind::UNDEFINED == m_color_kind[l_iter])
	      {
		m_color_kind[l_iter] = emp_types::t_kind::BORDER;
		m_color_id2kind_index[l_iter] = l_border_color_index;
	      }
	    ++l_border_color_index;
	  }

	unsigned int l_center_color_index = 0;
	m_color_kind_index2color_id[(unsigned int)emp_types::t_kind::CENTER] = new emp_types::t_color_id[m_center_colors.size()];
	for(auto l_iter: m_center_colors)
	  {
	    m_color_kind_index2color_id[(unsigned int)emp_types::t_kind::CENTER][l_center_color_index] = l_iter;
	    m_color_id2specific_kind_index[(unsigned int)emp_types::t_kind::CENTER][l_iter] = l_center_color_index;
	    if(emp_types::t_kind::UNDEFINED == m_color_kind[l_iter])
	      {
		m_color_kind[l_iter] = emp_types::t_kind::CENTER;
		m_color_id2kind_index[l_iter] = l_center_color_index;
	      }
	    ++l_center_color_index;
	  }


	// Compute max constraint code that can be coded with available colors
        m_max_constraint = 0;
	for(unsigned int l_index = 0 ;
	    l_index < 4 ;
	    ++l_index
	    )
	  {
	    m_max_constraint = m_max_constraint << m_color_id_size;
	    m_max_constraint |= (l_index != 2 ? m_border_color_id :  l_max_color_id);
	  }

	// Compute binary representations
	unsigned int l_index_by_kind[((uint32_t)emp_types::t_kind::CORNER) + 1] = {0,0,0};
	for(auto l_iter : p_pieces)
	  {
            unsigned int l_kind_index = (unsigned int)l_iter.get_kind();
	    for(unsigned int l_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
		l_orient_index <= (unsigned int)emp_types::t_orientation::WEST;
		++l_orient_index
		)
	      {
		unsigned int l_extended_index = (l_index_by_kind[l_kind_index] << 2 ) + l_orient_index;
		m_binary_pieces[l_kind_index][l_extended_index] = l_iter.get_bitfield_representation((emp_types::t_orientation)l_orient_index,m_coded_piece_id_size,m_color_id_size,m_border_color_id);
	      }
            ++l_index_by_kind[l_kind_index];
	  }


	// Create empty binary constraints
        // The placement bew will be used just after to initialised possible constraints
        m_binary_constraint_db = new emp_types::bitfield[m_max_constraint + 1];

        for(unsigned int l_color1 = 0;
	    l_color1 <= m_border_color_id;
	    ++l_color1
	    )
          {
            unsigned int l_nb_border = l_color1 != m_border_color_id ? 0 : 1;
            unsigned int l_constraint_index1 = l_color1 << m_color_id_size;
            for(unsigned int l_color2 = 0;
		l_color2 <= m_border_color_id;
		++l_color2
		)
              {
                if(l_color2 == m_border_color_id)
                  {
                    ++l_nb_border;
                  }
                unsigned int l_constraint_index2 = (l_constraint_index1 | l_color2 ) << m_color_id_size;
                for(unsigned int l_color3 = 0 ;
		    l_color3 <= m_border_color_id ;
		    ++l_color3
		    )
                  {
                    if(l_color3 == m_border_color_id)
                      {
                        ++l_nb_border;
                      }
                    unsigned int l_constraint_index3 = (l_constraint_index2 | l_color3 ) << m_color_id_size;
                    for(unsigned int l_color4 = 0;
			l_color4 <= m_border_color_id ;
			++l_color4
			)
                      {
                        if(l_color4 == m_border_color_id)
                          {
                            ++l_nb_border;
                          }
                        if(l_nb_border <= 2)
                          {
                            unsigned int l_constraint_index4 = l_constraint_index3 | l_color4 ;
                            new( &(m_binary_constraint_db[l_constraint_index4]))emp_types::bitfield(4 * m_nb_pieces[l_nb_border]);
                          }
                      }
                    --l_nb_border;
                    }
                --l_nb_border;
              }
            --l_nb_border;
          }

        // Compute constraints related to each piece
        for(auto l_iter : p_pieces)
          {
            compute_constraints(l_iter);
	    compute_binary_constraints(l_iter);
          }


	// Create empty bitfield for each kind of piece
        for(unsigned int l_kind_index = (unsigned int)emp_types::t_kind::CENTER;
            l_kind_index <= (unsigned int)emp_types::t_kind::CORNER;
            ++l_kind_index
	    )
          {
            m_binary_identical_pieces[l_kind_index] = (emp_types::bitfield*)operator new[](sizeof(emp_types::bitfield) * m_nb_pieces[l_kind_index] * 4);

	    // Call constructor with correct bitfield size
	    for(unsigned int l_constraint_index = 0;
		l_constraint_index < 4 * m_nb_pieces[l_kind_index];
		++l_constraint_index
		)
	      {
		new( &(m_binary_identical_pieces[l_kind_index][l_constraint_index]))emp_types::bitfield(4 * m_nb_pieces[l_kind_index],true);
                m_binary_identical_pieces[l_kind_index][l_constraint_index].set(0,1,l_constraint_index);
	      }
          }

        // Record identical pieces binary filters
        if(m_identical_pieces_db.size())
          {
            for(auto l_iter: m_identical_pieces_db)
              {
		// Get pieces characteristics
                emp_piece l_piece = get_piece(l_iter.first);
                emp_types::t_kind l_kind = l_piece.get_kind();
                unsigned int l_kind_index = m_piece_id2kind_index[l_iter.first - 1];

		for(auto l_iter_second : l_iter.second)
		  {
		    // Get chacteristics of all identical pieces
		    emp_piece l_piece_second = get_piece(l_iter_second);
		    unsigned int l_kind_index_second = m_piece_id2kind_index[l_iter_second - 1];

		    // Check for which orientation they are identic
		    for(unsigned int l_color_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
			l_color_orient_index <= (unsigned int)emp_types::t_orientation::WEST;
			++l_color_orient_index
			)
		      {
			if(l_piece.compare_to(l_piece_second,(emp_types::t_orientation) l_color_orient_index))
			  {
			    for(unsigned int l_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
				l_orient_index <= (unsigned int)emp_types::t_orientation::WEST;
				++l_orient_index
				)
			      {
				unsigned int l_extended_index = (l_kind_index << 2) + l_orient_index;
				unsigned int l_extended_index_second = (l_kind_index_second << 2) + ((l_color_orient_index + l_orient_index) % 4);
				m_binary_identical_pieces[(unsigned int)l_kind][l_extended_index].set(0,1,l_extended_index_second);
			      }
			  }
		      }
		  }
              }
#ifndef HANDLE_IDENTICAL_PIECES
            std::cout << "There are some identical pieces, please recompile with flag HANDLE_IDENTICAL_PIECES" << std::endl ;
            exit(-1);
#endif // HANDLE_IDENTICAL_PIECES
          }
#ifdef HANDLE_IDENTICAL_PIECES
        else
          {
            std::cout << "There are no identical pieces, please recompile without flag HANDLE_IDENTICAL_PIECES" << std::endl ;
            exit(-1);
          }
#endif // HANDLE_IDENTICAL_PIECES

        // Check color repartition on pieces
        unsigned int l_total = 0;
        bool l_error = false;
        for(auto l_iter: m_colors)
          {
            unsigned int l_nb = l_color2pieces.find(l_iter)->second.size();
            if(l_nb % 2) 
              {
                l_error = true;
                std::cout << "ERROR : Color = " << l_iter << " appears on " << l_nb << " pieces edge" << std::endl ;
              }
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


        // Compute links between pieces
        std::vector<emp_link> l_links;
        typedef std::multimap<emp_types::t_piece_id,emp_link> t_piece2links;
        t_piece2links l_piece2links;
        typedef std::multimap<emp_types::t_oriented_piece,emp_link> t_piece_edge_2links;
        t_piece_edge_2links l_piece_edge_2links;
        for(unsigned int l_index1 = 0;
	    l_index1 < p_pieces.size();
	    ++l_index1
	    )
          {
            for(unsigned int l_orient_index1 = (unsigned int)emp_types::t_orientation::NORTH ;
                l_orient_index1 <= (unsigned int)emp_types::t_orientation::WEST; 
                ++l_orient_index1
		)
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

        unsigned int l_total_pieces_links = 0;
        for(auto l_iter : p_pieces)
          {
            unsigned int l_nb = l_piece2links.count(l_iter.get_id());
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
                ++l_orient_index
		)
              {
                emp_types::t_piece_id l_id = l_iter.get_id();
                emp_types::t_orientation l_orientation = (emp_types::t_orientation)l_orient_index;
                if(l_iter.get_color(l_orientation))
                  {
                    unsigned int l_nb = l_piece_edge_2links.count(emp_types::t_oriented_piece(l_id,l_orientation));
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


        std::cout << "Width = " << p_width << std::endl ;
        std::cout << "Heigth = " << p_height << std::endl ;
        std::cout << "Nb edges : " << l_nb_edge << std::endl ;
        std::cout << "Piece id coded size = " << m_coded_piece_id_size << " bits" << std::endl ;
        std::cout << "Color id coded size = " << m_color_id_size << " bits" << std::endl ;
        std::cout << "Max color id  = " << l_max_color_id << " (0x" << std::hex << l_max_color_id << std::dec << ")" <<  std::endl ;
        std::cout << "Max code constraint  = " << m_max_constraint << " (0x" << std::hex << m_max_constraint << std::dec << ")" << std::endl ;

        // Display number of pieces ok each kind
        for(unsigned int l_index = ((uint32_t)emp_types::t_kind::CENTER) ;
            l_index <= ((uint32_t)emp_types::t_kind::CORNER);
            ++l_index
	    )
          {
            std::cout << "\t" << emp_types::kind2string((emp_types::t_kind)l_index) << "\t: " << m_nb_pieces[l_index] << std::endl ;
          }

        // Display number of auto similar pieces
        std::cout << "Number of pieces depending on auto_similarity :" << std::endl ;
        for(unsigned int l_index = ((uint32_t)emp_piece::t_auto_similarity::NONE) ;
            l_index <= ((uint32_t)emp_piece::t_auto_similarity::SIMILAR);
            ++l_index
	    )
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

        print_list("Colors",m_colors);
        print_list("Center colors",m_center_colors);
        print_list("Border colors",m_border_colors);
        print_list("Corner colors",m_corner_colors);
        print_list("Border to center colors",m_border2center_colors);

        std::cout << "Color repartition on pieces :" <<std::endl ;
        for(auto l_iter: m_colors)
          {
            unsigned int l_nb = l_color2pieces.find(l_iter)->second.size();
            std::cout << "Color = " << l_iter << " appears on " << l_nb << " pieces edge" << std::endl ;
          }

        std::cout << "Nb links = " << l_links.size() << std::endl ;
#if 0
        // Display number of links per pieces
        for(auto l_iter : p_pieces)
          {
            unsigned int l_nb = l_piece2links.count(l_iter.get_id());
            std::cout << emp_types::kind2string(l_iter.get_kind()) << "\t: Piece = " << l_iter.get_id() << " has " << l_nb << " links" << std::endl ;
          }

        // Display number of links per pieces edges
        for(auto l_iter : p_pieces)
          {
            for(unsigned int l_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
                l_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
                ++l_orient_index
		)
              {
                emp_types::t_piece_id l_id = l_iter.get_id();
                emp_types::t_orientation l_orientation = (emp_types::t_orientation)l_orient_index;
                if(l_iter.get_color(l_orientation))
                  {
                    unsigned int l_nb = l_piece_edge_2links.count(emp_types::t_oriented_piece(l_id,l_orientation));
                    std::cout << emp_types::kind2string(l_iter.get_kind()) << "\t: Piece = " << l_id << " Edge = " << emp_types::orientation2string(l_orientation) << " has " << l_nb << " links" << std::endl ;
                  }
              }
          }

        // Display pieces matching constraints
        for(unsigned int l_kind_index = (unsigned int)emp_types::t_kind::CENTER;
            l_kind_index <= (unsigned int)emp_types::t_kind::CORNER;
            ++l_kind_index
	    )
          {
            for(unsigned int l_constraint_size_index = 0 ;
		l_constraint_size_index < 4 - l_kind_index;
		++l_constraint_size_index
		)
              {
                for(auto l_constraint_iter : m_constraint_db[l_kind_index][l_constraint_size_index])
                  {
                    std::cout << emp_types::kind2string((emp_types::t_kind)l_kind_index) << " pieces matching size " << l_constraint_size_index + l_kind_index + 1 << " constraint : " << l_constraint_iter.first << ": " << l_constraint_iter.second.size() << std::endl ;
                    for(auto l_oriented_piece_iter : l_constraint_iter.second)
                      {
                        std::cout << "\t" << emp_types::orientation2string(l_oriented_piece_iter.second) << " oriented piece " << get_piece(l_oriented_piece_iter.first) << std::endl ;
                      }
                  }
              }
          }

        // Display constraints per piece
        for(auto l_iter : m_piece2constraint_db)
          {
            std::cout << "Computed constraints for " << emp_types::orientation2string(l_iter.first.second) << " oriented piece " << get_piece(l_iter.first.first) << std::endl ;
            for(auto l_iter2 : l_iter.second)
              {
                std::cout << "\t" << l_iter2 << std::endl ;
              }
          }

#endif // 0

	for(unsigned int l_index = 0;
	    l_index < m_border_color_id;
	    ++l_index)
	  {
	    std::cout << "Color[" << l_index << "] : " << emp_types::kind2string(m_color_kind[l_index]) ;
	    if(l_index)
	      {
		std::cout << "[" << get_color_kind_index(l_index) << "]" ;
	      }
	    std::cout << std::endl ;
	  }

        for(unsigned int l_kind = (unsigned int)emp_types::t_kind::CENTER;
            l_kind <= (unsigned int)emp_types::t_kind::CORNER;
            ++l_kind
	    )
          {
	    for(unsigned int l_kind_index = 0;
		l_kind_index < m_nb_color_kinds[l_kind];
		++l_kind_index
		)
	      {
		std::cout << emp_types::kind2string((emp_types::t_kind)l_kind) << "[" << l_kind_index << "] = " << get_color_id((emp_types::t_kind) l_kind, l_kind_index) << std::endl;
	      }
	  }

	// Display number of occurence for border colors
	std::cout << "Border2center colors occurences : " << std::endl ;
	unsigned int l_remaining = m_nb_pieces[(unsigned int)emp_types::t_kind::BORDER];
	uint64_t l_total_combi = 1;
	for(auto l_iter: m_border2center_colors_nb)
	  {
	    uint64_t l_numerator = l_remaining;
	    uint64_t l_denominator = 1;
	    for(unsigned int l_index = 1 ;
		l_index < l_iter.second ;
		++l_index
		)
	      {
		l_numerator *= l_remaining - l_index;
		l_denominator *= (l_index + 1);
	      }
	    uint64_t l_combi = l_numerator / l_denominator;
	    l_total_combi *= l_combi;
	    std::cout << "Border2center color " << l_iter.first << " is present on " <<   l_iter.second << " border pieces\t" << l_remaining << "C" << l_iter.second << " = " << l_combi << std::endl;
	    l_remaining -= l_iter.second;
	  }
	std::cout << "Total combination " << l_total_combi << std::endl ;

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
    const unsigned int & emp_piece_db::get_color_id_size(void)const
      {
	return m_color_id_size;
      }

    //----------------------------------------------------------------------------
    const unsigned int & emp_piece_db::get_piece_id_size(void)const
      {
        return m_coded_piece_id_size;
      }

    //----------------------------------------------------------------------------
    const unsigned int & emp_piece_db::get_dumped_piece_id_size(void)const
      {
        return m_dumped_piece_id_size;
      }

    //----------------------------------------------------------------------------
    const unsigned int &  emp_piece_db::get_border_color_id(void)const
      {
	return m_border_color_id;
      }

    //----------------------------------------------------------------------------
    const emp_types::t_binary_piece & emp_piece_db::get_piece(const emp_types::t_kind & p_kind,
							      const unsigned int & p_id
							      ) const
      {
	return m_binary_pieces[(unsigned int)p_kind][p_id];
      }

    //----------------------------------------------------------------------------
    void emp_piece_db::get_pieces(const emp_types::t_kind & p_kind,
                                  const std::set<emp_constraint> & p_constraints,
                                  std::vector<emp_types::t_oriented_piece> & p_pieces
				  )const
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
    const emp_types::bitfield & emp_piece_db::get_pieces(const emp_types::t_binary_piece & p_constraint)const
      {
        assert(p_constraint <= m_max_constraint);
        return m_binary_constraint_db[p_constraint];
      }

    //----------------------------------------------------------------------------
    const emp_types::bitfield & emp_piece_db::get_get_binary_identical_pieces(const emp_types::t_kind & p_kind,
									      const emp_types::t_piece_id & p_kind_id
									      )const
      {
        assert(p_kind_id < 4 * m_nb_pieces[(unsigned int)p_kind]);
        return m_binary_identical_pieces[(unsigned int)p_kind][p_kind_id];
      }

    //----------------------------------------------------------------------------
    void emp_piece_db::compute_binary_constraints(const emp_piece & p_piece)
    {
      unsigned int l_increment = pow(2,(unsigned int) p_piece.get_auto_similarity());
      // Compute for different orientations of piece
      for(unsigned int l_piece_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
          l_piece_orient_index <= (unsigned int)emp_types::t_orientation::WEST;
          l_piece_orient_index += l_increment
	  )
        {
	  unsigned int l_extended_kind_index = (m_piece_id2kind_index[p_piece.get_id() - 1] << 2) + l_piece_orient_index;

          // Compute the various constraints
          for(unsigned int l_constraint_squeleton = 0;
	      l_constraint_squeleton <= 15;
	      ++l_constraint_squeleton
	      )
            {
	      uint32_t l_constraint = 0;
              unsigned int l_nb_border = 0;
              for(unsigned int l_border_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
                  l_border_orient_index <= (unsigned int)emp_types::t_orientation::WEST;
                  ++l_border_orient_index
		  )
                {
		  // Check if this side of piece is meaningfull for this constraint skeleton
                  if((1 << l_border_orient_index) & l_constraint_squeleton)
                    {
		      emp_types::t_color_id l_color_id = p_piece.get_color((emp_types::t_orientation)l_border_orient_index,(emp_types::t_orientation)l_piece_orient_index);
                      if(!l_color_id)
                        {
                          l_color_id = m_border_color_id;
                          ++l_nb_border;
                        }
                      // Orientation for constraint is inverted : North constraint will be given by South of upper piece
                      l_constraint |= l_color_id << (m_color_id_size * ((l_border_orient_index + 2) % 4));
                    }
                }
	      assert(l_constraint < m_max_constraint);
              if(l_nb_border == (unsigned int) p_piece.get_kind())
                {
                  m_binary_constraint_db[l_constraint].set(1,1,l_extended_kind_index);
                }
	    }
	}
    }

    //----------------------------------------------------------------------------
    void emp_piece_db::compute_constraints(const emp_piece & p_piece)
    {
      unsigned int l_increment = pow(2,(unsigned int) p_piece.get_auto_similarity());
      // Compute for different orientations of piece
      for(unsigned int l_piece_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
          l_piece_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
          l_piece_orient_index += l_increment
	  )
        {
          emp_types::t_oriented_piece l_oriented_piece(p_piece.get_id(),(emp_types::t_orientation)l_piece_orient_index);
          std::array<const emp_constraint*,4> l_oriented_piece_constraints;
          // Fill array with oriented colors
          for(unsigned int l_border_orient_index = (unsigned int)emp_types::t_orientation::NORTH ;
              l_border_orient_index <= (unsigned int)emp_types::t_orientation::WEST; 
              ++l_border_orient_index
	      )
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
          for(unsigned int l_index = 1 ;
	      l_index <= 15;
	      ++l_index)
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
                                  const std::set<unsigned int> & p_list
				  )
    {
      std::cout << p_name << " : " << p_list.size() ;
      if(p_list.size())
        {
          std::cout << " = { " << *(p_list.begin());
          for(std::set<unsigned int>::const_iterator l_iter = ++(p_list.begin());
              l_iter != p_list.end();
              ++l_iter
	      )
            {
              std::cout << ", " << *l_iter ;
            }
          std::cout << "}" ;
        }
      std::cout << std::endl ;
    }

    //------------------------------------------------------------------------------
    void emp_piece_db::print_auto_similarities(const emp_piece::t_auto_similarity & p_similarity,
                                               const std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id> & p_auto_similarities
					       )
    {
      std::pair<std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::const_iterator,std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::const_iterator> l_values = p_auto_similarities.equal_range(p_similarity);
      if(l_values.first != l_values.second)
        {
          std::cout << "List of pieces which are " << emp_piece::auto_similarity2string(p_similarity) << ":" << std::endl ;
          std::cout  << "{ " << l_values.first->second ;
          for(std::multimap<emp_piece::t_auto_similarity,emp_types::t_piece_id>::const_iterator l_iter = ++(l_values.first) ;
              l_values.first != l_values.second;
              ++l_values.first
	      )
            {
              std::cout << ", " << l_iter->second ;
            }
          std::cout << "}"  << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void emp_piece_db::store_color2piece(t_color2oriented_pieces& p_color2pieces, 
                                         const emp_types::t_color_id & p_color_id,
                                         const emp_types::t_oriented_piece & p_piece
					 )
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
                                               const emp_types::t_piece_id & p_id2
					       )
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
        for(unsigned int l_index = 0 ;
	    l_index < 4 ;
	    ++l_index
	    )
          {
            delete m_corners[l_index];
          }

	for(unsigned int l_index = 0 ;
	    l_index < m_nb_pieces[(unsigned int)emp_types::t_kind::CENTER];
	    ++l_index
	    )
	  {
	    delete m_center_pieces[l_index];
	  }
	delete[] m_center_pieces;
	for(unsigned int l_index = 0 ;
	    l_index < m_nb_pieces[(unsigned int)emp_types::t_kind::BORDER];
	    ++l_index
	    )
	  {
	    delete m_border_pieces[l_index];
	  }
	delete[] m_border_pieces;

	for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
	    l_index <= (unsigned int) emp_types::t_kind::CORNER;
	    ++l_index
	    )
	  {
	    delete[] m_binary_pieces[l_index];
	  }

        for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
            l_index <= (unsigned int)emp_types::t_kind::CORNER;
            ++l_index
	    )
          {
            delete[] m_constraint_db[l_index];
          }
        delete[] m_constraint_db;
        delete[] m_binary_constraint_db;


	for(unsigned int l_kind_index = (unsigned int)emp_types::t_kind::CENTER;
	    l_kind_index <= (unsigned int) emp_types::t_kind::CORNER;
	    ++l_kind_index
	    )
	  {
	    for(unsigned int l_constraint_index = 0 ;
		l_constraint_index < m_nb_pieces[(unsigned int)l_kind_index] * 4;
		++l_constraint_index
		)
	      {
		m_binary_identical_pieces[l_kind_index][l_constraint_index].~quicky_bitfield();
	      }
	    operator delete[] ((void*) (m_binary_identical_pieces[l_kind_index]));
	  }
        delete[] m_binary_identical_pieces;

	delete[] m_binary_pieces;
        delete[] m_piece_id2kind_index;
	delete[] m_color_id2kind_index;
        for(unsigned int l_index = (unsigned int)emp_types::t_kind::CENTER;
            l_index <= (unsigned int)emp_types::t_kind::CORNER;
            ++l_index
	    )
          {
            delete[] m_color_kind_index2color_id[l_index];
	    delete[] m_color_id2specific_kind_index[l_index];
          }
	delete[] m_color_id2specific_kind_index;
	delete[] m_color_kind_index2color_id;
	delete[] m_color_kind;
      }

    //----------------------------------------------------------------------------
    unsigned int emp_piece_db::compute_nb_bits(unsigned int p_value)
    {
      unsigned int l_nb_bits = 0;
      while(p_value)
        {
          ++l_nb_bits;
          p_value = p_value >> 1;
        }
      return l_nb_bits;
    }

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
    const unsigned int emp_piece_db::get_kind_index(const emp_types::t_piece_id & p_id)const
    {
      assert(p_id <= m_pieces.size());
      assert(p_id);
      return m_piece_id2kind_index[p_id - 1];
    }

    //----------------------------------------------------------------------------
    const unsigned int emp_piece_db::get_color_kind_index(const emp_types::t_color_id & p_id)const
    {
      assert(p_id < m_border_color_id);
      assert(0xDEAD != m_color_id2kind_index[p_id]);
      return m_color_id2kind_index[p_id];
    }
    //----------------------------------------------------------------------------
    const unsigned int emp_piece_db::get_color_kind_index(const emp_types::t_color_id & p_id,
							  const emp_types::t_kind & p_kind
							  )const
    {
      assert(p_id < m_border_color_id);
      assert(emp_types::t_kind::UNDEFINED != p_kind);
      assert(0xDEAD != m_color_id2specific_kind_index[(unsigned int)p_kind][p_id]);
      return m_color_id2specific_kind_index[(unsigned int)p_kind][p_id];
    }

    //----------------------------------------------------------------------------
    const emp_types::kind emp_piece_db::get_color_kind(const emp_types::t_color_id & p_id)const
    {
      assert(m_border_color_id);
      assert(p_id <= m_border_color_id);
      return m_color_kind[p_id];
    }

    //----------------------------------------------------------------------------
    const emp_types::t_color_id emp_piece_db::get_color_id(const emp_types::t_kind & p_kind,
							   const unsigned int & p_index
							   ) const
    {
      assert(p_kind < emp_types::t_kind::UNDEFINED);
      assert(p_index < m_nb_color_kinds[(unsigned int)p_kind]);
      return m_color_kind_index2color_id[(unsigned int)p_kind][p_index];
    }

    //----------------------------------------------------------------------------
    const std::map<emp_types::t_color_id,unsigned int> & emp_piece_db::get_border2center_colors_nb(void)const
    {
      return m_border2center_colors_nb;
    }

    //--------------------------------------------------------------------------
    const emp_piece_db::t_color_list & emp_piece_db::get_colors(void) const
    {
      return m_colors;
    }

    //--------------------------------------------------------------------------
    const emp_piece_db::t_color_list & emp_piece_db::get_center_colors(void) const
    {
      return m_center_colors;
    }

    //--------------------------------------------------------------------------
    const emp_piece_db::t_color_list & emp_piece_db::get_border_colors(void) const
    {
      return m_border_colors;
    }

    //--------------------------------------------------------------------------
    const emp_piece_db::t_color_list & emp_piece_db::get_border2center_colors(void) const
    {
      return m_border2center_colors;
    }

    //--------------------------------------------------------------------------
    const emp_piece_db::t_color_list & emp_piece_db::get_corner_colors(void) const
    {
      return m_corner_colors;
    }
}
#endif // EMP_PIECE_DB_H
// EOF
