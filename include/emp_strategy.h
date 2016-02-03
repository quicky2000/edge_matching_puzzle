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

#ifndef EMP_STRATEGY_H
#define EMP_STRATEGY_H

#include "emp_piece_db.h"
#include "emp_position_strategy.h"
#include "emp_strategy_generator.h"
#include "emp_situation_binary_dumper.h"
#include "feature_if.h"
#include "emp_gui.h"
#include "emp_web_server.h"

#define WEBSERVER
namespace edge_matching_puzzle
{
  /**
     The strategy objects contains all precomputed information necessary to
     determine all transitions at each step of the strategy
  **/
  class emp_strategy: public feature_if
  {
  public:
    // Type representing color constraint
    typedef emp_types::t_binary_piece t_constraint;

    inline emp_strategy(const emp_strategy_generator & p_generator,
                        const emp_piece_db & p_piece_db,
                        emp_gui & p_gui,
                        const emp_FSM_info & p_FSM_info,
			const std::string & p_file_name);

    inline void run(void);


    inline ~emp_strategy(void);

#ifdef WEBSERVER
    inline void pause(void);
    inline void restart(void);
    inline bool is_paused(void)const;
    inline void send_info(uint64_t & p_nb_situations,
                          uint64_t & p_nb_sokutions,
                          unsigned int & p_shift,
                          emp_types::t_binary_piece * p_pieces,
                          const emp_FSM_info & p_FSM_info)const;
#endif // WEBSERVER
  private:
    /**
       To compute bitfield representation needed when dumping infile
    **/
    inline void compute_bin_id(emp_types::bitfield & p_bitfield)const;

    /**
       Comput real available transitions for given index by taking account 
       contraints and available pieces
     **/
    inline void compute_available_transitions(const unsigned int & p_index);

    /**
       Display current strategy state
    **/
    inline void display_on_gui(const unsigned int & p_index);

    const emp_piece_db & m_piece_db;

    /**
       Number of positions strategy
    **/
    const unsigned int m_size;

    /**
       Array t store position strategies
    **/
    emp_position_strategy * m_positions_strategy;

    /**
       Array to store for each kind of position the available pieces of previous position with same kind
    **/
    emp_types::bitfield const * m_previous_available_pieces[3];

    /**
       To bootstrap with all available centers
    **/
    emp_types::bitfield m_centers;

    /**
       To bootstrap with all available borders
    **/
    emp_types::bitfield m_borders;

    /**
       To bootstrap with all available corners
    **/
    emp_types::bitfield m_corners;

    emp_gui &m_gui;
    const emp_strategy_generator & m_generator;

    emp_situation_binary_dumper m_dumper;
    emp_types::bitfield m_bitfield;
    emp_types::bitfield m_empty_bitfield;

    uint64_t m_nb_situation_explored;
    uint64_t m_nb_solutions;
#ifdef WEBSERVER
    emp_web_server * m_web_server;
    bool m_pause_requested;
    bool m_paused;
#endif // WEBSERVER
  };

  //----------------------------------------------------------------------------
  emp_strategy::emp_strategy(const emp_strategy_generator & p_generator,
                             const emp_piece_db & p_piece_db,
                             emp_gui & p_gui,
                             const emp_FSM_info & p_FSM_info,
			     const std::string & p_file_name):
    m_piece_db(p_piece_db),
    m_size(p_generator.get_width() * p_generator.get_height()),
    // We allocate a supplementaty emp_position_strategy that will contains
    // no information. it will be used by corners and borders pieces as 
    // neighbour for orientations that are outside
    // Allocate memory for position strategies. It will allow to call kind specific constructor just after
    m_positions_strategy((emp_position_strategy *)operator new[](sizeof(emp_position_strategy) * (m_size + 1))),
    m_previous_available_pieces{& m_centers, & m_borders, & m_corners},
    m_centers(4 * p_piece_db.get_nb_pieces(emp_types::t_kind::CENTER),true),
    m_borders(4 * p_piece_db.get_nb_pieces(emp_types::t_kind::BORDER),true),
    m_corners(4 * p_piece_db.get_nb_pieces(emp_types::t_kind::CORNER),true),
    m_gui(p_gui),
    m_generator(p_generator),
    m_dumper(p_file_name,p_FSM_info,&m_generator,true),
    m_bitfield(m_size * (m_piece_db.get_piece_id_size() + 2)),
    m_nb_situation_explored(0),
    m_nb_solutions(0),
#ifdef WEBSERVER
      m_web_server(new emp_web_server(12345,*this,p_gui,p_FSM_info)),
      m_pause_requested(false),
      m_paused(false)
#endif //  WEBSERVER
    {

      uint32_t l_color_mask = (1 << p_piece_db.get_color_id_size()) - 1;
      std::cout << "Color id mask = 0x" << std::hex << l_color_mask << std::dec << std::endl ;

      new(&(m_positions_strategy[m_size]))emp_position_strategy(emp_types::t_kind::CENTER,m_empty_bitfield);
      emp_types::t_binary_piece l_fake_piece = 0;
      for(unsigned int l_index = 0 ; l_index < 4 ; ++l_index)
        {
	  l_fake_piece = l_fake_piece << p_piece_db.get_color_id_size();
	  l_fake_piece |= p_piece_db.get_border_color_id();
          emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[m_size]),l_index != 2 ? l_color_mask << (p_piece_db.get_color_id_size() * l_index) : 0x0);
          m_positions_strategy[m_size].set_neighbour_access((emp_types::t_orientation)((l_index + 2) % 4),l_access_info);
        }
      m_positions_strategy[m_size].set_piece_info(l_fake_piece);

      for(unsigned int l_index = 0 ; l_index < m_size ; ++l_index)
        {
          std::pair<unsigned int,unsigned int> l_current_position = p_generator.get_position(l_index);

          // Compute kind of current position
	  emp_types::t_kind l_kind = emp_types::t_kind::CENTER;
          bool l_x_border = l_current_position.first == 0 || l_current_position.first == p_generator.get_width() - 1;
          bool l_y_border = l_current_position.second == 0 || l_current_position.second == p_generator.get_height() - 1;
          if(l_x_border || l_y_border)
            {
              if(!l_x_border || !l_y_border)
                {
                  l_kind = emp_types::t_kind::BORDER;
                }
              else
                {
                  l_kind = emp_types::t_kind::CORNER;
                }
            }

          // Create position strategy in pre-allocated memory
          new(&(m_positions_strategy[l_index]))emp_position_strategy(l_kind,*(m_previous_available_pieces[(unsigned int)l_kind]));

          // Stored the new available pieces to be used by next position strategy of same kind
          m_previous_available_pieces[(unsigned int)l_kind] = &(m_positions_strategy[l_index].get_available_pieces());

          //Compute neighbour access info
          if(l_current_position.second > 0)
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[p_generator.get_position_index(l_current_position.first,l_current_position.second - 1)]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::SOUTH)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::NORTH,l_access_info);
            }
          else
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[m_size]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::SOUTH)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::NORTH,l_access_info);
            }
          if(l_current_position.first > 0)
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[p_generator.get_position_index(l_current_position.first - 1,l_current_position.second)]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::EAST)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::WEST,l_access_info);
            }
          else
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[m_size]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::EAST)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::WEST,l_access_info);
            }
          if(l_current_position.second < p_generator.get_height() - 1)
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[p_generator.get_position_index(l_current_position.first,l_current_position.second + 1)]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::NORTH)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::SOUTH,l_access_info);
            }
          else
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[m_size]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::NORTH)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::SOUTH,l_access_info);
            }
          if(l_current_position.first  < p_generator.get_width() - 1)
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[p_generator.get_position_index(l_current_position.first + 1,l_current_position.second)]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::WEST)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::EAST,l_access_info);
            }
          else
            {
              emp_position_strategy::t_neighbour_access l_access_info(&(m_positions_strategy[m_size]),l_color_mask << (p_piece_db.get_color_id_size() * (unsigned int)(emp_types::t_orientation::WEST)));
              m_positions_strategy[l_index].set_neighbour_access(emp_types::t_orientation::EAST,l_access_info);
            }

        }
      
    }

  //---------------------------------------------------------------------------
  void emp_strategy::compute_bin_id(emp_types::bitfield & p_bitfield)const
  {
    for(unsigned int l_index = 0 ; l_index < m_size ; ++l_index)
      {
        unsigned int l_data = m_positions_strategy[l_index].get_piece_info() >> ( 4 * m_piece_db.get_color_id_size());
        unsigned int l_offset = l_index * ( m_piece_db.get_piece_id_size() + 2);
        p_bitfield.set(l_data,m_piece_db.get_piece_id_size() + 2,l_offset);
      }
  }

  //--------------------------------------------------------------------------
  void emp_strategy::compute_available_transitions(const unsigned int & p_index)
  {
    assert(p_index <= m_size);
    emp_position_strategy & l_pos_strategy = m_positions_strategy[p_index];
    l_pos_strategy.compute_available_transitions(m_piece_db.get_pieces(l_pos_strategy.compute_constraint()));
  }

  //--------------------------------------------------------------------------
  emp_strategy::~emp_strategy(void)
    {
#ifdef WEBSERVER
      delete m_web_server;
#endif // WEBSERVER
      for(unsigned int l_index = 0 ; l_index <= m_size; ++l_index)
              {
                m_positions_strategy[l_index].~emp_position_strategy();
              }
      operator delete[] ((void*)m_positions_strategy);
      delete &m_generator;
    }

  //--------------------------------------------------------------------------
  void emp_strategy::display_on_gui(const unsigned int & p_index)
  {
    emp_FSM_situation l_situation;
    l_situation.set_context(*(new emp_FSM_context(m_size)));

    for(unsigned int l_index = 0 ; l_index <= p_index ; ++l_index)
      {
        const std::pair<unsigned int,unsigned int> & l_position = m_generator.get_position(l_index);
        emp_types::t_binary_piece l_binary = (m_positions_strategy[l_index].get_piece_info() >> (4 * m_piece_db.get_color_id_size()));
        emp_types::t_oriented_piece l_oriented_piece(1 + (l_binary >> 2),((emp_types::t_orientation)(l_binary & 0x3)));
        l_situation.set_piece(l_position.first,l_position.second,l_oriented_piece);
      }
    std::cout << l_situation.to_string() << std::endl ;
    m_gui.display(l_situation);
    m_gui.refresh();
  }


  //--------------------------------------------------------------------------
  void emp_strategy::run(void)
  {
#ifdef WEBSERVER
    m_web_server->start();
#endif // WEBSERVER

#ifdef GUI
    unsigned int l_max = 0;
#endif //GUI
    unsigned int l_index = 0;
    compute_available_transitions(l_index);
    bool l_continu = true;
    while(/*l_index < m_size && */ l_continu)
      {
        //#define GUI_SOLUTIONS
        //#define GUI
        unsigned int l_next_transition = m_positions_strategy[l_index].get_next_transition();
        if(l_next_transition)
          {
            // Need to decrement the transition id because 0 indicate no transition and bit[0] correspond to l_next_transition = 1
            --l_next_transition;
            m_positions_strategy[l_index].set_piece_info(m_piece_db.get_piece(m_positions_strategy[l_index].get_kind(),l_next_transition));
	    m_positions_strategy[l_index].select_piece(l_next_transition
#ifdef HANDLE_IDENTICAL_PIECES
                                                       ,m_piece_db.get_get_binary_identical_pieces(m_positions_strategy[l_index].get_kind(),l_next_transition)
#endif // HANDLE_IDENTICAL_PIECES
);
            ++m_nb_situation_explored;
#ifdef GUI
	    if(l_index > l_max)
	      {
		std::cout << "New max : " << l_index << std::endl ;
		display_on_gui(l_index);
		l_max = l_index;
	      }
#endif //GUI
            if(l_index == m_size - 1)
              {
                ++m_nb_solutions;
                compute_bin_id(m_bitfield);
                m_dumper.dump(m_bitfield,m_nb_situation_explored);
#ifdef GUI_SOLUTIONS
                display_on_gui(l_index);
#endif
              }

	    ++l_index;
	    compute_available_transitions(l_index);
          }
	else
	  {
            if(l_index < m_size)
              {
                m_positions_strategy[l_index].set_piece_info(0x0);
              }
            l_continu = l_index > 1;
	    --l_index;
#ifdef GUI
            if(l_continu && l_index > l_max)
              {
		std::cout << "New max : " << l_index << std::endl ;
                display_on_gui(l_index);
		l_max = l_index;
              }
#endif
	  }

#ifdef WEBSERVER
        if(m_pause_requested)
          {
#ifdef DEBUG_WEBSERVER
            std::cout << "Strategy entering in pause" << std::endl;
#endif
            m_paused = true;
            while(m_pause_requested)
              {
                usleep(1);
              }
#ifdef DEBUG_WEBSERVER
            std::cout << "Strategy leaving pause" << std::endl;
#endif
            m_paused = false;
          }
#endif // WEBSERVER

      }
    m_dumper.dump(m_nb_situation_explored);
    std::cout << "End of algorithm" << std::endl ;
    std::cout << "Total situations explored : "  << m_nb_situation_explored << std::endl ;
    std::cout << "Nb solutions : "  << m_nb_solutions << std::endl ;
  }


#ifdef WEBSERVER
  //--------------------------------------------------------------------------
  void emp_strategy::pause(void)
    {
      m_pause_requested = true;
    }

  //--------------------------------------------------------------------------
  void emp_strategy::restart(void)
    {
      m_pause_requested = false;
    }

  //--------------------------------------------------------------------------
  bool emp_strategy::is_paused(void)const
  {
    return m_paused;
  }
  //--------------------------------------------------------------------------
  void emp_strategy::send_info(uint64_t & p_nb_situations,
                               uint64_t & p_nb_solutions,
                               unsigned int & p_shift,
                               emp_types::t_binary_piece * p_pieces,
                               const emp_FSM_info & p_FSM_info)const
  {
    p_nb_situations = m_nb_situation_explored;
    p_nb_solutions = m_nb_solutions;
    p_shift = 4 * m_piece_db.get_color_id_size();
    for(unsigned int l_index = 0 ; l_index < m_size ; ++l_index)
      {
        const std::pair<unsigned int,unsigned int> & l_position = m_generator.get_position(l_index);
        p_pieces[p_FSM_info.get_width() * l_position.second + l_position.first] = m_positions_strategy[l_index].get_piece_info();
      }

  }
#endif // WEBSERVER

}
#endif // EMP_STRATEGY_H
//EOF
