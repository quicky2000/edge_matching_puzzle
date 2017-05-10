/*    This file is part of edge_matching_puzzle
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
#ifndef _BORDER_EXPLORATION_H_
#define _BORDER_EXPLORATION_H_

#include "enumerator.h"
#include "octet_array.h"
#include "light_border_pieces_db.h"
#include "border_color_constraint.h"
#include "sequential_border_backtracker.h"
#include "border_backtracker.h"
#include "emp_types.h"
#include <map>
#include <atomic>
#include <thread>
#include <chrono>
#include <unistd.h>

//#define DISPLAY_SITUATION_STRING
#define DISPLAY_ALL_SOLUTIONS

namespace edge_matching_puzzle
{
  class light_border_pieces_db;

  class border_exploration
  {
  public:
    inline border_exploration(const std::map<unsigned int, unsigned int> & p_B2C_color_count,
			      const std::map<unsigned int, unsigned int> & p_reorganised_colors,
			      const border_color_constraint  (&p_border_constraints)[23],
			      const light_border_pieces_db & p_border_pieces,
			      const std::string & p_situation_string
			      );
    inline ~border_exploration(void);
    inline void run(const unsigned int (&p_border_edges)[60]);
  private:
    inline static void periodic_display(const std::atomic<bool> & p_stop,
					std::atomic<bool> & p_display
					);

    inline void extract_initial_constraint(const std::string & p_situation_string,
					   octet_array & p_initial_constraint,
					   const light_border_pieces_db & p_border_pieces
					   )const;

    inline void constraint_to_string(std::string & p_result,
				     const octet_array & p_situation,
				     const unsigned int (&p_border_edges)[60]
				     )const;

    inline void situation_string_to_generator_word(const std::string & p_situation_string) const;

    border_color_constraint  m_border_constraints[23];
    light_border_pieces_db m_border_pieces;
    combinatorics::enumerator * m_enumerator;
    unsigned int * m_reference_word;
    unsigned int m_translation_rule[56];
    std::vector<combinatorics::symbol> m_symbols;
  };

  //-----------------------------------------------------------------------------
  border_exploration::border_exploration(const std::map<unsigned int, unsigned int> & p_B2C_color_count,
					 const std::map<unsigned int, unsigned int> & p_reorganised_colors,
					 const border_color_constraint  (&p_border_constraints)[23],
					 const light_border_pieces_db & p_border_pieces,
					 const std::string & p_situation_string
					 ):
    m_enumerator(nullptr),
    m_reference_word(nullptr)
    {
#ifdef INSTRUMENT_BORDER_EXPLORATION
      std::cout << "=================================" << std::endl ;
      std::cout << "p_B2C_color_count :" << std::endl;
      for(auto l_iter: p_B2C_color_count)
	{
	  std::cout << "<" << l_iter.first << "," << l_iter.second << ">" << std::endl ;
	}
      std::cout << "p_reorganised_colors :" << std::endl;
      for(auto l_iter: p_reorganised_colors)
	{
	  std::cout << "<" << l_iter.first << "," << l_iter.second << ">" << std::endl ;
	}
      std::cout << "p_border_constraints :" << std::endl;
      for(unsigned int l_index = 0;
	  l_index < 23;
	  ++l_index
	  )
	{
	  std::cout << "p_border_constraints[" << l_index << "] = " << p_border_constraints[l_index] << std::endl ;
	}
      std::cout << "p_border_pieces :" << std::endl;
      std::cout << p_border_pieces << std::endl;
      std::cout << "=================================" << std::endl ;
#endif // INSTRUMENT_BORDER_EXPLORATION

      for(unsigned int l_index = 0;
	  l_index < 56;
	  ++l_index)
	{
	  m_translation_rule[l_index] = (1 + l_index + l_index / ( 56 /4 ));
	}

      // Iterate on reorganised colors to count their number and create corresponding symbols
      for(auto l_count_iter:p_B2C_color_count)
	{
	  std::map<unsigned int, unsigned int>::const_iterator l_reorganised_iter = p_reorganised_colors.find(l_count_iter.first);
	  assert(p_B2C_color_count.end() != l_reorganised_iter);
	  m_symbols.push_back(combinatorics::symbol(l_reorganised_iter->second,l_count_iter.second));
	}

      // Create enumerator
      m_enumerator = new combinatorics::enumerator(m_symbols);

      // Create a temporary generator to obtain the first combination
      combinatorics::enumerator l_enumerator(m_symbols);
      l_enumerator.generate();

      // Rotate the first word to create the reference one
      assert(0 == (m_enumerator->get_word_size() % 4));
      unsigned int * l_tmp_word = new unsigned int[m_enumerator->get_word_size()];
      l_enumerator.get_word(l_tmp_word, m_enumerator->get_word_size());
      m_reference_word = new unsigned int[m_enumerator->get_word_size()];
      for(unsigned int l_index = 0;
	  l_index < m_enumerator->get_word_size();
	  ++l_index
	  )
	{
	  m_reference_word[l_index] = l_tmp_word[(l_index + (m_enumerator->get_word_size() / 4)) % m_enumerator->get_word_size()];
	  std::cout << (char)('A' - 1 + m_reference_word[l_index]) ;
	}
      std::cout << std::endl;
      // Rebuild border constraints using the reorganised colors
      m_border_constraints[0] = p_border_constraints[0];
      for(unsigned int l_index = 1;
	  l_index < 23;
	  ++l_index
	  )
	{
	  std::map<unsigned int, unsigned int>::const_iterator l_iter = p_reorganised_colors.find(l_index);
	  assert(p_reorganised_colors.end() != l_iter);
	  m_border_constraints[l_iter->second] = p_border_constraints[l_index];
	}

      // Rebuild border pieces using the reorganised colors
      for(unsigned int l_index = 0;
	  l_index < 60;
	  ++l_index
	  )
	{
	  uint32_t l_left_color;
	  uint32_t l_center_color;
	  uint32_t l_right_color;
	  p_border_pieces.get_colors(l_index, l_left_color, l_center_color, l_right_color);
	  std::map<unsigned int, unsigned int>::const_iterator l_iter = p_reorganised_colors.find(l_left_color);
	  assert(p_reorganised_colors.end() != l_iter);
	  l_left_color = l_iter->second;
	  // Check for reorganised colors only in case of border pieces ie center_color != 0
	  if(l_center_color)
	    {
	      l_iter = p_reorganised_colors.find(l_center_color);
	      assert(p_reorganised_colors.end() != l_iter);
	      l_center_color = l_iter->second;
	    }
	  l_iter = p_reorganised_colors.find(l_right_color);
	  assert(p_reorganised_colors.end() != l_iter);
	  l_right_color = l_iter->second;
	  m_border_pieces.set_colors(l_index, l_left_color, l_center_color, l_right_color);
	}
    }

    //-----------------------------------------------------------------------------
    border_exploration::~border_exploration(void)
      {
	delete m_enumerator;
	delete m_reference_word;
      }

    //-----------------------------------------------------------------------------
    void border_exploration::run(const unsigned int (&p_border_edges)[60])
    {
#ifdef INSTRUMENT_BORDER_EXPLORATION
      std::cout << "=================================" << std::endl ;
      std::cout << "p_border_edges :" << std::endl;
      for(unsigned int l_index = 0;
	  l_index < 60;
	  ++l_index
	  )
	{
	  std::cout << "p_border_edges[" << l_index << "] = " << p_border_edges[l_index] << std::endl;
	}
      std::cout << "=================================" << std::endl ;
#endif // INSTRUMENT_BORDER_EXPLORATION

      uint64_t l_nb_solution = 0;
      bool l_continu = true;
      octet_array l_initial_constraint;

#ifndef DISPLAY_ALL_SOLUTIONS
      std::atomic<bool> l_display_solution(false);
      std::atomic<bool> l_stop_thread(false);
      std::thread l_periodic_thread(periodic_display,std::ref(l_stop_thread),std::ref(l_display_solution));
#else // DISPLAY_ALL_SOLUTIONS
      bool l_display_solution = true;
#endif // DISPLAY_ALL_SOLUTIONS
      if(m_enumerator->get_word_size() != 56)
	{
	  throw quicky_exception::quicky_logic_exception("Algorithm hardcoded for Eternity2 !", __LINE__, __FILE__);
	}
      sequential_border_backtracker l_border_backtracker;
      while(l_continu && m_enumerator->generate())
	{
	  l_continu = m_enumerator->compare_word(m_reference_word) < 0 && 1000 >= m_enumerator->get_count();
	  if(l_continu)
	    {
	      std::cout << "-------------------------------------------------------------------" << std::endl;
	      std::cout << "Candidate : " ;
	      m_enumerator->display_word();
	      std::cout << std::endl ;
	      std::cout << "Min modified index in word : " << m_enumerator->get_min_index() << std::endl;

               for(unsigned int l_index = m_enumerator->get_min_index();
		  l_index < 56;
		  ++l_index
		  )
		{
		  l_initial_constraint.set_octet(m_translation_rule[l_index],
						 m_enumerator->get_word_item(l_index)
						 );
		}
	      octet_array l_solution;
	      l_border_backtracker.run(m_border_pieces,
				       m_border_constraints,
				       l_initial_constraint,
				       l_solution
				       );
	      if(!l_solution.get_octet(0))
		{
		  unsigned int l_max_index = l_solution.get_octet(59);

		  std::cout << "==> No solution found" << std::endl;
		  std::cout << "Max index in border = " << l_max_index << std::endl ;

		  // Max index should never be 0 as there are no constraints on first corner
		  assert(l_max_index);
		  l_max_index = l_max_index - 1 - l_max_index / 15;

		  std::cout << "Max index in word = " << l_max_index << std::endl ;

		  // We invalide l_max_index + 1 because index start at 0 so if
		  // max_index is I is valid it means that range [0:I] of size I + 1
		  // is not valid
		  m_enumerator->invalidate_root(1 + l_max_index);
		  // Reset Max
		  l_solution.set_octet(59,0);
		}
	      else
		{
		  std::cout << "==> Solution found" << std::endl ;
		  ++l_nb_solution;
		  if(l_display_solution)
		    {
		      std::cout << "[" << l_nb_solution << "] : ";
		      m_enumerator->display_word();
#ifdef DISPLAY_SITUATION_STRING
		      std::string l_situation_string;
		      constraint_to_string(l_situation_string,
					   l_solution,
					   p_border_edges
					   );
		      std::cout << l_situation_string << std::endl;
#endif // DISPLAY_SITUATION_STRING
#ifndef DISPLAY_ALL_SOLUTIONS
		      l_display_solution = false;
#endif // DISPLAY_ALL_SOLUTIONS
		    }
		}
	    }
	}
      m_enumerator->display_word();
#ifndef DISPLAY_ALL_SOLUTIONS
      // Stop periodic thread
      l_stop_thread = true;
      l_periodic_thread.join();
#endif // DISPLAY_ALL_SOLUTIONS
    }

    //------------------------------------------------------------------------------
    void border_exploration::periodic_display(const std::atomic<bool> & p_stop,
					      std::atomic<bool> & p_display
					      )
    {
      std::cout << "Launching thread" << std::endl;
      while(!static_cast<bool>(p_stop))
	{
	  std::this_thread::sleep_for(std::chrono::minutes(10));
	  p_display = true;
	  while(p_display)
	    {
	      usleep(1);
	    }
	}
    }

    //------------------------------------------------------------------------------
    void border_exploration::extract_initial_constraint(const std::string & p_situation_string,
							octet_array & p_initial_constraint,
							const light_border_pieces_db & p_border_pieces
							)const
    {
      assert(256 * 4 == p_situation_string.size());
      for(unsigned int l_situation_index = 0 ;
	  l_situation_index < 256 ;
	  ++l_situation_index
	  )
	{
	  std::string l_piece_id_str = p_situation_string.substr(l_situation_index * 4,3);
	  if("---" != l_piece_id_str)
	    {
	      unsigned int l_piece_id = std::stoi(l_piece_id_str);
	      unsigned int l_constraint_index= 0;
	      bool l_meaningful = true;
	      if(l_situation_index < 16)
		{
		  l_constraint_index = l_situation_index;
		}
	      else if(15 == l_situation_index % 16)
		{
		  l_constraint_index = 15 + (l_situation_index / 16);
		}
	      else if(15 == l_situation_index / 16)
		{
		  l_constraint_index = 255 - l_situation_index + 30;
		}
	      else if(0 == l_situation_index % 16)
		{
		  l_constraint_index = 45 - (l_situation_index / 16 ) + 15;
		}
	      else
		{
		  l_meaningful = false;
		}
	      if(l_meaningful)
		{
		  p_initial_constraint.set_octet(l_constraint_index, p_border_pieces.get_center(l_piece_id - 1));
		}
	    }
	}
    }

    //------------------------------------------------------------------------------
    void border_exploration::constraint_to_string(std::string & p_result,
						  const octet_array & p_situation,
						  const unsigned int (&p_border_edges)[60]
						  )const
    {
      p_result = "";
      char l_orientation2string[4] = {'N', 'E', 'S', 'W'};
      for(unsigned int l_y = 0;
	  l_y < 16;
	  ++l_y
	  )
	{
	  for(unsigned int l_x = 0;
	      l_x < 16;
	      ++l_x
	      )
	    {
	      std::stringstream l_stream;
	      if(0 == l_y && 0 == l_x)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(0) << l_orientation2string[(p_border_edges[p_situation.get_octet(0) - 1] + 1) % 4];
		  p_result += l_stream.str();
		}
	      else if(0 == l_y && 15 == l_x)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(15) << l_orientation2string[p_border_edges[p_situation.get_octet(15) - 1]];
		  p_result += l_stream.str();
		}
	      else if(15 == l_y && 15 == l_x)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(30) << l_orientation2string[(p_border_edges[p_situation.get_octet(30) - 1] + 3) % 4];
		  p_result += l_stream.str();
		}
	      else if(15 == l_y && 0 == l_x)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(45) << l_orientation2string[(p_border_edges[p_situation.get_octet(45) - 1] + 2) % 4];
		  p_result += l_stream.str();
		}
	      else if(0 == l_y)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(l_x) << l_orientation2string[p_border_edges[p_situation.get_octet(l_x) - 1]];
		  p_result += l_stream.str();
		}
	      else if(15 == l_x)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(15 + l_y) << l_orientation2string[(p_border_edges[p_situation.get_octet(l_x) - 1] + 3) % 4];
		  p_result += l_stream.str();
		}
	      else if(15 == l_y)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(30 - l_x + 15) << l_orientation2string[(p_border_edges[p_situation.get_octet(l_x) - 1] + 2) % 4];
		  p_result += l_stream.str();
		}
	      else if(0 == l_x)
		{
		  l_stream << std::setw(3) << p_situation.get_octet(45 - l_y + 15) << l_orientation2string[(p_border_edges[p_situation.get_octet(l_x) - 1] + 1) % 4];
		  p_result += l_stream.str();
		}
	      else
		{
		  p_result += "----";
		}
	    }
	}
    }

    //------------------------------------------------------------------------------
    void border_exploration::situation_string_to_generator_word(const std::string & p_situation_string) const
    {
      // Create word representing a known solution
      if("" != p_situation_string)
	{
	  octet_array l_example_constraint;
	  extract_initial_constraint(p_situation_string,
				     l_example_constraint,
				     m_border_pieces
				     );
	  unsigned int * l_solution_word = new unsigned int[m_enumerator->get_word_size()];
	  for(unsigned int l_index = 0;
	      l_index < 56;
	      ++l_index)
	    {
	      //	  std::cout << l_solution_example.get_octet((1 + l_index + l_index / ( 56 /4 ))) << std::endl;
	      l_solution_word[l_index] = l_example_constraint.get_octet((1 + l_index + l_index / ( 56 /4 )));
	    }
	  combinatorics::enumerator l_enumerator(m_symbols);
	  l_enumerator.set_word(l_solution_word,l_enumerator.get_word_size());
	  std::cout << "------------------------------------------------" << std::endl;
	  std::cout << "Known solution : " << std::endl;
	  std::cout << "------------------------------------------------" << std::endl;
	  l_enumerator.display_word();
	  delete[] l_solution_word;
	  octet_array l_solution_example;
	  border_backtracker l_border_backtracker;
	  l_border_backtracker.run(m_border_pieces,
				   m_border_constraints,
				   l_example_constraint,
				   l_solution_example
				   );
	  std::cout << "==> Corner = " << l_solution_example.get_octet(0) << std::endl ;
	  std::cout << "Max = " << l_solution_example.get_octet(59) << std::endl ;
	}
    }
}
#endif // _BORDER_EXPLORATION_H_
// EOF
