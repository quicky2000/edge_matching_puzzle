/* -*- C++ -*- */
/*    This file is part of edge_matching_puzzle
      Copyright (C) 2017  Julien Thevenon ( julien_thevenon at yahoo.fr )

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FINTESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>
*/
#ifndef _BORDER_CONSTRAINT_GENERATOR_H_
#define _BORDER_CONSTRAINT_GENERATOR_H_

#include "octet_array.h"
#include <map>
#include <random>
#include <algorithm>

namespace edge_matching_puzzle
{
  class border_constraint_generator
  {
  public:
    inline border_constraint_generator(const std::map<unsigned int, unsigned int> & p_B2C_color_count);
    inline void generate(octet_array & p_constraint);
  private:
    std::array<unsigned int,56> m_array;
    std::random_device m_random_device;
    std::seed_seq m_seed2;
    std::mt19937 m_random_engine;
  };

  //------------------------------------------------------------------------------
  border_constraint_generator::border_constraint_generator(const std::map<unsigned int, unsigned int> & p_B2C_color_count):
#ifndef DETERMINISTIC_SEED
    m_seed2{m_random_device(), m_random_device(), m_random_device(), m_random_device(), m_random_device(), m_random_device(), m_random_device(), m_random_device()},
#else
    m_seed2{0, 0, 0, 0, 0, 0, 0, 0},
#endif // DETERMINISTIC_SEED
    m_random_engine(m_seed2)
    {
      unsigned int l_total = 0;
      unsigned int l_index = 0;
      for(auto l_iter:p_B2C_color_count)
	{
	  l_total += l_iter.second;
	  for(unsigned int l_count = 0; l_count < l_iter.second; ++l_count, ++l_index)
	    {
	      m_array[l_index] = l_iter.first;
	    }
	}
      assert(l_total == m_array.size());
    }

  //------------------------------------------------------------------------------
  void border_constraint_generator::generate(octet_array & p_constraint)
  {
    unsigned int l_output_index = 1;
    p_constraint.set_octet(0,0);
    // Be carreful l_output_index is related to an array of size 60 whereas m_array is size 56
    for(unsigned int l_index = 0; l_index < 55; ++l_index, ++l_output_index)
      {
	if(15 == l_output_index || 30 == l_output_index || 45 == l_output_index)
	  {
	    p_constraint.set_octet(l_output_index,0);
	    ++l_output_index;
	  }
	std::uniform_int_distribution<int> l_uniform_dist(0,55 - l_index);
	int l_rand = l_uniform_dist(m_random_engine);
	p_constraint.set_octet(l_output_index, m_array[l_rand]);
	std::swap(m_array[l_rand], m_array[55 - l_index]);
      }
    // For latest we no more have choice
    p_constraint.set_octet(59,m_array[0]);
  }
}
#endif // _BORDER_CONSTRAINT_GENERATOR_H_
// EOF
