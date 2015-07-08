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

#ifndef EMP_STRATEGY_GENERATOR_H
#define EMP_STRATEGY_GENERATOR_H

#include "quicky_exception.h"
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "inttypes.h"

namespace edge_matching_puzzle
{
  /** The strategy generator indicate the order in which positions of EMP
      will be treated. It is uses to precompute strategy information
      This class is used only at the beginning of the execution so it
      don't particulary need to be optimised
  **/
  class emp_strategy_generator
  {
  public:
    inline emp_strategy_generator(const std::string & p_name,
				  const unsigned int & p_width,
				  const unsigned int & p_height);
    inline const std::string & get_name(void)const;

    /**
       Return the index in strategy sequence corresponding to position passed in parameter
    **/
    inline const unsigned int & get_position_index(const std::pair<unsigned int,unsigned int> & p_position)const;

    /**
       Return the index in strategy sequence corresponding to position passed in parameter
    **/
    inline const unsigned int & get_position_index(const unsigned int & p_x, const unsigned int & p_y)const;

    /**
       Return the position corresponding to an index in strategy sequence
    **/
    inline const std::pair<unsigned int,unsigned int> & get_position(const unsigned int & p_index)const;
    inline const unsigned int & get_width(void)const;
    inline const unsigned int & get_height(void)const;

    virtual void generate(void)=0;
    inline virtual ~emp_strategy_generator(void){}
  protected:
    inline void add_coordinate(const unsigned int & p_x,
                               const unsigned int & p_y);
  private:
    const std::string m_name;
    const unsigned int m_width;
    const unsigned int m_height;
    std::vector<std::pair<unsigned int,unsigned int> > m_index2positions;
    std::map<std::pair<unsigned int,unsigned int>,unsigned int> m_position2indexes;
  };

  //----------------------------------------------------------------------------
  emp_strategy_generator::emp_strategy_generator(const std::string & p_name,
						 const unsigned int & p_width,
						 const unsigned int & p_height):
    m_name(p_name),
    m_width(p_width),
    m_height(p_height)
    {

    }

    //----------------------------------------------------------------------------
    const unsigned int & emp_strategy_generator::get_width()const
      {
        return m_width;
      }

    //----------------------------------------------------------------------------
    const unsigned int & emp_strategy_generator::get_height()const
      {
        return m_height;
      }

    //----------------------------------------------------------------------------
    const std::string & emp_strategy_generator::get_name(void)const
      {
	return m_name;
      }

    //----------------------------------------------------------------------------
    void emp_strategy_generator::add_coordinate(const unsigned int & p_x,
						const unsigned int & p_y)
    {
      assert(p_x < m_width);
      assert(p_y < m_height);
      std::pair<unsigned int,unsigned int> l_pair(p_x,p_y);
      if(m_position2indexes.end() != m_position2indexes.find(l_pair))
	{
	  std::stringstream l_x_str;
	  l_x_str << p_x;
	  std::stringstream l_y_str;
	  l_y_str << p_y;
	  throw quicky_exception::quicky_logic_exception("Position(" + l_x_str.str() + "," + l_x_str.str() + ") already added for strategy \"" + m_name + "\"",__LINE__,__FILE__);
	}
      m_position2indexes.insert(std::map<std::pair<unsigned int,unsigned int>,unsigned int>::value_type(l_pair,m_index2positions.size()));
      m_index2positions.push_back(l_pair);
    }

    //--------------------------------------------------------------------------
    const unsigned int & emp_strategy_generator::get_position_index(const std::pair<unsigned int,unsigned int> & p_position)const
    {
      std::map<std::pair<unsigned int,unsigned int>,unsigned int>::const_iterator l_iter = m_position2indexes.find(p_position);
      assert(m_position2indexes.end() != l_iter);
      return l_iter->second;
    }

    //--------------------------------------------------------------------------
    const unsigned int & emp_strategy_generator::get_position_index(const unsigned int & p_x, const unsigned int & p_y)const
      {
	return get_position_index(std::pair<unsigned int,unsigned int>(p_x,p_y));
      }

    //--------------------------------------------------------------------------
    const std::pair<unsigned int,unsigned int> & emp_strategy_generator::get_position(const unsigned int & p_index)const
      {
        assert(p_index < m_index2positions.size());
        return m_index2positions[p_index];
      }

}

#endif // EMP_STRATEGY_GENERATORH
//EOF
