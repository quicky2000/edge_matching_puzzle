/* -*- C++ -*- */
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
#ifndef _BORDER_COLOR_CONSTRAINT_H_
#define _BORDER_COLOR_CONSTRAINT_H_

#include <cinttypes>
#include <cstring>
#include <iomanip>

namespace edge_matching_puzzle
{
  /**
     Class representing corner and border pieces matching a color constraint
  */
  class border_color_constraint
  {
    friend inline std::ostream & operator<<(std::ostream & p_stream, const border_color_constraint & p_constraint);
  public:
    inline border_color_constraint(bool p_init = false);
    inline border_color_constraint(const border_color_constraint & p_constraint);
    inline void operator&(const border_color_constraint & p_constraint);
    inline void operator&(const uint64_t & p_constraint);
    inline void operator=(const border_color_constraint & p_constraint);
    inline void toggle_bit(unsigned int p_index, bool p_value);

    inline void fill(bool p_init);
    inline void set_bit(uint32_t p_index);
    inline void unset_bit(uint32_t p_index);

    inline bool get_bit(uint32_t p_index) const;

    inline int ffs(void) const;

  private:
    uint64_t m_constraint;
  };

  //------------------------------------------------------------------------------
  border_color_constraint::border_color_constraint(bool p_init):
    m_constraint(p_init ? UINT64_MAX : 0x0)
  {
  }

  //------------------------------------------------------------------------------
  border_color_constraint::border_color_constraint(const border_color_constraint & p_constraint):
    m_constraint(p_constraint.m_constraint)
  {
  }

  //------------------------------------------------------------------------------
  void border_color_constraint::operator=(const border_color_constraint & p_constraint)
  {
    m_constraint = p_constraint.m_constraint;  
  }

  //------------------------------------------------------------------------------
  void border_color_constraint::operator&(const border_color_constraint & p_constraint)
  {
    m_constraint &= p_constraint.m_constraint;  
  }

  //------------------------------------------------------------------------------
  void border_color_constraint::operator&(const uint64_t & p_constraint)
  {
    m_constraint &= p_constraint;  
  }

  //------------------------------------------------------------------------------
  void border_color_constraint::toggle_bit(unsigned int p_index, bool p_value)
  {
    m_constraint ^= ((uint64_t)p_value) << p_index;
  }

  //------------------------------------------------------------------------------
  void border_color_constraint::fill(bool p_init)
  {
    m_constraint = p_init ?  UINT64_MAX : 0x0;
  }

  //------------------------------------------------------------------------------
  void border_color_constraint::set_bit(uint32_t p_index)
  {
    m_constraint |= ((uint64_t)0x1) << p_index;
  }

  //------------------------------------------------------------------------------
  void border_color_constraint::unset_bit(uint32_t p_index)
  {
    m_constraint &= ~(((uint64_t)0x1) << p_index);
  }

  //------------------------------------------------------------------------------
  bool border_color_constraint::get_bit(uint32_t p_index) const
  {
    return m_constraint & (((uint64_t)0x1) << p_index);
  }

  //------------------------------------------------------------------------------
  int border_color_constraint::ffs(void) const
  {
    return ::ffsll(m_constraint);
  }

  //------------------------------------------------------------------------------
  std::ostream & operator<<(std::ostream & p_stream, const border_color_constraint & p_constraint)
  {
    p_stream << "0x" << std::hex << p_constraint.m_constraint << std::dec ;
    return p_stream;
  }
}
#endif // _BORDER_COLOR_CONSTRAINT_H_
// EOF
