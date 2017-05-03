/*    This file is part of edge_matching_puzzle
      Copyright (C) 2016  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef _OCTET_ARRAY_H_
#define _OCTET_ARRAY_H_

#include <cinttypes>
#include <cassert>

namespace edge_matching_puzzle
{
  /**
     Class representing constraint exerced by center side of borders.
     Colors oriented to center are 22 so we need 5 bits to represent them
     0 means no colour, it is used for corner pieces.
     Only 4 colours are stored per 32 bits word to make determination of word
     easier : shift is sufficient
  */
  class octet_array
  {
  public:
    inline octet_array(void);
    inline void set_octet(unsigned int p_index,
			  uint32_t p_color_id
			  );
    inline uint32_t get_octet(unsigned int p_index) const;
  private:
    uint32_t m_octets[15];
};

  //------------------------------------------------------------------------------
  octet_array::octet_array(void):
    m_octets{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    {
    }

    //------------------------------------------------------------------------------
    void octet_array::set_octet(unsigned int p_index,
				uint32_t p_color_id
				)
    {
      assert(p_index < 60);
      m_octets[p_index >> 2] &= ~(((uint32_t)0xFF) << (8 * (p_index & 0x3)));
      m_octets[p_index >> 2] |= p_color_id << (8 * (p_index & 0x3));
    }

    //------------------------------------------------------------------------------
    uint32_t octet_array::get_octet(unsigned int p_index) const
    {
      assert(p_index < 60);
      return ((m_octets[p_index >> 2]) >> (8 * (p_index & 0x3))) & 0xFF;
    }
}
#endif // _OCTET_ARRAY_H_
// EOF
