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
#ifndef _BORDER_BACK_TRACKER_H_
#define _BORDER_BACK_TRACKER_H_

#include <map>

namespace edge_matching_puzzle
{
  class light_border_pieces_db;
  class border_color_constraint;
  class octet_array;

  class border_backtracker
  {
  public:
    void run(const light_border_pieces_db & p_border_pieces,
	     const border_color_constraint  (&p_border_constraints)[23],
	     const octet_array & p_initial_constraint,
	     octet_array & p_solution
	     );
  private:
  };
}
#endif // _BORDER_BACK_TRACKER_H_
// EOF
