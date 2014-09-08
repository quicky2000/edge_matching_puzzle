/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
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
#ifndef SIGNAL_HANDLER_LISTENER_IF_H
#define SIGNAL_HANDLER_LISTENER_IF_H

namespace edge_matching_puzzle
{
  class signal_handler_listener_if
  {
  public:
    virtual void handle(int p_signal)=0;
    inline virtual ~signal_handler_listener_if(void){}
  };
}
#endif // SIGNAL_HANDLER_LISTENER_IF_H
//EOF
