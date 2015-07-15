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
#ifndef FEATURE_DUMP_SUMMARY_H
#define FEATURE_DUMP_SUMMARY_H

#include "feature_if.h"
#include "emp_situation_binary_reader.h"

namespace edge_matching_puzzle
{
  class feature_dump_summary:public feature_if
  {
  public:
    inline feature_dump_summary(const std::string & p_file_name,
				const emp_FSM_info & p_info);
      // Virtual methods inherited from feature_if
    inline void run(void);
    // End of virtual methods inherited from feature_if    
  private:
    emp_situation_binary_reader m_reader;
  };
 
  //----------------------------------------------------------------------------
  feature_dump_summary::feature_dump_summary(const std::string & p_file_name,
					     const emp_FSM_info & p_info):
    m_reader(p_file_name,p_info)
  {
  }

  //----------------------------------------------------------------------------
  void feature_dump_summary::run(void)
  {
  }
 }
#endif // FEATURE_DUMP_SUMMARY_H
//EOF
