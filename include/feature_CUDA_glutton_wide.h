/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2024  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_FEATURE_CUDA_GLUTTON_WIDE_H
#define EDGE_MATCHING_PUZZLE_FEATURE_CUDA_GLUTTON_WIDE_H

#include "emp_FSM_info.h"
#include "feature_if.h"
#include "quicky_exception.h"

namespace edge_matching_puzzle
{
    class emp_piece_db;

    /**
     * Class implementing glutton wide algorithm relying on CUDA GOU acceleration
     */
    class feature_CUDA_glutton_wide: public feature_if
    {
      public:

        feature_CUDA_glutton_wide(const emp_piece_db & p_piece_db
                                 ,const emp_FSM_info & p_info
                                 );

        // Methods inherited from feature_if
        void run() override;

        // End of methods inherited from feature_if
      private:

        const emp_piece_db & m_piece_db;

        const emp_FSM_info & m_info;
    };

    /**
     * Launch CUDA glutton wides
     */
    void
    launch_CUDA_glutton_wide(const emp_piece_db & p_piece_db
                            ,const emp_FSM_info & p_info
                            );


    //-------------------------------------------------------------------------
    feature_CUDA_glutton_wide::feature_CUDA_glutton_wide(const emp_piece_db & p_piece_db
                                                        ,const emp_FSM_info & p_info
                                                        )
    :m_piece_db(p_piece_db)
    ,m_info{p_info}
    {

    }

    //-------------------------------------------------------------------------
    void
    feature_CUDA_glutton_wide::run()
    {
        launch_CUDA_glutton_wide(m_piece_db, m_info);
    }

}
#endif //EDGE_MATCHING_PUZZLE_FEATURE_CUDA_GLUTTON_WIDE_H
// EOF

