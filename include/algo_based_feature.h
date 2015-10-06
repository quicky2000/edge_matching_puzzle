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
#ifndef ALGO_BASED_FEATURE
#define ALGO_BASED_FEATURE

#include "signal_handler_listener_if.h"
#include "FSM_UI.h"
#include "feature_if.h"
#include "emp_FSM.h"
#include <signal.h>
#include "signal_handler.h"
namespace edge_matching_puzzle
{
  
  class emp_piece_db;
  class emp_FSM_info;

  template<typename ALGO>
    class algo_based_feature:public feature_if,
    public quicky_utils::signal_handler_listener_if,
    public FSM_base::FSM_UI<emp_FSM_situation>
    {
    public:
      inline algo_based_feature(const emp_piece_db & p_db,
                                const emp_FSM_info & p_info,
                                emp_gui & p_gui);

      // Virtual methods inherited from signal_handler_listener_if
      inline void handle(int p_signal);
      // End of virtual methods inherited from signal_handler_listener_if
      // Virtual methods inherited from feature_if
      inline void run(void);
      // End of virtual methods inherited from feature_if
      virtual void print_status(void)=0;
    protected:
      inline emp_gui & get_gui(void);
      inline const ALGO & get_algo(void)const;
      inline ALGO & get_algo(void);
    private:
      quicky_utils::signal_handler m_signal_handler;
      ALGO m_algo;
      emp_gui * m_gui;
      emp_FSM m_FSM;
    };

  //----------------------------------------------------------------------------
  template<typename ALGO>
    algo_based_feature<ALGO>::algo_based_feature(const emp_piece_db & p_piece_db,
                                                 const emp_FSM_info & p_info,
                                                 emp_gui & p_gui):
    m_signal_handler(*this),
    m_gui(&p_gui),
    m_FSM(p_info,p_piece_db)
      {
        m_algo.set_fsm(&m_FSM);
      }
    //----------------------------------------------------------------------------
    template<typename ALGO>
      void algo_based_feature<ALGO>::run(void)
      {
        m_algo.set_fsm_ui(this);
        m_algo.run();
      }
    //----------------------------------------------------------------------------
    template<typename ALGO>
      emp_gui & algo_based_feature<ALGO>::get_gui(void)
      {
        assert(m_gui);
        return *m_gui;
      }
    //----------------------------------------------------------------------------
    template<typename ALGO>
    const ALGO & algo_based_feature<ALGO>::get_algo(void)const
      {
        return m_algo;
      }

    //----------------------------------------------------------------------------
    template<typename ALGO>
    ALGO & algo_based_feature<ALGO>::get_algo(void)
      {
        return m_algo;
      }

    //----------------------------------------------------------------------------
    template<typename ALGO>
      void algo_based_feature<ALGO>::handle(int p_signal)
      {
        switch(p_signal)
          {
          case SIGTERM:
          case SIGINT:
            std::cout << "=> Received SIGTERM or SIGINT : request algorithm stop" << std::endl ;
            m_algo.print_status();
            m_algo.stop();
            break;
          case SIGUSR1:
            std::cout << "=> Received SIGUSR1 : request algorithm status" << std::endl ;
            m_algo.print_status();
            this->print_status();
            break;
          default:
            std::cout << "=> Received unhandled signal " << p_signal << std::endl ;
            break;
       
          }
      }

}
#endif // ALGO_BASED_FEATURE
//EOF
