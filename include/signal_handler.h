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
#ifndef SIGNAL_HANDLER_H
#define SIGNAL_HANDLER_H

#include <signal.h>
#include "algorithm_deep_raw.h"
#include <iostream>

namespace edge_matching_puzzle
{

  class signal_handler
  {
  public:
    inline signal_handler(FSM_framework::algorithm_deep_raw & p_algo);
    inline static void handler(int p_signal);
  private:
    static FSM_framework::algorithm_deep_raw * m_algo;
  };

  //----------------------------------------------------------------------------
  signal_handler::signal_handler(FSM_framework::algorithm_deep_raw & p_algo)
    {
      m_algo = &p_algo;
#ifndef _WIN32
      //Preparing signal handling to manage stop
      /* Déclaration d'une structure pour la mise en place des gestionnaires */
      struct sigaction l_signal_action;
  
      /* Remplissage de la structure pour la mise en place des gestionnaires */
      /* adresse du gestionnaire */
      l_signal_action.sa_handler=handler;
      // Mise a zero du champ sa_flags theoriquement ignoré
      l_signal_action.sa_flags=0;
      /* On ne bloque pas de signaux spécifiques */
      sigemptyset(&l_signal_action.sa_mask);
    
      /* Mise en place du gestionnaire bidon pour trois signaux */
      sigaction(SIGINT,&l_signal_action,0);
      sigaction(SIGTERM,&l_signal_action,0);
      sigaction(SIGUSR1,&l_signal_action,0);
#else
      signal(SIGINT,handler);
      signal(SIGTERM,handler);
      signal(SIGUSR1,handler);
#endif
    }

  //----------------------------------------------------------------------------
  void signal_handler::handler(int p_signal)
  {
    switch(p_signal)
      {
      case SIGTERM:
      case SIGINT:
        std::cout << "=> Received SIGTERM or SIGINT : request algorithm stop" << std::endl ;
        m_algo->print_status();
        m_algo->stop();
        break;
      case SIGUSR1:
        std::cout << "=> Received SIGUSR1 : request algorithm status" << std::endl ;
        m_algo->print_status();
        break;
      default:
        std::cout << "=> Received unhandled signal " << p_signal << std::endl ;
        break;
       
      }
  }
}
#endif // SIGNAL_HANDLER_H
//EOF
