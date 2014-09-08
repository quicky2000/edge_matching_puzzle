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
//TO DELETE#include "algorithm_random.h"
#include "signal_handler_listener_if.h"
#include <iostream>

namespace edge_matching_puzzle
{

  class signal_handler
  {
  public:
    //TO DELETE   inline signal_handler(FSM_framework::algorithm_random & p_algo);
    inline signal_handler(signal_handler_listener_if & p_listener);
    inline static void handler(int p_signal);
  private:
    //TO DELETE    static FSM_framework::algorithm_random * m_algo;
    static signal_handler_listener_if * m_listener;
  };

  //----------------------------------------------------------------------------
  //TO DELETE  signal_handler::signal_handler(FSM_framework::algorithm_random & p_algo)
  signal_handler::signal_handler(signal_handler_listener_if & p_listener)
    {
      //TO DELETE      m_algo = &p_algo;
      m_listener = &p_listener;
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
    m_listener->handle(p_signal);
    //TO DELETE    switch(p_signal)
    //TO DELETE      {
    //TO DELETE      case SIGTERM:
    //TO DELETE      case SIGINT:
    //TO DELETE        std::cout << "=> Received SIGTERM or SIGINT : request algorithm stop" << std::endl ;
    //TO DELETE        m_algo->print_status();
    //TO DELETE        m_algo->stop();
    //TO DELETE        break;
    //TO DELETE      case SIGUSR1:
    //TO DELETE        std::cout << "=> Received SIGUSR1 : request algorithm status" << std::endl ;
    //TO DELETE        m_algo->print_status();
    //TO DELETE        break;
    //TO DELETE      default:
    //TO DELETE        std::cout << "=> Received unhandled signal " << p_signal << std::endl ;
    //TO DELETE        break;
    //TO DELETE       
    //TO DELETE      }
  }
}
#endif // SIGNAL_HANDLER_H
//EOF
