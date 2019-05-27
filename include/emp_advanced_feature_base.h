/*    This file is part of edge_matching_puzzle
      Copyright (C) 2019  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef _EMP_ADVANCED_FEATURE_BASE_H_
#define _EMP_ADVANCED_FEATURE_BASE_H_

#include "feature_if.h"
#include "emp_situation_binary_dumper.h"
#include "emp_strategy_generator.h"
#include "emp_FSM_info.h"
#include "emp_piece_db.h"
#include "emp_web_server.h"
#include "emp_gui.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <cinttypes>

//#define WEBSERVER
//#define SAVE_THREAD

namespace edge_matching_puzzle
{
    /**
     * Base class for advanced feature
     * Provide some services like integrated web server, save thread etc
     */
    class emp_advanced_feature_base: public feature_if
    {
      public:
        inline
        emp_advanced_feature_base(std::unique_ptr<emp_strategy_generator> & p_generator
                                 ,const emp_FSM_info & p_info
                                 ,const emp_piece_db & p_piece_db
                                 ,emp_gui & p_gui
                                 );

        // Methods used by webserver when activated
        inline
        void pause();

        inline
        void restart();

        inline
        bool is_paused() const;

        // End of methods used by webserver when activated

        // Methods of feature_if
        inline
        void run() override;
        // End of methods of feature_if

        inline
        ~emp_advanced_feature_base() override;

        // Methods to be implemented by advanced features
        virtual
        void compute_partial_bin_id(emp_types::bitfield & p_bitfield
                                   ,unsigned int p_max
                                   ) const = 0;

        virtual
        void specific_run() = 0;

        virtual
        void send_info(uint64_t & p_nb_situations
                      ,uint64_t & p_nb_solutions
                      ,unsigned int & p_shift
                      ,emp_types::t_binary_piece * p_pieces
                      ,const emp_FSM_info & p_FSM_info
                      ) const = 0;

        // End of methods to be implemented by advanced features

      protected:
        inline
        void start();

        inline
        const emp_piece_db & get_piece_db() const;

        inline
        const std::unique_ptr<emp_strategy_generator> & get_generator() const;

        inline
        const emp_FSM_info & get_info() const;

        inline
        emp_gui & get_gui() const;

        inline
        bool is_pause_requested() const;

#ifdef SAVE_THREAD
        /**
         * Perform periodic action when entering pause
         * @param p_index represent the level of resolution
         * @param p_nb_situation_explored number of sitation explored at this time
         */
        inline
        void perform_save_action(unsigned int p_index
                                ,uint64_t p_nb_situation_explored
                                );
#endif // SAVE_THREAD

      private:

#ifdef SAVE_THREAD
        inline static
        void periodic_save(const std::atomic<bool> & p_stop
                          ,emp_advanced_feature_base & p_strategy
                          );
#endif // SAVE_THREAD

        /**
         * Atomic variable use by save thread to indicate to feature to pause
         */
        std::atomic<bool> m_pause_requested;

        /**
         * Atomic variable used by feature to indicate it is in pause
         */
        std::atomic<bool> m_paused;

#ifdef SAVE_THREAD
        std::atomic<bool> m_stop_save_thread;
        std::thread * m_save_thread;
        bool m_tic_toc = false;
        std::string m_save_name[2] = {"save_tic.bin", "save_toc.bin"};
#endif // SAVE_THREAD

        const emp_FSM_info & m_info;

        /**
         * Strategy generator that will determine order of positions
         */
        std::unique_ptr<emp_strategy_generator> m_generator;

        const emp_piece_db & m_piece_db;

        emp_gui & m_gui;

#ifdef WEBSERVER
        emp_web_server * m_web_server;
#endif // WEBSERVER

#ifdef PERFORMANCE_CHECK
        unsigned int m_dump_counter = 0;
#endif
    };

    //-------------------------------------------------------------------------
    emp_advanced_feature_base::emp_advanced_feature_base(std::unique_ptr<emp_strategy_generator> & p_generator
                                                        ,const emp_FSM_info & p_info
                                                        ,const emp_piece_db & p_piece_db
                                                        ,emp_gui & p_gui
                                                        )
    : m_pause_requested(false)
    , m_paused(false)
#ifdef SAVE_THREAD
    , m_stop_save_thread(false)
    , m_save_thread(nullptr)
#endif // SAVE_THREAD
    , m_info(p_info)
    , m_generator(std::move(p_generator))
    , m_piece_db(p_piece_db)
    , m_gui(p_gui)
#ifdef WEBSERVER
    , m_web_server(new emp_web_server(12345,*this,p_gui,m_info))
#endif //  WEBSERVER
    {
#ifdef SAVE_THREAD
        m_save_thread = new std::thread(periodic_save,std::ref(m_stop_save_thread),std::ref(*this));
#endif // SAVE_THREAD
    }

    //--------------------------------------------------------------------------
    emp_advanced_feature_base::~emp_advanced_feature_base()
    {
#ifdef WEBSERVER
        delete m_web_server;
#endif // WEBSERVER
#ifdef SAVE_THREAD
        m_stop_save_thread = true;
        m_save_thread->join();
        delete m_save_thread;
#endif // SAVE_THREAD
    }

    //-------------------------------------------------------------------------
    bool
    emp_advanced_feature_base::is_pause_requested() const
    {
        return m_pause_requested;
    }

    //--------------------------------------------------------------------------
    void emp_advanced_feature_base::pause()
    {
        m_pause_requested = true;
    }

    //--------------------------------------------------------------------------
    void emp_advanced_feature_base::restart()
    {
        m_pause_requested = false;
    }

    //--------------------------------------------------------------------------
    bool emp_advanced_feature_base::is_paused() const
    {
        return m_paused;
    }

    //-------------------------------------------------------------------------
    const emp_piece_db &
    emp_advanced_feature_base::get_piece_db() const
    {
        return m_piece_db;
    }

    //-------------------------------------------------------------------------
    const std::unique_ptr<emp_strategy_generator> &
    emp_advanced_feature_base::get_generator() const
    {
        return m_generator;
    }

    //-------------------------------------------------------------------------
    const emp_FSM_info & emp_advanced_feature_base::get_info() const
    {
        return m_info;
    }

    //-------------------------------------------------------------------------
    emp_gui & emp_advanced_feature_base::get_gui() const
    {
        return m_gui;
    }

    //-------------------------------------------------------------------------
    void emp_advanced_feature_base::run()
    {
#ifdef WEBSERVER
        m_web_server->start();
#endif // WEBSERVER
        this->specific_run();
    }

#ifdef SAVE_THREAD
    //--------------------------------------------------------------------------
    void emp_advanced_feature_base::periodic_save(const std::atomic<bool> & p_stop, emp_advanced_feature_base & p_strategy)
    {
        std::cout << "Create save thread" << std::endl ;
        while(!static_cast<bool>(p_stop))
        {
            std::this_thread::sleep_for(std::chrono::duration<int>(60));
            //	std::cout << "Ask for save" << std::endl ;
            p_strategy.pause();
            //	std::cout << "Wait for save done" << std::endl ;
            while(!p_strategy.is_paused())
            {
                usleep(1);
            }
            //	std::cout << "Save done" << std::endl ;
            p_strategy.restart();
        }
    }

    //--------------------------------------------------------------------------
    void
    emp_advanced_feature_base::perform_save_action(unsigned int p_index
                                                  ,uint64_t p_nb_situation_explored
                                                  )
    {
#if defined DEBUG_WEBSERVER || defined DEBUG_SAVE_THREAD
        std::cout << "Feature entering in pause" << std::endl;
#endif // defined DEBUG_WEBSERVER || defined DEBUG_SAVE_THREAD
        {
            emp_situation_binary_dumper l_dumper(m_save_name[m_tic_toc], m_info, m_generator,false);
            m_tic_toc = !m_tic_toc;
            emp_types::bitfield l_partial_bitfield(m_info.get_width() * m_info.get_height() * (m_piece_db.get_dumped_piece_id_size() + 2));
            compute_partial_bin_id(l_partial_bitfield, p_index);
            l_dumper.dump(l_partial_bitfield, p_nb_situation_explored);
            // Dump pseudo total number of situation explored to have non truncated file
            l_dumper.dump(p_nb_situation_explored);
        }
        m_paused = true;
        while(is_pause_requested())
        {
            usleep(1);
        }
        m_paused = false;
#if defined DEBUG_WEBSERVER || defined DEBUG_SAVE_THREAD
        std::cout << "Feature leaving pause" << std::endl;
#endif //defined DEBUG_WEBSERVER || defined DEBUG_SAVE_THREAD

#ifdef PERFORMANCE_CHECK
        ++m_dump_counter;
        std::cout << "Dump counter = " << m_dump_counter << std::endl;
        if(2 == m_dump_counter) exit(0);
#endif
    }

#endif // SAVE_THREAD

}
#endif // _EMP_ADVANCED_FEATURE_BASE_H_
