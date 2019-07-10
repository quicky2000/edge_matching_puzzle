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

#ifndef _EMP_SYSTEM_EQUATION_H_
#define _EMP_SYSTEM_EQUATION_H_

#include "emp_advanced_feature_base.h"
#include "emp_piece_db.h"
#include "emp_FSM_info.h"
#include "emp_gui.h"
#include "emp_FSM_situation.h"
#include "emp_variable_generator.h"
#include "emp_types.h"
#include "emp_se_step_info.h"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>

//#define USE_KILL_SYNCHRO
#ifdef USE_KILL_SYNCHRO
#include <csetjmp>
#include "multi_thread_signal_handler.h"
#endif // USE_KILL_SYNCHRO

#ifdef USE_PIPE_SYNCHRO
#include <unistd.h>
#endif // USE_PIPE_SYNCHRO

namespace edge_matching_puzzle
{
    class emp_se_step_info;
    class emp_strategy_generator;

    class feature_system_equation: public emp_advanced_feature_base
#ifdef USE_KILL_SYNCHRO
                                 , public quicky_utils::multi_thread_signal_handler_listener_if
#endif // USE_KILL_SYNCHRO
    {
      public:
        feature_system_equation(const emp_piece_db & p_db
                               ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                               ,const emp_FSM_info & p_info
                               ,const std::string & p_initial_situation
                               ,const std::string & p_hint_string
                               ,emp_gui & p_gui
                               );

        // Method inherited from emp_advanced_feature_base
        void specific_run() override;

        void send_info(uint64_t & p_nb_situations
                      ,uint64_t & p_nb_solutions
                      ,unsigned int & p_shift
                      ,emp_types::t_binary_piece * p_pieces
                      ,const emp_FSM_info & p_FSM_info
                      ) const override;

        void compute_partial_bin_id(emp_types::bitfield & p_bitfield
                                   ,unsigned int p_max
                                   ) const override;
        // End of method inherited from feature if

        ~feature_system_equation() override = default;

      private:

        emp_FSM_situation extract_situation(const std::vector<emp_se_step_info> & p_stack
                                           ,unsigned int p_step
                                           );

#ifdef DEBUG
        void print_bitfield(const emp_types::bitfield & p_bitfield);
#endif // DEBUG

        /**
         * Indicate that piece corresponding to this variable should no more
         * be checked to detect if it cannot be used
         * @param p_variable variable indicating piece and its location
         * @return index in piece check list
         */
        unsigned int mark_checked(const simplex_variable & p_variable);

        /**
         * Method executed by worker thread
         * @param p_id thread id
         */
        void worker_run(unsigned int p_id);

        /**
         * Method to launch a worker thread
         * @param p_this reference on object containing info to be treated by
         * thread
         * @param p_thread_id thread id
         */
        static
        void launch_worker(feature_system_equation & p_this
                          ,unsigned int p_thread_id
                          );

#ifdef USE_KILL_SYNCHRO
        void
        handle_signal(int p_signal
                     ,unsigned int p_thread_index
                     ) override;

        /**
         * Send a signal to all worker thread finishing by the emitter one
         * @param p_thread_index thread index of emitter
         */
        void kill_all(unsigned int p_thread_index);
#endif // USE_KILL_SYNCHRO

        /**
         * Type representing tasks to execute by worker threads
         */
        typedef enum class thread_cmd {WAIT=0, START_AND, START_CHECK_POSITION, START_CHECK_PIECE, STOP} t_thread_cmd;

        /**
         * Method driving worker threads to execute task
         * Return when task has been executed
         * @param p_cmd command indicating task to execute
         */
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
        bool
#else // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
        void
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
        execute_task(t_thread_cmd p_cmd);

        /**
         * Inidicate to main thread that task is terminated
         * @param p_thread_id Thread Id
         */
        void finish_task(
#if defined(USE_PIPE_SYNCHRO) || defined(DEBUG_MULTITHREAD)
                         unsigned int p_thread_id
#endif // USE_PIPE_SYNCHRO || DEBUG_MULTITHREAD
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                        ,bool p_continu
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                        );

        /**
         * Contains initial situation
         */
        emp_FSM_situation m_initial_situation;

        /**
         * Contains hint situation
         */
        emp_FSM_situation m_hint_situation;

        /**
         * Generate variables of equation system representing the puzzle
         */
        emp_variable_generator m_variable_generator;


        std::vector<emp_types::bitfield> m_pieces_and_masks;

        /**
         * Masks used to check if a position can still be used
         */
        std::vector<emp_types::bitfield> m_positions_check_mask;

        /**
         * Masks used to check if a piece can still be used
         * Boolean indicate if make is usable or not:
         * true : mask can be used, piece still not used
         * false: mask cannot be used, piece is used
         */
        std::vector<std::pair<bool,emp_types::bitfield> > m_pieces_check_mask;

        std::vector<emp_se_step_info> m_stack;

        uint64_t m_counter = 0;

        unsigned int m_step;

        /**
         * Number of pieces in puzzle
         */
        unsigned int m_nb_pieces;


        /**
         * Index of variable to be applied at current step
         */
        unsigned int m_variable_index = 0;

        /**
         * Number of computing threads
         */
        static const unsigned int m_nb_worker_thread = 4;
        static_assert(!((m_nb_worker_thread - 1) & m_nb_worker_thread), "Nb worker thread must be a power of 2");

        /**
         * Threads
         */
        std::array<std::thread *, m_nb_worker_thread> m_threads;

#ifdef USE_PIPE_SYNCHRO
        std::array<int, 2 * m_nb_worker_thread> m_cmd_pipe_fd;

        std::array<int, 2 * m_nb_worker_thread> m_return_pipe_fd;
#else // USE_PIPE_SYNCHRO
        /**
         * Variable used to pass commands to threads
         */
        t_thread_cmd m_thread_cmd[m_nb_worker_thread];

        /**
         * Variable counting number of thread whose task is terminated
         */
        std::atomic<unsigned int> m_finished_thread_counter{0};

        /**
         * Mutex used to protect condition variable used to synchronize threads
         * start operation
         */
        std::mutex m_mutex_start;

        /**
         * Condition variable used to synchronize threads start operation
         */
        std::condition_variable m_condition_variable_start;

        /**
         * Mutex used to protect condition variable used to indicate threads
         * end task
         */
        std::mutex m_mutex_end;

        /**
         * Condition variable used to indicate threads are started
         */
        std::condition_variable m_condition_variable_end;

#endif // USE_PIPE_SYNCHRO

#if !(defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO))
        /**
         *
         */
        std::atomic<bool> m_continu_check;
#endif // !(USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO)
#ifdef USE_KILL_SYNCHRO
        /**
         * Buffer used by stejmp longjmp for thread synchronisation
         */
        std::array<std::jmp_buf, m_nb_worker_thread>  m_jump_buffer;

        /**
         * Thread Ids
         */
        std::array<std::thread::id, m_nb_worker_thread> m_thread_ids;

#endif // USE_KILL_SYNCHRO

        /**
         * Operator displaying name of thread command
         * @param p_stream stream on which thread command is displayed
         * @param p_cmd thread command whose name will be displayed
         * @return stream on which thread command is displayed
         */
        friend std::ostream & operator<<(std::ostream & p_stream, feature_system_equation::t_thread_cmd p_cmd);
    };
}
#endif // _EMP_SYSTEM_EQUATION_H_
// EOF
