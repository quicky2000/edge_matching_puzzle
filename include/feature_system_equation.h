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

namespace edge_matching_puzzle
{
    class emp_se_step_info;
    class emp_strategy_generator;

    class feature_system_equation: public emp_advanced_feature_base
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

        /**
         * Type representing tasks to execute by worker threads
         */
        typedef enum class thread_cmd {WAIT=0, START_AND, START_CHECK_POSITION, START_CHECK_PIECE, STOP} t_thread_cmd;

        /**
         * Method driving worker threads to execute task
         * Return when task has been executed
         * @param p_cmd command indicating task to execute
         */
        void execute_task(t_thread_cmd p_cmd);

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
        unsigned int m_nb_worker_thread = 4;

        /**
         * Variable used to pass commands to threads
         */
        std::atomic<unsigned int> m_thread_cmd{(unsigned int)t_thread_cmd::WAIT};

        /**
         * Variable counting number of thread whose task is started
         */
        std::atomic<unsigned int> m_started_thread_counter{0};

        /**
         * Variable counting number of thread whose task is terminated
         */
        std::atomic<unsigned int> m_finished_thread_counter{0};

        /**
         * Variable counting number of thread ready to start a new task
         */
        std::atomic<unsigned int> m_ready_thread_counter{0};

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
