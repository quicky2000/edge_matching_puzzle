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

#include <feature_system_equation.h>

#include "feature_system_equation.h"
#include "emp_se_step_info.h"
#include <thread>

//#define DEBUG_PIECE_CHECK
//#define DEBUG_POSITION_CHECK

namespace edge_matching_puzzle
{
    //-------------------------------------------------------------------------
    feature_system_equation::feature_system_equation(const emp_piece_db & p_db
                                                    ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                                    ,const emp_FSM_info & p_info
                                                    ,const std::string & p_initial_situation
                                                    ,const std::string & p_hint_string
                                                    ,emp_gui & p_gui
                                                    )
    : emp_advanced_feature_base(p_strategy_generator,  p_info, p_db, p_gui)
    , m_variable_generator(p_db
                          ,*get_generator()
                          ,p_info
                          ,p_hint_string
                          ,m_hint_situation
                          )
    ,m_nb_pieces(p_info.get_height() * p_info.get_width())
    {
        m_initial_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));
        m_hint_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));

        // Normally hint just done indication about piece position but not orientation
        assert(m_hint_situation.to_string() == m_initial_situation.to_string());

        if(!p_initial_situation.empty())
        {
            m_initial_situation.set(p_initial_situation);
        }

        auto l_nb_variables = (unsigned int)m_variable_generator.get_variables().size();

        std::cout << "Nb variables : " << l_nb_variables << std::endl;

        // Prepare maks that will be applied when selecting a piece
        m_pieces_and_masks.clear();
        assert(m_pieces_and_masks.empty());
        m_pieces_and_masks.resize(l_nb_variables,emp_types::bitfield(l_nb_variables, true));

        // Mask other variables with same position or same piece id
        for(auto l_iter: m_variable_generator.get_variables())
        {
            unsigned int l_position_index = p_info.get_position_index(l_iter->get_x(), l_iter->get_y());
            unsigned int l_id = l_iter->get_id();

            // Mask bits corresponding to other variables with same position
            for(auto l_iter_variable: m_variable_generator.get_position_variables(l_position_index))
            {
                m_pieces_and_masks[l_id].set(0, 1, l_iter_variable->get_id());
            }

            // Mask bits corresponding to other variables with same ppiece id
            for(auto l_iter_variable: m_variable_generator.get_piece_variables(l_iter->get_piece_id()))
            {
                m_pieces_and_masks[l_id].set(0, 1, l_iter_variable->get_id());
            }
        }

        // Use a local variable to give access to this member by waiting C++20
        // that allow "=, this" capture
        auto & l_pieces_and_masks = m_pieces_and_masks;
        auto & l_variable_generator = m_variable_generator;

        auto l_lamda = [=, & l_pieces_and_masks, & l_variable_generator](const simplex_variable & p_var1
                                                ,const simplex_variable & p_var2
                                                )
        {
            unsigned int l_id1 = p_var1.get_id();
            assert(l_id1 < l_variable_generator.get_variables().size());
            unsigned int l_id2 = p_var2.get_id();
            assert(l_id2 < l_variable_generator.get_variables().size());
            l_pieces_and_masks[l_id1].set(0, 1, l_id2);
            l_pieces_and_masks[l_id2].set(0, 1, l_id1);
        };

        // Mask variables due to incompatible borders
        m_variable_generator.treat_piece_relations(l_lamda);

        // Prepare masks that will be used to check if a position is still usable
        m_positions_check_mask.resize(get_info().get_width() * get_info().get_height(), emp_types::bitfield(l_nb_variables));
        for(auto l_iter: m_variable_generator.get_variables())
        {
            // Position index is index in strategy generator sequence !
            unsigned int l_position_index = get_generator()->get_position_index(l_iter->get_x(), l_iter->get_y());
            unsigned int l_id = l_iter->get_id();
            assert(l_position_index < m_positions_check_mask.size());
            m_positions_check_mask[l_position_index].set(1, 1, l_id);
        }

        // Prepare masks that will be used to check if a piece is still usable
        m_pieces_check_mask.resize(get_info().get_width() * get_info().get_height(), std::make_pair(true, emp_types::bitfield(l_nb_variables)));
        for(auto l_iter: m_variable_generator.get_variables())
        {
            // Position index is index in strategy generator sequence !
            unsigned int l_index = l_iter->get_piece_id() - 1;
            unsigned int l_id = l_iter->get_id();
            assert(l_index < m_pieces_check_mask.size());
            m_pieces_check_mask[l_index].second.set(1, 1, l_id);
        }
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::specific_run()
    {
        get_gui().display(m_initial_situation);

        // Prepare stack
        for(unsigned int l_index = 0; l_index < m_nb_pieces; ++l_index)
        {
            unsigned int l_x;
            unsigned int l_y;
            std::tie(l_x, l_y) = get_generator()->get_position(l_index);
            m_stack.emplace_back(emp_se_step_info(get_info().get_position_kind(l_x, l_y), (unsigned int)m_variable_generator.get_variables().size(), l_x, l_y));
        }
        m_stack.emplace_back(emp_se_step_info(emp_types::t_kind::UNDEFINED
                                             ,(unsigned int)m_variable_generator.get_variables().size()
                                             ,std::numeric_limits<unsigned int>::max()
                                             ,std::numeric_limits<unsigned int>::max()
                                             )
                            );

        // Set initial situation if provided
        unsigned int l_index = 0;
        bool l_continu = true;
        while(l_continu && l_index < m_nb_pieces)
        {
            unsigned int l_x;
            unsigned int l_y;
            std::tie(l_x, l_y) = get_generator()->get_position(l_index);
            l_continu = m_initial_situation.contains_piece(l_x,l_y);
            if(l_continu)
            {
                const emp_types::t_oriented_piece & l_oriented_piece = m_initial_situation.get_piece(l_x,l_y);
                unsigned int l_variable_id = 0;
                // Search variable correspoding to oriented piece in this position
                for(auto l_iter:m_variable_generator.get_position_variables(l_index))
                {
                    if(l_oriented_piece == l_iter->get_oriented_piece())
                    {
                        l_variable_id = l_iter->get_id();
                        break;
                    }
                }
                unsigned int l_variable_index = 0;
                bool l_found = false;
                while(!l_found && m_stack[l_index].get_next_variable(l_variable_index))
                {
                    l_found = l_variable_index == l_variable_id;
                    m_stack[l_index + 1].select_variable(l_variable_index, m_stack[l_index]);
                    m_stack[l_index + 1].apply_and(l_variable_index, m_stack[l_index], m_pieces_and_masks[l_variable_index], 0, 1);

                }
                if(!l_found)
                {
                    throw quicky_exception::quicky_logic_exception("No variable corresponding to initial situation provided at step " + std::to_string(l_index), __LINE__, __FILE__);
                }
                simplex_variable & l_variable = *m_variable_generator.get_variables()[l_variable_index];
                mark_checked(l_variable);
                ++l_index;
            }
        }

        // Create worker threads
#ifdef USE_KILL_SYNCHRO
        quicky_utils::multi_thread_signal_handler<m_nb_worker_thread>::create_unique_instance(*this, m_thread_ids, {SIGUSR1});
#endif // USE_KILL_SYNCHRO
#ifdef USE_PIPE_SYNCHRO
        for(unsigned int l_thread_index = 0; l_thread_index < m_nb_worker_thread; ++l_thread_index)
        {
            if (pipe(&m_cmd_pipe_fd[2 * l_thread_index]))
            {
                throw quicky_exception::quicky_runtime_exception(
                        "Error when creating command pipe for thread " + std::to_string(l_thread_index),
                        __LINE__,
                        __FILE__
                                                                );
            }
            if (pipe(&m_return_pipe_fd[2 * l_thread_index]))
            {
                throw quicky_exception::quicky_runtime_exception(
                        "Error when creating return pipe for thread " + std::to_string(l_thread_index),
                        __LINE__,
                        __FILE__
                                                                );
            }
        }
#endif // USE_PIPE_SYNCHRO
        for(unsigned int l_thread_index = 0; l_thread_index < m_nb_worker_thread; ++l_thread_index)
        {
#ifndef USE_PIPE_SYNCHRO
            m_thread_cmd[l_thread_index] = t_thread_cmd::WAIT;
#endif // USE_PIPE_SYNCHRO
            m_threads[l_thread_index] = new std::thread(feature_system_equation::launch_worker,std::ref(*this), l_thread_index);
#ifdef USE_KILL_SYNCHRO
            m_thread_ids[l_thread_index] = m_threads[l_thread_index]->get_id();
#endif // USE_KILL_SYNCHRO
        }

        m_step = l_index;
        unsigned int l_max_step = 0;
        while(m_step < m_nb_pieces)
        {
#if defined WEBSERVER || defined SAVE_THREAD
            if(is_pause_requested())
            {
                perform_save_action(m_step, m_counter);
            }
#endif
            ++m_counter;
            if(m_stack[m_step].get_next_variable(m_variable_index))
            {
                m_stack[m_step + 1].select_variable(m_variable_index, m_stack[m_step]);

                // Thread management
                execute_task(t_thread_cmd::START_AND);

                if(m_step > l_max_step)
                {
                    l_max_step = m_step;
                    emp_FSM_situation l_situation = extract_situation(m_stack, m_step);
                    std::cout << "Step: " << m_step << " => " << l_situation.to_string() << std::endl;
                    get_gui().display(l_situation);
                    get_gui().refresh();
                }
                simplex_variable & l_variable = *m_variable_generator.get_variables()[m_variable_index];
                if(m_stack[m_step].get_x() == l_variable.get_x() && m_stack[m_step].get_y() == l_variable.get_y())
                {
#if 0

                    ++m_step;
                    continue;
#endif // 0
#if !defined(USE_PIPE_SYNCHRO) || !defined(USE_KILL_SYNCHRO)
                    // Check if there are no lock positions
                    m_continu_check.store(true, std::memory_order_release);
#endif // !(USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO)

                    // Thread management
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                    bool l_result =
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                    execute_task(t_thread_cmd::START_CHECK_POSITION);

                    if(
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                      l_result
#else // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                      m_continu_check.load(std::memory_order_acquire)
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                      )
                    {
                        // Indicate that this piece should not be more checked
                        unsigned int l_check_piece_index = mark_checked(l_variable);

                        // Thread management
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                        l_result =
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                        execute_task(t_thread_cmd::START_CHECK_PIECE);

                        if(
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                           l_result
#else // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                           m_continu_check.load(std::memory_order_acquire)
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                          )
                        {
                            // Store piece check index associated with this step
                            // to be able to make pieces checkable again in case of rollback
                            m_stack[m_step].set_check_piece_index(l_check_piece_index);
                            ++m_step;
                        }
                        else
                        {
                            m_pieces_check_mask[l_check_piece_index].first = true;
#ifdef DEBUG_PIECE_CHECK
                            std::cout << "No more possible positions for piece " << l_tested_index + 1 << " after step " << m_step << std::endl;
#endif // DEBUG_PIECE_CHECK
                        }

                    }
#ifdef DEBUG_POSITION_CHECK
                    else
                    {
                        unsigned int l_fail_step_x;
                        unsigned int l_fail_step_y;
                        std::tie(l_fail_step_x, l_fail_step_y) = m_strategy_generator->get_position(m_step + 1);
                        std::cout << "No more possible pieces for position[" << l_fail_step_x << ", " << l_fail_step_y << "] @step " << m_step << std::endl;
                    }
#endif // DEBUG_POSITION_CHECK
                    continue;
                }
#ifdef DEBUG_POSITION_CHECK
                else
                {
                    std::cout << "Skipping position[" << m_stack[m_step].get_x() << ", " << m_stack[m_step].get_y() << "] @step " << m_step << "  and count " << m_counter << std::endl;
                }
#endif // DEBUG_POSITION_CHECK

            }
            assert(m_step);
            --m_step;
            m_variable_index = m_stack[m_step].get_variable_index();
            // Make piece that was used at this step checkable again
            unsigned int l_piece_check_index = m_stack[m_step].get_check_piece_index();
            assert(l_piece_check_index < m_pieces_check_mask.size());
            assert(!m_pieces_check_mask[l_piece_check_index].first);
            m_pieces_check_mask[l_piece_check_index].first = true;
        }
        std::cout << "Solution found after " << m_counter << " iterations" << std::endl;
        execute_task(t_thread_cmd::STOP);
        for(auto l_iter: m_threads)
        {
            l_iter->join();
            delete l_iter;
        }
    }

    //-------------------------------------------------------------------------
    emp_FSM_situation
    feature_system_equation::extract_situation(const std::vector<emp_se_step_info> & p_stack,
                                               unsigned int p_step
                                              )
    {
        emp_FSM_situation l_result;
        l_result.set_context(*(new emp_FSM_context(get_info().get_width() * get_info().get_height())));
        assert(p_step < (unsigned int)p_stack.size());
        for(unsigned int l_index = 0; l_index <= p_step; ++l_index)
        {
            unsigned int l_variable_index = p_stack[l_index].get_variable_index();
            simplex_variable & l_variable = *m_variable_generator.get_variables()[l_variable_index];
            l_result.set_piece(l_variable.get_x(), l_variable.get_y(), l_variable.get_oriented_piece());
        }
        return l_result;
    }

    //-------------------------------------------------------------------------
    unsigned int
    feature_system_equation::mark_checked(const simplex_variable & p_variable)
    {
        unsigned int l_check_piece_index = p_variable.get_piece_id() - 1;
        assert(l_check_piece_index < m_pieces_check_mask.size());
        assert(m_pieces_check_mask[l_check_piece_index].first);
        m_pieces_check_mask[l_check_piece_index].first = false;
        return l_check_piece_index;
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::send_info(uint64_t & p_nb_situations,
                                       uint64_t & p_nb_solutions,
                                       unsigned int & p_shift,
                                       emp_types::t_binary_piece *p_pieces,
                                       const emp_FSM_info & p_FSM_info
                                      ) const
    {
        p_nb_situations = m_counter;
        p_nb_solutions = 0;
        p_shift = 4 * get_piece_db().get_color_id_size();
        for(unsigned int l_index = 0 ; l_index < m_step; ++l_index)
        {
            const std::pair<unsigned int,unsigned int> & l_position = get_generator()->get_position(l_index);
            unsigned int l_variable_index = m_stack[l_index].get_variable_index();
            unsigned int l_piece_id = m_variable_generator.get_variables()[l_variable_index]->get_piece_id();
            unsigned int l_kind_index = get_piece_db().get_kind_index(l_piece_id);
            p_pieces[p_FSM_info.get_width() * l_position.second + l_position.first] = get_piece_db().get_piece(m_stack[l_index].get_kind(), (l_kind_index << 2) + (unsigned int)m_variable_generator.get_variables()[l_variable_index]->get_orientation());
        }
        for(unsigned int l_index = m_step ; l_index < get_info().get_width() * get_info().get_height(); ++l_index)
        {
            const std::pair<unsigned int,unsigned int> & l_position = get_generator()->get_position(l_index);
            p_pieces[p_FSM_info.get_width() * l_position.second + l_position.first] = 0;
        }
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::compute_partial_bin_id(emp_types::bitfield & p_bitfield
                                                   , unsigned int p_max
                                                   ) const
    {
        for(unsigned int l_index = 0 ; l_index <= p_max ; ++l_index)
        {
            unsigned int l_variable_index = m_stack[l_index].get_variable_index();
            unsigned int l_piece_id = 4 * m_variable_generator.get_variables()[l_variable_index]->get_piece_id() + (unsigned int) m_variable_generator.get_variables()[l_variable_index]->get_orientation();
            unsigned int l_offset = l_index * ( get_piece_db().get_dumped_piece_id_size() + 2);
            p_bitfield.set(l_piece_id,get_piece_db().get_dumped_piece_id_size() + 2,l_offset);
        }
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::worker_run(unsigned int p_id)
    {
#ifdef DEBUG_MULTITHREAD
        std::cout << "Thread " << p_id << " starting" << std::endl;
#endif // DEBUG_MULTITHREAD
#ifdef USE_KILL_SYNCHRO
        assert(m_jump_buffer.size() > p_id);
        if(setjmp(m_jump_buffer[p_id]))
        {
#ifndef USE_PIPE_SYNCHRO
            // In case thread was not started before receiving signal
            m_thread_cmd[p_id] = t_thread_cmd::WAIT;
#endif // USE_PIPE_SYNCHRO
            finish_task(
#if defined(DEBUG_MULTITHREAD) || defined(USE_PIPE_SYNCHRO)
                        p_id
#endif // DEBUG_MULTITHREAD || USE_PIPE_SYNCHRO
#ifdef USE_PIPE_SYNCHRO
                       ,false
#endif // USE_PIPE_SYNCHRO
                       );
        }
#endif // USE_KILL_SYNCHRO

#ifdef USE_PIPE_SYNCHRO
        t_thread_cmd l_thread_cmd;
#ifdef USE_KILL_SYNCHRO
        bool l_continu = true;
#endif // USE_KILL_SYNCHRO
#else // USE_PIPE_SYNCHRO
        t_thread_cmd & l_thread_cmd = m_thread_cmd[p_id];
#endif // USE_PIPE_SYNCHRO

        do
        {
#ifdef DEBUG_MULTITHREAD
            std::cout << "Thread " << p_id << " ready for new task" << std::endl;
#endif // DEBUG_MULTITHREAD

#ifdef USE_PIPE_SYNCHRO
            size_t l_size = read(m_cmd_pipe_fd[2 * p_id], &l_thread_cmd, sizeof(l_thread_cmd));
            assert(sizeof(l_thread_cmd) == l_size);
#ifdef USE_KILL_SYNCHRO
            l_continu = true;
#endif // USE_KILL_SYNCHRO
#else // USE_PIPE_SYNCHRO
            {
                std::unique_lock<std::mutex> l_lock(m_mutex_start);
                m_condition_variable_start.wait(l_lock
                                               , [&] {return t_thread_cmd::WAIT != l_thread_cmd;}
                                               );
                l_lock.unlock();
            }
#endif // USE_PIPE_SYNCHRO
#ifdef DEBUG_MULTITHREAD
            std::cout << "Thread " << p_id << " receive command \"" << l_thread_cmd << "\"" << std::endl;
#endif // DEBUG_MULTITHREAD

#ifdef DEBUG_MULTITHREAD
            std::cout << "Thread " << p_id << " start task" << std::endl;
#endif // DEBUG_MULTITHREAD
            // Perform task
            switch(l_thread_cmd)
            {
                case t_thread_cmd::START_AND:
                    m_stack[m_step + 1].apply_and(m_variable_index
                                                 ,m_stack[m_step]
                                                 ,m_pieces_and_masks[m_variable_index]
                                                 ,p_id
                                                 ,m_nb_worker_thread
                                                 );
#ifndef USE_PIPE_SYNCHRO
                    l_thread_cmd = t_thread_cmd::WAIT;
#endif // USE_PIPE_SYNCHRO
                    break;
                case t_thread_cmd::START_CHECK_POSITION:
#ifndef USE_PIPE_SYNCHRO
                    l_thread_cmd = t_thread_cmd::WAIT;
#endif // USE_PIPE_SYNCHRO
                    for(unsigned int l_tested_index = m_step + 1 + p_id;
#ifndef USE_KILL_SYNCHRO
                        m_continu_check.load(std::memory_order_acquire) &&
#endif // USE_KILL_SYNCHRO
                        l_tested_index < m_nb_pieces;
                        l_tested_index += m_nb_worker_thread
                       )
                    {
                        if(!m_stack[m_step + 1].check_mask(m_positions_check_mask[l_tested_index], m_variable_index + 1))
                        {
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                            l_continu = false;
#else // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                            m_continu_check.store(false, std::memory_order_release);
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
#ifdef USE_KILL_SYNCHRO
                            kill_all(p_id);
#endif // USE_KILL_SYNCHRO
                            break;
                        }
                    }
                    break;
                case t_thread_cmd::START_CHECK_PIECE:
                {
#ifndef USE_PIPE_SYNCHRO
                    l_thread_cmd = t_thread_cmd::WAIT;
#endif // USE_PIPE_SYNCHRO
                    // Check pieces
                    unsigned int l_tested_index = 0;
                    for (;
#ifndef USE_KILL_SYNCHRO
                         m_continu_check.load(std::memory_order_acquire) &&
#endif // USE_KILL_SYNCHRO
                         l_tested_index < m_nb_pieces;
                         ++l_tested_index
                        )
                    {
                        if (m_pieces_check_mask[l_tested_index].first)
                        {
                            if (!m_stack[m_step + 1].check_mask(m_pieces_check_mask[l_tested_index].second, m_variable_index + 1))
                            {
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                                l_continu = false;
#else // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                                m_continu_check.store(false, std::memory_order_release);
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
#ifdef USE_KILL_SYNCHRO
                                kill_all(p_id);
#endif // USE_KILL_SYNCHRO
                                break;
                            }
                        }
                    }
                }
                break;
                case t_thread_cmd::STOP:
                    break;
                default:
                    throw quicky_exception::quicky_logic_exception("Unknown thread command " + std::to_string((unsigned int) l_thread_cmd), __LINE__, __FILE__);
            }
            finish_task(
#if defined(USE_PIPE_SYNCHRO) || defined(DEBUG_MULTITHREAD)
                        p_id
#endif // USE_PIPE_SYNCHRO || DEBUG_MULTITHREAD
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                       ,l_continu
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
                       );
        } while(t_thread_cmd::STOP != l_thread_cmd);
#ifdef DEBUG_MULTITHREAD
        std::cout << "Thread " << p_id << " terminate" << std::endl;
#endif // DEBUG_MULTITHREAD
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::finish_task(
#if defined(USE_PIPE_SYNCHRO) || defined(DEBUG_MULTITHREAD)
                                         unsigned int p_thread_id
#endif // USE_PIPE_SYNCHRO && DEBUG_MULTITHREAD
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
                                        ,bool p_continu
#endif // DEBUG_MULTITHREAD && USE_KILL_SYNCHR
                                        )
    {
#ifdef DEBUG_MULTITHREAD
        std::cout << "Thread " << p_thread_id << " task finished" << std::endl;
#endif // DEBUG_MULTITHREAD
        // Indicate task is terminated

#ifdef USE_PIPE_SYNCHRO
#ifdef USE_KILL_SYNCHRO
        write(m_return_pipe_fd[ 2 * p_thread_id + 1], &p_continu, sizeof(p_continu));
#else // USE_KILL_SYNCHRO
        bool l_continu = true;
        write(m_return_pipe_fd[ 2 * p_thread_id + 1], &l_continu, sizeof(l_continu));
#endif // USE_KILL_SYNCHRO
#else // USE_PIPE_SYNCHRO
        if(m_finished_thread_counter.fetch_add(1, std::memory_order_release) == m_nb_worker_thread - 1)
        {
#ifdef DEBUG_MULTITHREAD
            std::cout << "Thread " << p_thread_id << " notify end of task" << std::endl;
#endif // DEBUG_MULTITHREAD
            std::lock_guard<std::mutex> l_lock(m_mutex_end);
            m_condition_variable_end.notify_one();
        }
#endif // USE_PIPE_SYNCHRO
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::launch_worker(feature_system_equation & p_this,
                                           unsigned int p_thread_id
                                          )
    {
        p_this.worker_run(p_thread_id);
    }

    //-------------------------------------------------------------------------
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
    bool
#else // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
    void
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
    feature_system_equation::execute_task(feature_system_equation::t_thread_cmd p_cmd)
    {
#ifdef DEBUG_MULTITHREAD
        std::cout << "Launch " << p_cmd << std::endl;
#endif // DEBUG_MULTITHREAD

#ifdef USE_PIPE_SYNCHRO
        for(unsigned int l_thread_index = 0; l_thread_index < m_nb_worker_thread; ++l_thread_index)
        {
            write(m_cmd_pipe_fd[2 * l_thread_index + 1], &p_cmd, sizeof(p_cmd));
        }
#else // USE_PIPE_SYNCHRO
        {
            std::lock_guard<std::mutex> l_lock(m_mutex_start);
            for(unsigned int l_thread_index = 0; l_thread_index < m_nb_worker_thread; ++l_thread_index)
            {
                m_thread_cmd[l_thread_index] = p_cmd;
            }
        }
#ifdef DEBUG_MULTITHREAD
        std::cout << "Notify threads to start" << std::endl;
#endif // DEBUG_MULTITHREAD
        m_condition_variable_start.notify_all();
#endif // USE_PIPE_SYNCHRO
#ifdef DEBUG_MULTITHREAD
        std::cout << "Wait end of task execution" << std::endl;
#endif // DEBUG_MULTITHREAD
#ifdef USE_PIPE_SYNCHRO
#ifdef USE_KILL_SYNCHRO
        bool l_result{true};
#endif // USE_KILL_SYNCHRO
        bool l_continu;
        for(unsigned int l_thread_index = 0; l_thread_index < m_nb_worker_thread; ++l_thread_index)
        {
            read(m_return_pipe_fd[2 * l_thread_index], &l_continu, sizeof(l_continu));
#ifdef USE_KILL_SYNCHRO
            l_result &= l_continu;
#endif // USE_KILL_SYNCHRO
        }
#else // USE_PIPE_SYNCHRO
        {
            std::unique_lock<std::mutex> l_lock(m_mutex_end);
            std::atomic<unsigned int> & l_finished_thread_counter = m_finished_thread_counter;
            m_condition_variable_end.wait(l_lock
#ifdef USE_KILL_SYNCHRO
            // Some process can receive signal after update of finished thread counter
                                         , [&] {return l_finished_thread_counter >= m_nb_worker_thread;}
#else // USE_KILL_SYNCHRO
                                         , [&] {return l_finished_thread_counter == m_nb_worker_thread;}
#endif // USE_KILL_SYNCHRO
                                         );
            l_lock.unlock();
        }
        m_finished_thread_counter.store(0, std::memory_order_release);
#endif // USE_PIPE_SYNCHRO
#ifdef DEBUG_MULTITHREAD
        std::cout << "Task execution terminated" << std::endl;
#endif // DEBUG_MULTITHREAD
#if defined(USE_PIPE_SYNCHRO) && defined(USE_KILL_SYNCHRO)
        return l_result;
#endif // USE_PIPE_SYNCHRO && USE_KILL_SYNCHRO
    }

#ifdef USE_KILL_SYNCHRO
    //-------------------------------------------------------------------------
    void
    feature_system_equation::handle_signal(int p_signal
                                          ,unsigned int p_thread_index
                                          )
    {
#ifdef DEBUG_MULTITHREAD
        std::cout << "Signal received by thread " << p_thread_index << std::endl;
#endif // DEBUG_MULTITHREAD
        assert(p_signal == SIGUSR1);
        assert(p_thread_index < m_nb_worker_thread);
        longjmp(m_jump_buffer[p_thread_index], 1);
    }

    //-------------------------------------------------------------------------
    void
    feature_system_equation::kill_all(unsigned int p_thread_index)
    {
#ifdef DEBUG_MULTITHREAD
        std::cout << "Thread " << p_thread_index << " send signal" << std::endl;
#endif // DEBUG_MULTITHREAD
        for(unsigned int l_thread_index = p_thread_index + 1; l_thread_index < m_nb_worker_thread; ++l_thread_index)
        {
            pthread_kill(m_threads[l_thread_index]->native_handle(), SIGUSR1);
        }
        for(unsigned int l_thread_index = 0; l_thread_index <= p_thread_index; ++l_thread_index)
        {
            pthread_kill(m_threads[l_thread_index]->native_handle(), SIGUSR1);
        }
    }

#endif // USE_KILL_SYNCHRO

    //-------------------------------------------------------------------------
    std::ostream & operator<<(std::ostream & p_stream, feature_system_equation::t_thread_cmd p_cmd)
    {
        switch(p_cmd)
        {
            case feature_system_equation::t_thread_cmd::WAIT:
                p_stream << "WAIT";
                break;
            case feature_system_equation::t_thread_cmd::START_AND:
                p_stream << "START_AND";
                break;
            case feature_system_equation::t_thread_cmd::START_CHECK_POSITION:
                p_stream << "START_CHECK_POSITION";
                break;
            case feature_system_equation::t_thread_cmd::START_CHECK_PIECE:
                p_stream << "START_CHECK_PIECE";
                break;
            case feature_system_equation::t_thread_cmd::STOP:
                p_stream << "STOP";
                break;
            default:
                throw quicky_exception::quicky_logic_exception("Unknown thread command " + std::to_string((unsigned int)p_cmd), __LINE__, __FILE__);
        }
        return p_stream;
    }

#ifdef DEBUG
    //-------------------------------------------------------------------------
    void
    feature_system_equation::print_bitfield(const emp_types::bitfield & p_bitfield)
    {
        for(unsigned int l_index = 0; l_index < p_bitfield.bitsize(); ++l_index)
        {
            assert(l_index < m_variable_generator.get_variables().size());
            const simplex_variable & l_variable = *m_variable_generator.get_variables()[l_index];
            unsigned int l_value;
            p_bitfield.get(l_value, 1, l_index);
            std::cout << "Bit[" << l_index << "] = " << l_value << " => " << l_variable << std::endl;
        }
    }
#endif // 0

}
// EOF
