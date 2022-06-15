//
// Created by quickux on 04/02/2021.
//

#ifndef EDGE_MATCHING_PUZZLE_FEATURE_SYS_EQUA_CUDA_BASE_H
#define EDGE_MATCHING_PUZZLE_FEATURE_SYS_EQUA_CUDA_BASE_H

#include "emp_piece_db.h"
#include "emp_strategy_generator.h"
#include "system_equation_for_CUDA.h"
#include "transition_manager.h"
#include "emp_situation.h"
#include "common.h"
#include <map>

namespace edge_matching_puzzle
{
    template<unsigned int NB_PIECES> class situation_capability;

    class feature_sys_equa_CUDA_base
    {

      public:

        inline
        feature_sys_equa_CUDA_base(const emp_piece_db & p_piece_db
                                  , const emp_FSM_info & p_info
                                  , std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                  , const std::string & p_initial_situation
                                  );

      protected:

        template<unsigned int NB_PIECES>
        std::unique_ptr<const transition_manager<NB_PIECES>> prepare_run(situation_capability<2 * NB_PIECES> & p_situation_capability
                                                                        ,std::map<unsigned int, unsigned int> p_variable_translator
                                                                        );


        [[nodiscard]] inline
        const emp_piece_db & get_piece_db() const;

        [[nodiscard]] inline
        const emp_FSM_info & get_info() const;

        [[nodiscard]] inline
        const emp_FSM_situation & get_initial_situation() const;

        [[nodiscard]] inline
        const emp_strategy_generator & get_strategy_generator() const;

        template <unsigned int NB_PIECES>
        static
        void set_piece(emp_situation & p_situation
                      ,const situation_capability<2 * NB_PIECES> & p_situation_capability
                      ,unsigned int p_x
                      ,unsigned int p_y
                      ,unsigned int p_piece_index
                      ,emp_types::t_orientation p_orientation
                      ,situation_capability<2 * NB_PIECES> & p_result_capability
                      ,const transition_manager<NB_PIECES> & p_transition_manager
                      );


      private:

        const emp_piece_db & m_piece_db;

        const emp_FSM_info & m_info;

        /**
         * Contains initial situation
         * Should be declared before variable generator to be fully built
         */
        emp_FSM_situation m_initial_situation;

        /**
         * Generate variables of equation system representing the puzzle
         */
        emp_variable_generator m_variable_generator;

        const emp_strategy_generator & m_strategy_generator;

    };

    //-------------------------------------------------------------------------
    feature_sys_equa_CUDA_base::feature_sys_equa_CUDA_base(const emp_piece_db & p_piece_db
                                                          ,const emp_FSM_info & p_info
                                                          ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                                                          ,const std::string & p_initial_situation
                                                          )
    :m_piece_db(p_piece_db)
    ,m_info(p_info)
    ,m_variable_generator(p_piece_db, *p_strategy_generator, p_info, "", m_initial_situation)
    ,m_strategy_generator(*p_strategy_generator)
    {
        m_initial_situation.set_context(*(new emp_FSM_context(p_info.get_width() * p_info.get_height())));
        if(!p_initial_situation.empty())
        {
            m_initial_situation.set(p_initial_situation);
        }
    }

    //-------------------------------------------------------------------------
    const emp_piece_db &
    feature_sys_equa_CUDA_base::get_piece_db() const
    {
        return m_piece_db;
    }

    //-------------------------------------------------------------------------
    const emp_FSM_info &
    feature_sys_equa_CUDA_base::get_info() const
    {
        return m_info;
    }

    //-------------------------------------------------------------------------
    const emp_FSM_situation &
    feature_sys_equa_CUDA_base::get_initial_situation() const
    {
        return m_initial_situation;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    std::unique_ptr<const transition_manager <NB_PIECES>>
    feature_sys_equa_CUDA_base::prepare_run(situation_capability<2 * NB_PIECES> & p_situation_capability
                                           ,std::map<unsigned int, unsigned int> p_variable_translator
                                           )
    {
        system_equation_for_CUDA::prepare_initial<NB_PIECES, situation_capability<2 * NB_PIECES>>(m_variable_generator, m_strategy_generator, p_situation_capability);
        const transition_manager<NB_PIECES> * l_transition_manager = system_equation_for_CUDA::prepare_transitions<NB_PIECES
                                                                                                                  ,situation_capability<2 * NB_PIECES>
                                                                                                                  ,transition_manager<NB_PIECES>>(m_info
                                                                                                                                                 ,m_variable_generator
                                                                                                                                                 ,m_strategy_generator
                                                                                                                                                 ,p_variable_translator
                                                                                                                                                 );
        return std::unique_ptr<const transition_manager<NB_PIECES>>(l_transition_manager);
    }

    //-------------------------------------------------------------------------
    const emp_strategy_generator &
    feature_sys_equa_CUDA_base::get_strategy_generator() const
    {
        return m_strategy_generator;
    }

    //-------------------------------------------------------------------------
    template <unsigned int NB_PIECES>
    void
    feature_sys_equa_CUDA_base::set_piece(emp_situation & p_situation
                                         ,const situation_capability<2 * NB_PIECES> & p_situation_capability
                                         ,unsigned int p_x
                                         ,unsigned int p_y
                                         ,unsigned int p_piece_index
                                         ,emp_types::t_orientation p_orientation
                                         ,situation_capability<2 * NB_PIECES> & p_result_capability
                                         ,const transition_manager<NB_PIECES> & p_transition_manager
                                         )
    {
        unsigned int l_raw_variable_id = system_equation_for_CUDA::compute_raw_variable_id(p_x
                                                                                          ,p_y
                                                                                          ,p_piece_index
                                                                                          ,p_orientation
                                                                                          ,p_situation.get_info()
                                                                                          );
        p_result_capability.apply_and(p_situation_capability, p_transition_manager.get_transition(l_raw_variable_id));
        p_situation.set_piece(p_x, p_y, emp_types::t_oriented_piece(p_piece_index + 1, p_orientation));
    }

}


#endif //EDGE_MATCHING_PUZZLE_FEATURE_SYS_EQUA_CUDA_BASE_H
