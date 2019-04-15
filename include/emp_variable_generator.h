/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
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
#ifndef EMP_VARIABLE_GENERATOR_H
#define EMP_VARIABLE_GENERATOR_H

#include "emp_piece_db.h"
#include "emp_FSM_info.h"
#include "simplex_variable.h"
#include "emp_FSM_situation.h"
#include "emp_types.h"
#include <vector>
#include <string>
#include <map>

namespace edge_matching_puzzle
{
    class emp_variable_generator
    {
      public:
        inline
        emp_variable_generator(const emp_piece_db & p_db
                              ,const emp_FSM_info & p_info
                              ,std::string p_hints
                              ,emp_FSM_situation & p_situation
                              );

        /**
         * Accessor to tring representation of initial situation without hints
         * @return string representation of initial situation without hints
         */
        inline
        const std::string & get_initial_situation_str() const;

        /**
         * Accessor on variable list
         * @return reference on variable list
         */
        inline
        const std::vector<simplex_variable*> & get_variables() const;

        /**
         * Accessor to variables related to a position index
         * @param p_index position index
         * @return reference on variable list
         */
        inline
        const std::vector<simplex_variable*> & get_position_variables(unsigned int p_index) const;

        /**
         * Perform operations related to relationship between 2 adjacent postions
         * @tparam T type used to define algorithm containing operations to perform
         * @param p_variables_pos1 simplex variables related to position 1
         * @param p_variables_pos2 simplex variables related to position 2
         * @param p_horizontal true if positions are related horizontally
         * @param p_lambda algorithm containing operations to perform
         */
        template <typename T>
        void treat_piece_relation_equation(const std::vector<simplex_variable*> & p_variables_pos1
                                          ,const std::vector<simplex_variable*> & p_variables_pos2
                                          ,bool p_horizontal
                                          ,T & p_lambda
                                          );

        template <typename T>
        void treat_piece_relations(T & p_lambda);

        /**
          * Destructor
          */
         inline
         ~emp_variable_generator();

      private:
        /**
         * Treat hint string representation to extract hints ( pieces whose
         * location is know but not orientation) and remove them from string
         * representation
         * @param p_situation Hint string representation of situation
         * @return string representation without hints
         */
        inline
        void extract_hints();

        /**
         * Record hint information: piece + its location
         * @param p_piece_id Piece Id
         * @param p_x X location
         * @param p_y Y location
         */
        inline
        void record_hint(unsigned int p_piece_id
                        ,unsigned int p_x
                        ,unsigned int p_y
                        );

        /**
         * Remove pieces already placed from available pieces
         * @param p_db Piece database
         * @param p_situation Initial situation
         */
        inline
        void
        record_unavailable_pieces(const emp_piece_db & p_db
                                 ,const emp_FSM_situation & p_situation
                                 ) const;

        /**
         * Create all system equations variables by taking in account hints and
         * initial situation constraints
         * @param p_db Piece database info
         */
        inline
        void
        create_variables(const emp_piece_db & p_db
                        ,const emp_FSM_situation & p_situation
                        );

        /**
         * Indicate if position defined by parameters is corner/border/center
         * @param p_x column index
         * @param p_y row index
         * @return kind of position: corner/border/center
         */
        inline
        emp_types::t_kind get_position_kind(const unsigned int & p_x
                                           ,const unsigned int & p_y
                                           ) const;

        /**
           Compute index related to position X,Y
           @param X position
           @param Y position
           @return index related to position
        */
        inline
        unsigned int get_position_index(const unsigned int & p_x
                                       ,const unsigned int & p_y
                                       ) const;

        /**
         * Reference on database of pieces composing puzzle
         */
        const emp_piece_db & m_db;

        /**
         * Reference on puzzle info
         */
        const emp_FSM_info & m_info;

        /**
         * Hints used to compute possible variables:
         * _ positionned pieces
         * _ positionned and oriented pieces
         */
        std::string m_hints;

        /**
         * Positionned and oriented pieces used to compute possible variables:
         */
        std::string m_initial_situation;

        /**
         * Variables related to the puzzle ( Piece id, orientation and
         * Position)
         */
        std::vector<simplex_variable*> m_simplex_variables;

        /**
         * Store all simplex variables related to a position index
         * Position index = width * Y + X)
         */
        std::vector<simplex_variable*> * m_position_variables;

        /**
         * Positions where a piece is placed at the beginning: hint
         */
        std::map<std::pair<unsigned int, unsigned int>, unsigned int> m_position_hint;

        /**
         * Pieces whose position is known at the beginning: hint
         */
        std::map<unsigned int, std::pair<unsigned int, unsigned int>> m_piece_hint;

        /**
         * Bitfield representing available corner pieces ( 1bit per orientation)
         */
        emp_types::bitfield m_available_corners;

        /**
         * Bitfield representing available border pieces ( 1bit per orientation)
         */
        emp_types::bitfield m_available_borders;

        /**
         * Bitfield representing available center pieces ( 1bit per orientation)
         */
        emp_types::bitfield m_available_centers;

        /**
         * Bitfield array representing available pieces per type where type
         * is used as array index
         */
        emp_types::bitfield * const m_available_pieces[3];

    };

    //-------------------------------------------------------------------------
    emp_variable_generator::emp_variable_generator(const edge_matching_puzzle::emp_piece_db & p_db
                                                  ,const emp_FSM_info & p_info
                                                  ,std::string p_hints
                                                  ,emp_FSM_situation & p_situation
                                                  )
    : m_db(p_db)
    , m_info(p_info)
    , m_hints(std::move(p_hints))
    , m_initial_situation{m_hints}
    , m_position_variables(new std::vector<simplex_variable*>[p_info.get_width() * p_info.get_height()])
    , m_available_corners(4 * p_db.get_nb_pieces(emp_types::t_kind::CORNER),true)
    , m_available_borders(4 * p_db.get_nb_pieces(emp_types::t_kind::BORDER),true)
    , m_available_centers(4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER),true)
    , m_available_pieces{&m_available_centers, &m_available_borders, &m_available_corners}

    {
        if(!m_hints.empty())
        {
            extract_hints();
        }

        if(!m_initial_situation.empty())
        {
            p_situation.set(m_initial_situation);
        }

        // Remove pieces used in hints from bitfield
        record_unavailable_pieces(p_db, p_situation);

        // Create problem variables
        create_variables(p_db, p_situation);

    }

    //-------------------------------------------------------------------------
    void
    emp_variable_generator::extract_hints()
    {
        // Search if some pieces have a determined position but no orientation
        unsigned int l_piece_width = emp_FSM_situation::get_piece_representation_width();
        assert(!(m_initial_situation.size() % l_piece_width));
        for(unsigned int l_index = 0; l_index < m_info.get_height() * m_info.get_width(); ++l_index)
        {
            char l_orientation = m_initial_situation[l_piece_width * (l_index + 1) - 1];
            if(' ' == l_orientation)
            {
                unsigned int l_x = l_index % m_info.get_width();
                unsigned int l_y = l_index / m_info.get_width();
                auto l_piece_id = (emp_types::t_piece_id) std::stoul(m_initial_situation.substr(l_index * l_piece_width, l_piece_width - 1));
                std::cout << "Hint " << l_piece_id << " @(" << l_x << "," << l_y << ")" << std::endl;
                m_initial_situation.replace(l_piece_width * l_index, l_piece_width, std::string(l_piece_width, '-'));
                record_hint(l_piece_id, l_x, l_y);
            }
        }

    }

    //-------------------------------------------------------------------------
    void
    emp_variable_generator::record_hint(unsigned int p_piece_id
                                       ,unsigned int p_x
                                       ,unsigned int p_y
                                       )
    {
        assert(m_piece_hint.end() == m_piece_hint.find(p_piece_id));
        m_piece_hint.insert({p_piece_id, {p_x, p_y}});
        assert(m_position_hint.end() == m_position_hint.find({p_x, p_y}));
        m_position_hint.insert({{p_x, p_y}, p_piece_id});
    }

    //-------------------------------------------------------------------------
    void
    emp_variable_generator::record_unavailable_pieces(const emp_piece_db & p_db
                                                     ,const emp_FSM_situation & p_situation
                                                     ) const
    {
        // Mark used piece as unavailable
        for(unsigned int l_y = 0;
            l_y < m_info.get_height();
            ++l_y
                )
        {
            for(unsigned int l_x = 0;
                l_x < m_info.get_width();
                ++l_x
                    )
            {
                if(p_situation.contains_piece(l_x, l_y))
                {
                    const emp_types::t_oriented_piece & l_piece = p_situation.get_piece(l_x, l_y);
                    emp_types::t_piece_id l_id = l_piece.first;
                    m_available_pieces[(unsigned int)p_db.get_piece(l_id).get_kind()]->set(0x0, 4, 4 * p_db.get_kind_index(l_id));
                }
            }
        }

        // Mark hint pieces as unavailable
        for(auto l_iter: m_piece_hint)
        {
            m_available_pieces[(unsigned int)p_db.get_piece(l_iter.first).get_kind()]->set(0x0, 4, 4 * p_db.get_kind_index(l_iter.first));
        }

    }

    //-------------------------------------------------------------------------
    const std::string &
    emp_variable_generator::get_initial_situation_str() const
    {
        return m_initial_situation;
    }

    //-------------------------------------------------------------------------
    void
    emp_variable_generator::create_variables(const emp_piece_db & p_db
                                            ,const emp_FSM_situation & p_situation
                                            )
    {
        emp_types::bitfield l_matching_corners(4 * p_db.get_nb_pieces(emp_types::t_kind::CORNER));
        emp_types::bitfield l_matching_borders(4 * p_db.get_nb_pieces(emp_types::t_kind::BORDER));
        emp_types::bitfield l_matching_centers(4 * p_db.get_nb_pieces(emp_types::t_kind::CENTER));
        emp_types::bitfield * const l_matching_pieces[3] = {&l_matching_centers,&l_matching_borders,&l_matching_corners};

        // Determine for each position which piece match constraints
        for(unsigned int l_y = 0;
            l_y < m_info.get_height();
            ++l_y
                )
        {
            for(unsigned int l_x = 0;
                l_x < m_info.get_width();
                ++l_x
                    )
            {
                emp_types::t_kind l_type = get_position_kind(l_x,l_y);
                emp_types::bitfield l_possible_neighborhood(4 * p_db.get_nb_pieces(l_type), true);

                const auto l_position_hint_iter = m_position_hint.find({l_x, l_y});
                if(m_position_hint.end() != l_position_hint_iter)
                {
                    for(auto l_orientation = (unsigned int)emp_types::orientation::NORTH;
                        l_orientation <= (unsigned int)emp_types::orientation::WEST;
                        ++l_orientation
                            )
                    {
                        simplex_variable *l_variable = new simplex_variable((unsigned int) m_simplex_variables.size()
                                ,l_x
                                ,l_y
                                ,l_position_hint_iter->second
                                ,(emp_types::orientation) l_orientation
                                                                           );
                        m_simplex_variables.push_back(l_variable);
                    }
                }
                else if(!p_situation.contains_piece(l_x,l_y))
                {
                    // Compute surrounding constraints

                    // Compute EAST constraint
                    emp_types::t_binary_piece l_east_constraint = 0x0;
                    if(l_x < m_info.get_width() - 1)
                    {
                        if(p_situation.contains_piece(l_x + 1,l_y))
                        {
                            const emp_types::t_oriented_piece l_oriented_piece = p_situation.get_piece(l_x + 1 ,l_y);
                            const emp_piece & l_east_piece = p_db.get_piece(l_oriented_piece.first);
                            l_east_constraint = l_east_piece.get_color(emp_types::t_orientation::WEST,l_oriented_piece.second);
                        }
                        else
                        {
                            const auto l_neighbor_hint_iter = m_position_hint.find({l_x + 1, l_y});
                            if(m_position_hint.end() != l_neighbor_hint_iter)
                            {
                                l_possible_neighborhood.apply_and(l_possible_neighborhood
                                        ,m_db.compute_possible_neighborhood(l_type
                                                ,l_neighbor_hint_iter->second
                                                ,emp_types::t_orientation::EAST
                                                                           )
                                                                 );
                            }
                        }
                    }
                    else
                    {
                        l_east_constraint = p_db.get_border_color_id();
                    }
                    emp_types::t_binary_piece l_constraint = l_east_constraint;

                    // Compute NORTH constraint
                    emp_types::t_binary_piece l_north_constraint = 0x0;
                    if(l_y)
                    {
                        if(p_situation.contains_piece(l_x,l_y - 1))
                        {
                            const emp_types::t_oriented_piece l_oriented_piece = p_situation.get_piece(l_x,l_y - 1);
                            const emp_piece & l_north_piece = p_db.get_piece(l_oriented_piece.first);
                            l_north_constraint = l_north_piece.get_color(emp_types::t_orientation::SOUTH,l_oriented_piece.second);
                        }
                        else
                        {
                            const auto l_neighbor_hint_iter = m_position_hint.find({l_x, l_y - 1});
                            if(m_position_hint.end() != l_neighbor_hint_iter)
                            {
                                l_possible_neighborhood.apply_and(l_possible_neighborhood
                                        ,m_db.compute_possible_neighborhood(l_type
                                                ,l_neighbor_hint_iter->second
                                                ,emp_types::t_orientation::NORTH
                                                                           )
                                                                 );
                            }
                        }
                    }
                    else
                    {
                        l_north_constraint = p_db.get_border_color_id();
                    }
                    l_constraint = (l_constraint << p_db.get_color_id_size()) | l_north_constraint;

                    // Compute WEST constraint
                    emp_types::t_binary_piece l_west_constraint = 0x0;
                    if(l_x > 0)
                    {
                        if(p_situation.contains_piece(l_x - 1,l_y))
                        {
                            const emp_types::t_oriented_piece l_oriented_piece = p_situation.get_piece(l_x - 1,l_y);
                            const emp_piece & l_west_piece = p_db.get_piece(l_oriented_piece.first);
                            l_west_constraint = l_west_piece.get_color(emp_types::t_orientation::EAST,l_oriented_piece.second);
                        }
                        else
                        {
                            const auto l_neighbor_hint_iter = m_position_hint.find({l_x - 1, l_y});
                            if(m_position_hint.end() != l_neighbor_hint_iter)
                            {
                                l_possible_neighborhood.apply_and(l_possible_neighborhood
                                        ,m_db.compute_possible_neighborhood(l_type
                                                ,l_neighbor_hint_iter->second
                                                , emp_types::t_orientation::WEST
                                                                           )
                                                                 );
                            }
                        }
                    }
                    else
                    {
                        l_west_constraint = p_db.get_border_color_id();
                    }
                    l_constraint = (l_constraint << p_db.get_color_id_size()) | l_west_constraint;

                    // Compute SOUTH constraint
                    emp_types::t_binary_piece l_south_constraint = 0x0;
                    if(l_y < m_info.get_height() - 1)
                    {
                        if(p_situation.contains_piece(l_x,l_y + 1))
                        {
                            const emp_types::t_oriented_piece l_oriented_piece = p_situation.get_piece(l_x, l_y + 1);
                            const emp_piece & l_south_piece = p_db.get_piece(l_oriented_piece.first);
                            l_south_constraint = l_south_piece.get_color(emp_types::t_orientation::NORTH,l_oriented_piece.second);
                        }
                        else
                        {
                            const auto l_neighbor_hint_iter = m_position_hint.find({l_x, l_y + 1});
                            if(m_position_hint.end() != l_neighbor_hint_iter)
                            {
                                l_possible_neighborhood.apply_and(l_possible_neighborhood
                                        ,m_db.compute_possible_neighborhood(l_type
                                                ,l_neighbor_hint_iter->second
                                                ,emp_types::t_orientation::SOUTH
                                                                           )
                                                                 );
                            }
                        }

                    }
                    else
                    {
                        l_south_constraint = p_db.get_border_color_id();
                    }
                    l_constraint = (l_constraint << p_db.get_color_id_size()) | l_south_constraint;

                    // Compute pieces matching to constraint
                    l_matching_pieces[(unsigned int)l_type]->apply_and(p_db.get_pieces(l_constraint),*m_available_pieces[(unsigned int)l_type]);

                    // Filter with possible neigborhood related to hints
                    l_matching_pieces[(unsigned int)l_type]->apply_and(*l_matching_pieces[(unsigned int)l_type], l_possible_neighborhood);

                    // Iterating on matching pieces
                    emp_types::bitfield l_loop_pieces(*l_matching_pieces[(unsigned int)l_type]);
                    int l_ffs = 0;
                    while((l_ffs = l_loop_pieces.ffs()) != 0)
                    {
                        // We decrement because 0 mean no piece in other cases this
                        // is the index of oriented piece in piece list by kind
                        unsigned int l_piece_kind_id = (unsigned int)l_ffs - 1;
                        l_loop_pieces.set(0,1,l_piece_kind_id);
                        const emp_types::t_binary_piece l_piece = p_db.get_piece(l_type,l_piece_kind_id);
                        unsigned int l_truncated_piece = l_piece >> (4 * p_db.get_color_id_size());
                        auto l_orientation = (emp_types::t_orientation)(l_truncated_piece & 0x3);
                        unsigned int l_piece_id = 1 + (l_truncated_piece >> 2);
                        simplex_variable * l_variable = new simplex_variable((unsigned int)m_simplex_variables.size(), l_x, l_y, l_piece_id, l_orientation);
                        m_simplex_variables.push_back(l_variable);
                        m_position_variables[get_position_index(l_x, l_y)].push_back(l_variable);
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------
    emp_types::t_kind
    emp_variable_generator::get_position_kind(const unsigned int & p_x
                                             ,const unsigned int & p_y
                                             ) const
    {
        assert(p_x < m_info.get_width());
        assert(p_y < m_info.get_height());
        emp_types::t_kind l_type = emp_types::t_kind::CENTER;
        if(!p_x || !p_y || m_info.get_width() - 1 == p_x || p_y == m_info.get_height() - 1)
        {
            l_type = emp_types::t_kind::BORDER;
            if((!p_x && !p_y) ||
               (!p_x && p_y == m_info.get_height() - 1) ||
               (!p_y && p_x == m_info.get_width() - 1) ||
               (p_y == m_info.get_height() - 1 && p_x == m_info.get_width() - 1)
                    )
            {
                l_type = emp_types::t_kind::CORNER;
            }
        }
        return l_type;
    }

    //-------------------------------------------------------------------------
    unsigned int
    emp_variable_generator::get_position_index(const unsigned int & p_x
                                              ,const unsigned int & p_y
                                              ) const
    {
        assert(p_x < m_info.get_width());
        assert(p_y < m_info.get_height());
        return m_info.get_width() * p_y + p_x;
    }

    //-------------------------------------------------------------------------
    const std::vector<simplex_variable *> &
    emp_variable_generator::get_variables() const
    {
        return m_simplex_variables;
    }

    //-------------------------------------------------------------------------
    emp_variable_generator::~emp_variable_generator()
    {
        for(auto l_iter: m_simplex_variables)
        {
            delete l_iter;
        }
        delete[] m_position_variables;
    }

    //-------------------------------------------------------------------------
    const std::vector<simplex_variable *> &
    emp_variable_generator::get_position_variables(unsigned int p_index) const
    {
        assert(p_index < m_info.get_width() * m_info.get_height());
        return m_position_variables[p_index];
    }

    //-------------------------------------------------------------------------
    template <typename T>
    void
    emp_variable_generator::treat_piece_relation_equation(const std::vector<simplex_variable *> & p_variables_pos1
                                                         ,const std::vector<simplex_variable *> & p_variables_pos2
                                                         ,bool p_horizontal
                                                         ,T & p_lambda
                                                         )
    {
        emp_types::t_orientation l_border1 = p_horizontal ? emp_types::t_orientation::EAST : emp_types::t_orientation::SOUTH;
        emp_types::t_orientation l_border2 = p_horizontal ? emp_types::t_orientation::WEST : emp_types::t_orientation::NORTH;

        for(auto l_iter_pos1: p_variables_pos1)
        {
            for(auto l_iter_pos2: p_variables_pos2)
            {
                if(l_iter_pos1->get_piece_id() != l_iter_pos2->get_piece_id())
                {
                    emp_types::t_color_id l_color1 = m_db.get_piece(l_iter_pos1->get_piece_id()).get_color(l_border1,l_iter_pos1->get_orientation());
                    emp_types::t_color_id l_color2 = m_db.get_piece(l_iter_pos2->get_piece_id()).get_color(l_border2,l_iter_pos2->get_orientation());

                    if(l_color1 != l_color2)
                    {
                        p_lambda(*l_iter_pos1, *l_iter_pos2);
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------
    template <typename T>
    void
    emp_variable_generator::treat_piece_relations(T & p_lambda)
    {
        for(unsigned int l_y = 0;
            l_y < m_info.get_height();
            ++l_y
                )
        {
            for(unsigned int l_x = 0;
                l_x < m_info.get_width();
                ++l_x
                    )
            {
                if(l_x < m_info.get_width() - 1)
                {
                    treat_piece_relation_equation(m_position_variables[get_position_index(l_x, l_y)]
                                                 ,m_position_variables[get_position_index(l_x + 1, l_y)]
                                                 ,true
                                                 ,p_lambda
                                                 );
                }
                if(l_y < m_info.get_height() - 1)
                {
                    treat_piece_relation_equation(m_position_variables[get_position_index(l_x, l_y)]
                                                 ,m_position_variables[get_position_index(l_x, l_y + 1)]
                                                 ,false
                                                 ,p_lambda
                                                 );
                }
            }
        }
    }
}
#endif //EMP_VARIABLE_GENERATOR_H
