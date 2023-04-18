/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2021  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_STACK_DUMPER_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_STACK_DUMPER_H

#include "CUDA_glutton_max_stack.h"
#include "CUDA_memory_managed_array.h"
#include "xmlParser.h"
#include "emp_types.h"
#include <string>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace edge_matching_puzzle
{
    class CUDA_glutton_stack_dumper
    {
      public:

        inline explicit
        CUDA_glutton_stack_dumper(std::string p_name);

        inline
        void
        dump(const CUDA_glutton_max_stack & p_stack);

      private:

        void
        dump_size(uint32_t p_size
                 ,XMLNode & p_node
                 );

        void
        dump_level(uint32_t p_level
                  ,XMLNode & p_node
                  );

        void
        dump_nb_pieces(uint32_t p_nb_pieces
                      ,XMLNode & p_node
                      );

        void
        dump_info_to_position(uint32_t p_size
                             ,const my_cuda::CUDA_memory_managed_array<position_index_t> & p_info_to_position
                             ,XMLNode & p_node
                             );

        void
        dump_position_to_info(uint32_t p_size
                             ,const my_cuda::CUDA_memory_managed_array<info_index_t> & p_position_to_info
                             ,XMLNode & p_node
                             );

        void
        dump_played_info(uint32_t p_size
                        ,const my_cuda::CUDA_memory_managed_array<CUDA_glutton_max_stack::played_info_t> & p_played_info
                        ,XMLNode & p_node
                        );

        void
        dump_available_pieces(const uint32_t (&p_available_pieces)[8]
                             ,XMLNode & p_node
                             );

        void
        dump_position_infos(const CUDA_glutton_max_stack & p_stack
                           ,XMLNode & p_node
                           );

        void
        dump_level(uint32_t p_level
                  ,const CUDA_glutton_max_stack & p_stack
                  ,XMLNode & p_node
                  );

        void
        dump(uint32_t p_index
            ,const CUDA_piece_position_info2 & p_position_info
            ,XMLNode & p_node
            );

        template<typename T>
        void
        dump_array(std::string p_name
                  ,uint32_t p_size
                  ,const T & p_info_to_position
                  ,XMLNode & p_node
                  );

        std::string m_name;
    };

    //-------------------------------------------------------------------------
    CUDA_glutton_stack_dumper::CUDA_glutton_stack_dumper(std::string p_name)
    :m_name{std::move(p_name)}
    {

    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump(const CUDA_glutton_max_stack & p_stack)
    {
        XMLNode l_root_node = XMLNode::createXMLTopNode("CUDA_glutton_stack");
        dump_size(static_cast<uint32_t>(p_stack.get_size()), l_root_node);
        dump_level(p_stack.get_level(), l_root_node);
        dump_nb_pieces(static_cast<uint32_t>(p_stack.get_nb_pieces()), l_root_node);
        dump_info_to_position(p_stack.get_size(), p_stack.m_info_index_to_position_index, l_root_node);
        dump_position_to_info(static_cast<uint32_t>(p_stack.get_nb_pieces()), p_stack.m_position_index_to_info_index, l_root_node);
        dump_played_info(p_stack.get_size(), p_stack.m_played_info, l_root_node);
        dump_available_pieces(p_stack.m_available_pieces, l_root_node);
        dump_position_infos(p_stack, l_root_node);
        l_root_node.writeToFile(m_name.c_str());
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_position_infos(const CUDA_glutton_max_stack & p_stack
                                                  ,XMLNode & p_node
                                                  )
    {
        XMLNode l_node = p_node.addChild("position_infos");
        for(uint32_t l_index = 0; l_index < p_stack.get_size(); ++l_index)
        {
            dump_level(l_index, p_stack, l_node);
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_level(uint32_t p_level
                                         ,const CUDA_glutton_max_stack & p_stack
                                         ,XMLNode & p_node
                                         )
    {
        XMLNode l_node = p_node.addChild("level");
        l_node.addAttribute("index", std::to_string(p_level).c_str());
        for(uint32_t l_index = 0; l_index < p_stack.get_size() - p_level; ++l_index)
        {
            dump(l_index, p_stack.get_position_info(p_level, static_cast<info_index_t>(l_index)), l_node);
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump(uint32_t p_index
                                   ,const CUDA_piece_position_info2 & p_position_info
                                   ,XMLNode & p_node
                                   )
    {
        XMLNode l_node = p_node.addChild("position_info");
        l_node.addAttribute("index", std::to_string(p_index).c_str());
        for(uint32_t l_index = 0; l_index < 32; ++l_index)
        {
            std::stringstream l_stream;
            l_stream << "0x" << std::hex << std::setfill('0') << std::setw(8) << p_position_info.get_word(l_index);
            XMLNode l_item_node = l_node.addChild("word");
            l_item_node.addText(l_stream.str().c_str());
            l_item_node.addAttribute("index", std::to_string(l_index).c_str());
        }
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_size(uint32_t p_size
                                        ,XMLNode & p_node
                                        )
    {
        p_node.addAttribute("size", std::to_string(p_size).c_str());
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_level(uint32_t p_level
                                         ,XMLNode & p_node
                                         )
    {
        p_node.addAttribute("level", std::to_string(p_level).c_str());
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_nb_pieces(uint32_t p_nb_pieces
                                             ,XMLNode & p_node
                                             )
    {
        p_node.addAttribute("nb_pieces", std::to_string(p_nb_pieces).c_str());
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_info_to_position(uint32_t p_size
                                                    ,const my_cuda::CUDA_memory_managed_array<position_index_t> & p_info_to_position
                                                    ,XMLNode & p_node
                                                    )
    {
        dump_array("info_to_position", p_size, p_info_to_position, p_node);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_position_to_info(uint32_t p_size
                                                    ,const my_cuda::CUDA_memory_managed_array<info_index_t> & p_position_to_info
                                                    ,XMLNode & p_node
                                                    )
    {
        dump_array("position_to_info", p_size, p_position_to_info, p_node);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_played_info(uint32_t p_size
                                               ,const my_cuda::CUDA_memory_managed_array<CUDA_glutton_max_stack::played_info_t> & p_played_info
                                               ,XMLNode & p_node
                                               )
    {
        dump_array("played_info", p_size, p_played_info, p_node);
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_stack_dumper::dump_available_pieces(const uint32_t (&p_available_pieces)[8]
                                                    ,XMLNode & p_node
                                                    )
    {
        dump_array("available_pieces", 8, p_available_pieces, p_node);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void
    CUDA_glutton_stack_dumper::dump_array(std::string p_name
                                         ,uint32_t p_size
                                         ,const T & p_info_to_position
                                         ,XMLNode & p_node
                                         )
    {
        XMLNode l_array_node = p_node.addChild(p_name.c_str());
        l_array_node.addAttribute("size", std::to_string(p_size).c_str());
        for(uint32_t l_index = 0; l_index < p_size; ++l_index)
        {
            std::stringstream l_stream;
            l_stream << "0x" << std::hex << std::setfill('0') << std::setw(8) << p_info_to_position[l_index];
            XMLNode l_item_node = l_array_node.addChild("item");
            l_item_node.addText(l_stream.str().c_str());
            l_item_node.addAttribute("index", std::to_string(l_index).c_str());
        }
    }

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_STACK_DUMPER_H
// EOF