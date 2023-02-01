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
#ifndef EDGE_MATCHING_PUZZLE_CUDA_TYPES_H
#define EDGE_MATCHING_PUZZLE_CUDA_TYPES_H

#include "CUDA_strong_primitive.h"

namespace edge_matching_puzzle
{
    using info_index_t = my_cuda::CUDA_strong_primitive<uint32_t, struct info_index>;
    using position_index_t = my_cuda::CUDA_strong_primitive<uint32_t, struct position_index>;
}
#endif //EDGE_MATCHING_PUZZLE_CUDA_TYPES_H
// EOF