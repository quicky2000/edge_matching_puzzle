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
#ifndef EDGE_MATCHING_PUZZLE_CUDA_STRONG_PRIMITIVE_H
#define EDGE_MATCHING_PUZZLE_CUDA_STRONG_PRIMITIVE_H

#include "my_cuda.h"

namespace edge_matching_puzzle
{
    template <typename T, typename PHANTOM>
    class CUDA_strong_primitive;

    template <typename T, typename PHANTOM>
    std::ostream &
    operator<<(std::ostream &, const CUDA_strong_primitive<T, PHANTOM> &);

    template <typename T, typename PHANTOM>
    class CUDA_strong_primitive
    {
      public:

        template <typename U, typename PHANTOM_U>
        friend
        std::ostream &
        operator<<(std::ostream &
                  ,const CUDA_strong_primitive<U, PHANTOM_U> &
                  );

        inline
        explicit
        __host__ __device__
        CUDA_strong_primitive()
#ifndef ENABLE_CUDA_CODE
        = default
#endif // ENABLE_CUDA_CODE
        ;

        inline
        explicit
        __host__ __device__
        CUDA_strong_primitive(const T & p_value);

        inline
        explicit
        __host__ __device__
        CUDA_strong_primitive(T && p_value);

        inline
        explicit
        __host__ __device__
        operator T() const;

        inline
        __host__ __device__
        T & operator=(T);

        inline
        __host__ __device__
        bool
        operator<(T) const;

        inline
        __host__ __device__
        bool
        operator<(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        bool
        operator<=(T) const;

        inline
        __host__ __device__
        bool
        operator<=(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        bool
        operator==(T) const;

        inline
        __host__ __device__
        bool
        operator==(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        bool
        operator!=(T) const;

        inline
        __host__ __device__
        bool
        operator!=(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator+(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator+(T) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator-(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator-(T) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator/(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator/(T) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator*(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator*(T) const;

        // prefix increment
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM> &
        operator++();

      private:
        // Make get private to force the use of an explicit cast
        inline
        __host__ __device__
        const T & get() const;

        inline
        __host__ __device__
        T & get();


        T m_value;
    };

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>::CUDA_strong_primitive(const T & p_value)
    :m_value{p_value}
    {

    }

    //-------------------------------------------------------------------------
    template <typename T,typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>::CUDA_strong_primitive(T && p_value)
    :m_value(std::move(p_value))
    {
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    const T &
    CUDA_strong_primitive<T, PHANTOM>::get() const
    {
        return m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    T &
    CUDA_strong_primitive<T, PHANTOM>::get()
    {
        return m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>::operator T() const
    {
        return m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    T &
    CUDA_strong_primitive<T, PHANTOM>::operator=(T p_value)
    {
        m_value = p_value;
        return *this;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<(T p_value) const
    {
        return m_value < p_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value < p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<=(T p_operand) const
    {
        return m_value <= p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<=(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value <= p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM> &
    CUDA_strong_primitive<T, PHANTOM>::operator++()
    {
        ++m_value;
        return *this;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator==(T p_value) const
    {
        return m_value == p_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator==(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value == p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator!=(T p_value) const
    {
        return m_value != p_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator!=(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value != p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator+(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value + p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator+(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value + p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator-(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value - p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator-(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value - p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator/(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value / p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator/(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value / p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator*(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value * p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator*(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value * p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    std::ostream &
    operator<<(std::ostream & p_stream
              ,const CUDA_strong_primitive<T, PHANTOM> & p_operand
              )
    {
        p_stream << p_operand.m_value;
        return p_stream;
    }

}

#endif //EDGE_MATCHING_PUZZLE_CUDA_STRONG_PRIMITIVE_H
// EOF