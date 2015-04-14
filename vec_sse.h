#ifndef vec_sse_H
#define vec_sse_H

#include "vec.h"
#include "StaticFor.h"

//! This class stores 4 D-dimensional vectors. All operations are performed using SSE.
template<unsigned int D>
class vec_simd
{
public:
    __m128 componentRegisters[D];

    static const unsigned int REGISTER_SIZE = 4;

    typedef vec<D, float> vecf;

    Inline void setVecs(vecf vectors[4])
    {
        StaticFor<0, D>::step( [&] (unsigned int d) {
            componentRegisters[d] = _mm_setr_ps(vectors[0][d], vectors[1][d], vectors[2][d], vectors[3][d]);
        });
    }

    Inline void setVecs(const vecf& v0, const vecf& v1, const vecf& v2, const vecf& v3)
    {
        StaticFor<0, D>::step([&](unsigned int d) {
            componentRegisters[d] = _mm_setr_ps(v0[d], v1[d], v2[d], v3[d]);
        });
    }

    template<class Getter>
    Inline void setVecs(const Getter& func)
    {
        setVecs(func(0), func(1), func(2), func(3));
    }

    template<class Getter>
    Inline void setVecs(const Getter& func, unsigned int numElements, const vecf& fillValue)
    {
        begin_disable_warnings      // Yes, we may read uninitialized memory here. No, we do not plan do use it.
        setVecs(func(0),
                (numElements > 1) ? func(1) : fillValue,
                (numElements > 2) ? func(2) : fillValue,
                (numElements > 3) ? func(3) : fillValue);
        end_disable_warnings
    }

    template<class Getter>
    Inline void setScalars(const Getter& func)
    {
        componentRegisters[0] = _mm_setr_ps(func(0), func(1), func(2), func(3));
    }

    template<class Getter>
    Inline void setScalars(const Getter& func, unsigned int numElements, float fillValue)
    {
        begin_disable_warnings      // Yes, we may read uninitialized memory here. No, we do not plan do use it.
        componentRegisters[0] = _mm_setr_ps(func(0),
                (numElements > 1) ? func(1) : fillValue,
                (numElements > 2) ? func(2) : fillValue,
                (numElements > 3) ? func(3) : fillValue);
        end_disable_warnings
    }

    vec_simd() {}

    //! Scalar constructor.
    //! These template programming techniques are a world full of wonders...
    template <unsigned int E = D> vec_simd(float scalar, typename std::enable_if<E == 1, int>::type* = 0)
    {
        componentRegisters[0] = _mm_set1_ps(scalar);
    }

    template <unsigned int E = D> vec_simd(const vecf& v, typename std::enable_if<E >= 2, int>::type* = 0)
    {
        StaticFor<0, D>::step([&](unsigned int d) { componentRegisters[d] = _mm_set1_ps(v[d]); });
    }

    void storeVecs(vecf& v0, vecf& v1, vecf& v2, vecf& v3) const
    {
        float buffer[4];
        StaticFor<0, D>::step([&](unsigned int d) {
            _mm_storeu_ps(buffer, componentRegisters[d]);
            v0[d] = buffer[0];
            v1[d] = buffer[1];
            v2[d] = buffer[2];
            v3[d] = buffer[3];
        });
    }

    template<class Getter>
    Inline void storeVecs(const Getter& func)
    {
        storeVecs(func(0), func(1), func(2), func(3));
    }

    void store(float* values, unsigned int d = 0) const
    {
        _mm_storeu_ps(values, componentRegisters[d]);
    }

    void load(const float* values, unsigned int d = 0)
    {
        componentRegisters[d] = _mm_loadu_ps(values);
    }

    template<class Getter>
    Inline void storeScalars(const Getter& func)
    {
        float values[4];
        store(values, 0);
        func(0) = values[0];
        func(1) = values[1];
        func(2) = values[2];
        func(3) = values[3];
    }

    vec<4, float> operator()(unsigned int d) const
    {
        vec<4, float> out;
        store(&out[0], d);
        return out;
    }

    void setZero()
    {
        StaticFor<0, D>::step([&](unsigned int d) { componentRegisters[d] = _mm_setzero_ps(); });
    }

    // Operators
    vec_simd<D>& operator+=(const vec_simd<D>& vector)
    {
        StaticFor<0, D>::step( [&] (unsigned int d) { componentRegisters[d] = _mm_add_ps(componentRegisters[d], vector.componentRegisters[d]); });
        return *this;
    }
    vec_simd<D>& operator-=(const vec_simd<D>& vector)
    {
        StaticFor<0, D>::step( [&] (unsigned int d) { componentRegisters[d] = _mm_sub_ps(componentRegisters[d], vector.componentRegisters[d]); });
        return *this;
    }
    vec_simd<D>& operator*=(const vec_simd<1>& scalar)
    {
        StaticFor<0, D>::step( [&] (unsigned int d) { componentRegisters[d] = _mm_mul_ps(componentRegisters[d], scalar.componentRegisters[0]); });
        return *this;
    }
    vec_simd<D>& operator/=(const vec_simd<1>& scalar)
    {
        StaticFor<0, D>::step( [&] (unsigned int d) { componentRegisters[d] = _mm_div_ps(componentRegisters[d], scalar.componentRegisters[0]); });
        return *this;
    }

    Inline vec_simd<D> elementWiseMultiplication(const vec_simd<D>& other) const
    {
        vec_simd<D> out;
        StaticFor<0, D>::step( [&] (unsigned int d) { out.componentRegisters[d] = _mm_mul_ps(componentRegisters[d], other.componentRegisters[d]); });
        return out;
    }

    vec_simd<D> elementWiseDivision(const vec_simd<D>& other) const
    {
        vec_simd<D> out;
        StaticFor<0, D>::step([&](unsigned int d) { out.componentRegisters[d] = _mm_div_ps(componentRegisters[d], other.componentRegisters[d]); });
        return out;
    }

    Inline vec_simd<1> dotProduct(const vec_simd<D>& other) const
    {
        vec_simd<1> out;
        out.componentRegisters[0] = _mm_mul_ps(componentRegisters[0], other.componentRegisters[0]);
        StaticFor<1, D>::step([&](unsigned int d) {
            out.componentRegisters[0] = _mm_add_ps(out.componentRegisters[0], _mm_mul_ps(componentRegisters[d], other.componentRegisters[d]));
        });
        return out;
    }

    void sqrt()
    {
        StaticFor<0, D>::step( [&] (unsigned int d) { componentRegisters[d] = _mm_sqrt_ps(componentRegisters[d]); });
    }

    static vec_simd<1> sqrt(vec_simd<1> in)
    {
        in.sqrt();
        return in;
    }

    vec_simd<1> length()
    {
        vec_simd<1> out = dotProduct(*this);
        out.sqrt();
        return out;
    }

    //! For each register, the 4 values are summed up.
    Inline vecf horizontalAdd() const
    {
        vecf out;
        StaticFor<0, D>::step([&](unsigned int d){
            __m128 x = _mm_hadd_ps(componentRegisters[d], componentRegisters[d]);
            x = _mm_hadd_ps(x, x);
            _mm_store_ss(&out[d], x);
        });
        return out;
    }

    // Vec3 specific functionality. Includes cross product, but also optimized load and store methods.
    template <unsigned int E = D>
    static typename std::enable_if<E == 3, vec_simd<3> >::type crossProduct(const vec_simd<3>& v1, const vec_simd<3>& v2)
    {
        vec_simd<3> out;
        out.componentRegisters[0] = _mm_sub_ps(_mm_mul_ps(v1.componentRegisters[1], v2.componentRegisters[2]), _mm_mul_ps(v1.componentRegisters[2], v2.componentRegisters[1]));
        out.componentRegisters[1] = _mm_sub_ps(_mm_mul_ps(v1.componentRegisters[2], v2.componentRegisters[0]), _mm_mul_ps(v1.componentRegisters[0], v2.componentRegisters[2]));
        out.componentRegisters[2] = _mm_sub_ps(_mm_mul_ps(v1.componentRegisters[0], v2.componentRegisters[1]), _mm_mul_ps(v1.componentRegisters[1], v2.componentRegisters[0]));
        return out;
    }

    template <unsigned int E = D>
    Inline void setVecs3(const typename std::enable_if<E == 3, vec<3, float> >::type& v0, const typename std::enable_if<E == 3, vec<3, float> >::type& v1, const typename std::enable_if<E == 3, vec<3, float> >::type& v2, const typename std::enable_if<E == 3, vec<3, float> >::type& v3)
    {
        __m128 xy0, xy1;
        xy0 = xy1 = _mm_setzero_ps();
        xy0 = _mm_loadl_pi(xy0, (const __m64*)&v0[0]);
        xy1 = _mm_loadl_pi(xy1, (const __m64*)&v2[0]);
        xy0 = _mm_loadh_pi(xy0, (const __m64*)&v1[0]);
        xy1 = _mm_loadh_pi(xy1, (const __m64*)&v3[0]);
        componentRegisters[0] = _mm_shuffle_ps(xy0, xy1, _MM_SHUFFLE(2, 0, 2, 0));
        componentRegisters[1] = _mm_shuffle_ps(xy0, xy1, _MM_SHUFFLE(3, 1, 3, 1));
        componentRegisters[2] = _mm_setr_ps(v0[2], v1[2], v2[2], v3[2]);
    }
    template<class Getter>
    Inline void setVecs3(const Getter& func)
    {
        setVecs3(func(0), func(1), func(2), func(3));
    }

    template<class Getter>
    Inline void setVecs3(const Getter& func, unsigned int numElements, const vecf& fillValue)
    {
        begin_disable_warnings      // Yes, we may read uninitialized memory here. No, we do not plan do use it.
        setVecs3(func(0),
                (numElements > 1) ? func(1) : fillValue,
                (numElements > 2) ? func(2) : fillValue,
                (numElements > 3) ? func(3) : fillValue);
        end_disable_warnings
    }

    //! Optimized version of horizontal add for 3D-vectors. Performs four _mm_hadd_ps instead of six.
    template <unsigned int E = D>
    Inline typename std::enable_if<E == 3, vec<3, float> >::type horizontalAdd3() const
    {
        vec<3, float> out;
        __m128 xy = _mm_hadd_ps(componentRegisters[0], componentRegisters[1]);
        __m128 z =  _mm_hadd_ps(componentRegisters[2], componentRegisters[2]);
        xy = _mm_hadd_ps(xy, xy);
        z = _mm_hadd_ps(z, z);
        _mm_storel_pi((__m64*)&out[0], xy);
        _mm_store_ss(&out[2], z);
        return out;
    }
};

template<unsigned int D>
vec_simd<D> operator+(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_add_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}

template<unsigned int D>
vec_simd<D> operator-(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_sub_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}

template<unsigned int D>
vec_simd<D> operator*(const vec_simd<1>& scalar, const vec_simd<D>& v)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&scalar, &v, &out] (unsigned int d) { out.componentRegisters[d] = _mm_mul_ps(scalar.componentRegisters[0], v.componentRegisters[d]); });
    return out;
}

template<unsigned int D>
typename std::enable_if<D != 1, vec_simd<D> >::type operator*(const vec_simd<D>& v, const vec_simd<1>& scalar)
{
    return scalar * v;
}

template<unsigned int D>
vec_simd<D> operator/(const vec_simd<D>& v, const vec_simd<1>& scalar)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&scalar, &v, &out] (unsigned int d) { out.componentRegisters[d] = _mm_div_ps(v.componentRegisters[d], scalar.componentRegisters[0]); });
    return out;
}

template<unsigned int D>
vec_simd<D> operator-(const vec_simd<D>& v)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v, &out] (unsigned int d) { out.componentRegisters[d] = _mm_xor_ps(v.componentRegisters[d], _mm_set1_ps(-0.f)); });
    return out;
}

// Bitwise operations (sometimes useful to eliminate branching).
template<unsigned int D>
vec_simd<D> operator&(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_and_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator|(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_or_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator^(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_xor_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}

// Comparison operators
template<unsigned int D>
vec_simd<D> operator>(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_cmpgt_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator>=(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_cmpge_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator<(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_cmplt_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator<=(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_cmple_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator==(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm_cmpeq_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}

#endif // vec_sse_H
