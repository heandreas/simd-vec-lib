#ifndef vec_avx_H
#define vec_avx_H

#include "vec.h"
#include "StaticFor.h"

//! This class stores 4 D-dimensional vectors. All operations are performed using SSE.
template<unsigned int D>
class vec_simd
{
public:
    __m256 componentRegisters[D];

    static const unsigned int REGISTER_SIZE = 8;

    Inline void setVecs(const vec<D, float>& v0, const vec<D, float>& v1, const vec<D, float>& v2, const vec<D, float>& v3, const vec<D, float>& v4, const vec<D, float>& v5, const vec<D, float>& v6, const vec<D, float>& v7)
    {
        StaticFor<0, D>::step( [&] (unsigned int d) { this->componentRegisters[d] = _mm256_setr_ps(v0[d], v1[d], v2[d], v3[d], v4[d], v5[d], v6[d], v7[d]); });
    }

    template<class Getter>
    Inline void setVecs(const Getter& func)
    {
        setVecs(func(0), func(1), func(2), func(3), func(4), func(5), func(6), func(7));
    }

    template<class Getter>
    Inline void setVecs(const Getter& func, unsigned int numElements, const vec<D, float>& fillValue)
    {
        begin_disable_warnings      // Yes, we may read uninitialized memory here. No, we do not plan do use it.
        setVecs(func(0),
                (numElements > 1) ? func(1) : fillValue,
                (numElements > 2) ? func(2) : fillValue,
                (numElements > 3) ? func(3) : fillValue,
                (numElements > 4) ? func(4) : fillValue,
                (numElements > 5) ? func(5) : fillValue,
                (numElements > 6) ? func(6) : fillValue,
                (numElements > 7) ? func(7) : fillValue);
        end_disable_warnings
    }

    template<class Getter>
    Inline void setScalars(const Getter& func)
    {
        componentRegisters[0] = _mm256_setr_ps(func(0), func(1), func(2), func(3), func(4), func(5), func(6), func(7));
    }

    template<class Getter>
    Inline void setScalars(const Getter& func, unsigned int numElements, float fillValue)
    {
        begin_disable_warnings      // Yes, we may read uninitialized memory here. No, we do not plan do use it.
                componentRegisters[0] = _mm256_setr_ps(func(0),
                                   (numElements > 1) ? func(1) : fillValue,
                                   (numElements > 2) ? func(2) : fillValue,
                                   (numElements > 3) ? func(3) : fillValue,
                                   (numElements > 4) ? func(4) : fillValue,
                                   (numElements > 5) ? func(5) : fillValue,
                                   (numElements > 6) ? func(6) : fillValue,
                                   (numElements > 7) ? func(7) : fillValue);
        end_disable_warnings
    }

    vec_simd() {}

    //! Scalar constructor.
    //! These template programming techniques are a world full of wonders...
    template <unsigned int E = D> vec_simd(float scalar, typename std::enable_if<E == 1, int>::type* = 0)
    {
        componentRegisters[0] = _mm256_set1_ps(scalar);
    }

    template <unsigned int E = D> vec_simd(const vec<D, float>& v, typename std::enable_if<E >= 2, int>::type* = 0)
    {
        StaticFor<0, D>::step([this, &v](unsigned int d) { componentRegisters[d] = _mm256_set1_ps(v[d]); });
    }

    void getVecs(vec<D, float>& v0, vec<D, float>& v1, vec<D, float>& v2, vec<D, float>& v3, vec<D, float>& v4, vec<D, float>& v5, vec<D, float>& v6, vec<D, float>& v7) const
    {
        float buffer[8];
        StaticFor<0, D>::step( [&] (unsigned int d) {
            _mm256_storeu_ps(buffer, this->componentRegisters[d]);
            v0[d] = buffer[0];
            v1[d] = buffer[1];
            v2[d] = buffer[2];
            v3[d] = buffer[3];
            v4[d] = buffer[4];
            v5[d] = buffer[5];
            v6[d] = buffer[6];
            v7[d] = buffer[7];
        });
    }

    void store(float* values, unsigned int d) const
    {
        _mm256_storeu_ps(values, componentRegisters[d]);
    }

    template<class Getter>
    Inline void storeVecs(const Getter& func)
    {
        getVecs(func(0), func(1), func(2), func(3), func(4), func(5), func(6), func(7));
    }

    template<class Getter>
    Inline void storeScalars(const Getter& func)
    {
        float values[8];
        store(values, 0);
        func(0) = values[0];
        func(1) = values[1];
        func(2) = values[2];
        func(3) = values[3];
        func(4) = values[4];
        func(5) = values[5];
        func(6) = values[6];
        func(7) = values[7];
    }

    vec<4, float> operator()(unsigned int d) const
    {
        vec<4, float> out;
        store(&out[0], d);
        return out;
    }

    void setZero()
    {
        StaticFor<0, D>::step( [this] (unsigned int d) { this->componentRegisters[d] = _mm256_setzero_ps(); });
    }

    // Operators
    vec_simd<D>& operator+=(const vec_simd<D>& vector)
    {
        StaticFor<0, D>::step( [this, &vector] (unsigned int d) { this->componentRegisters[d] = _mm256_add_ps(this->componentRegisters[d], vector.componentRegisters[d]); });
        return *this;
    }
    vec_simd<D>& operator-=(const vec_simd<D>& vector)
    {
        StaticFor<0, D>::step( [this, &vector] (unsigned int d) { this->componentRegisters[d] = _mm256_sub_ps(this->componentRegisters[d], vector.componentRegisters[d]); });
        return *this;
    }
    vec_simd<D>& operator*=(const vec_simd<1>& scalar)
    {
        StaticFor<0, D>::step( [this, &scalar] (unsigned int d) { this->componentRegisters[d] = _mm256_mul_ps(this->componentRegisters[d], scalar.componentRegisters[0]); });
        return *this;
    }
    vec_simd<D>& operator/=(const vec_simd<1>& scalar)
    {
        StaticFor<0, D>::step( [this, &scalar] (unsigned int d) { this->componentRegisters[d] = _mm256_div_ps(this->componentRegisters[d], scalar.componentRegisters[0]); });
        return *this;
    }

    Inline vec_simd<D> elementWiseMultiplication(const vec_simd<D>& other) const
    {
        vec_simd<D> out;
        StaticFor<0, D>::step( [this, &other, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_mul_ps(this->componentRegisters[d], other.componentRegisters[d]); });
        return out;
    }

    vec_simd<D> elementWiseDivision(const vec_simd<D>& other) const
    {
        vec_simd<D> out;
        StaticFor<0, D>::step( [this, &other, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_div_ps(this->componentRegisters[d], other.componentRegisters[d]); });
        return out;
    }

    Inline vec_simd<1> dotProduct(const vec_simd<D>& other) const
    {
        vec_simd<1> out;
        out.componentRegisters[0] = _mm256_mul_ps(componentRegisters[0], other.componentRegisters[0]);
        StaticFor<1, D>::step( [this, &other, &out] (unsigned int d) {
            out.componentRegisters[0] = _mm256_add_ps(out.componentRegisters[0], _mm256_mul_ps(this->componentRegisters[d], other.componentRegisters[d]));
        });
        return out;
    }

    void sqrt()
    {
        StaticFor<0, D>::step( [this] (unsigned int d) { this->componentRegisters[d] = _mm256_sqrt_ps(this->componentRegisters[d]); });
    }

    vec_simd<1> length()
    {
        vec_simd<1> out = dotProduct(*this);
        out.sqrt();
        return out;
    }

    //! For each register, the 4 values are summed up.
    Inline vec<D, float> horizontalAdd() const
    {
        float tmp[8];
        vec<D, float> out;
        StaticFor<0, D>::step([&](unsigned int d){
            _mm256_storeu_ps(tmp, componentRegisters[d]);
            for (unsigned int i = 0; i < 8; i++)
                out[d] += tmp[i];
            /*__m256 x = _mm256_hadd_ps(this->componentRegisters[d], this->componentRegisters[d]);
            x = _mm256_hadd_ps(x, x);
            x = _mm256_hadd_ps(x, x);
            _mm256_maskstore_ps(&out[d], x);*/
        });
        return out;
    }

    // Vec3 specific functionality. Includes cross product, but also optimized load and store methods.
    template <unsigned int E = D>
    static typename std::enable_if<E == 3, vec_simd<3> >::type crossProduct(const vec_simd<3>& v1, const vec_simd<3>& v2)
    {
        vec_simd<3> out;
        out.componentRegisters[0] = _mm256_sub_ps(_mm256_mul_ps(v1.componentRegisters[1], v2.componentRegisters[2]), _mm256_mul_ps(v1.componentRegisters[2], v2.componentRegisters[1]));
        out.componentRegisters[1] = _mm256_sub_ps(_mm256_mul_ps(v1.componentRegisters[2], v2.componentRegisters[0]), _mm256_mul_ps(v1.componentRegisters[0], v2.componentRegisters[2]));
        out.componentRegisters[2] = _mm256_sub_ps(_mm256_mul_ps(v1.componentRegisters[0], v2.componentRegisters[1]), _mm256_mul_ps(v1.componentRegisters[1], v2.componentRegisters[0]));
        return out;
    }

    /*template <unsigned int E = D>
    Inline void setVecs3(const typename std::enable_if<E == 3, vec<3, float> >::type& v0, const typename std::enable_if<E == 3, vec<3, float> >::type& v1, const typename std::enable_if<E == 3, vec<3, float> >::type& v2, const typename std::enable_if<E == 3, vec<3, float> >::type& v3)
    {
        __m256 xy0, xy1;
        xy0 = xy1 = _mm256_setzero_ps();
        xy0 = _mm256_loadl_pi(xy0, (const __m64*)&v0[0]);
        xy1 = _mm256_loadl_pi(xy1, (const __m64*)&v2[0]);
        xy0 = _mm256_loadh_pi(xy0, (const __m64*)&v1[0]);
        xy1 = _mm256_loadh_pi(xy1, (const __m64*)&v3[0]);
        componentRegisters[0] = _mm256_shuffle_ps(xy0, xy1, _mm256_SHUFFLE(2, 0, 2, 0));
        componentRegisters[1] = _mm256_shuffle_ps(xy0, xy1, _mm256_SHUFFLE(3, 1, 3, 1));
        componentRegisters[2] = _mm256_setr_ps(v0[2], v1[2], v2[2], v3[2]);
    }*/
    template<class Getter>
    Inline void setVecs3(const Getter& func)
    {
        setVecs(func(0), func(1), func(2), func(3), func(4), func(5), func(6), func(7));
        // setVecs3(func(0), func(1), func(2), func(3)); TODO
    }

    template<class Getter>
    Inline void setVecs3(const Getter& func, unsigned int numElements, const vec<D, float>& fillValue)
    {
        setVecs(func, numElements, fillValue);
        /*begin_disable_warnings      // Yes, we may read uninitialized memory here. No, we do not plan do use it.
        setVecs3(func(0),
                (numElements > 1) ? func(1) : fillValue,
                (numElements > 2) ? func(2) : fillValue,
                (numElements > 3) ? func(3) : fillValue);
        end_disable_warnings*/
    }

    //! Optimized version of horizontal add for 3D-vectors. Performs four _mm256_hadd_ps instead of six.
    template <unsigned int E = D>
    Inline typename std::enable_if<E == 3, vec<3, float> >::type horizontalAdd3() const
    {
        return horizontalAdd();
        /*vec<3, float> out;
        __m256 xy = _mm256_hadd_ps(this->componentRegisters[0], this->componentRegisters[1]);
        __m256 z =  _mm256_hadd_ps(this->componentRegisters[2], this->componentRegisters[2]);
        xy = _mm256_hadd_ps(xy, xy);
        z = _mm256_hadd_ps(z, z);
        _mm256_storel_pi((__m64*)&out[0], xy);
        _mm256_store_ss(&out[2], z);
        return out;*/
    }
};

template<unsigned int D>
vec_simd<D> operator+(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_add_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}

template<unsigned int D>
vec_simd<D> operator-(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_sub_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}

template<unsigned int D>
vec_simd<D> operator*(const vec_simd<1>& scalar, const vec_simd<D>& v)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&scalar, &v, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_mul_ps(scalar.componentRegisters[0], v.componentRegisters[d]); });
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
    StaticFor<0, D>::step( [&scalar, &v, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_div_ps(v.componentRegisters[d], scalar.componentRegisters[0]); });
    return out;
}

template<unsigned int D>
vec_simd<D> operator-(const vec_simd<D>& v)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_xor_ps(v.componentRegisters[d], _mm256_set1_ps(-0.f)); });
    return out;
}

// Bitwise operations (sometimes useful to eliminate branching).
template<unsigned int D>
vec_simd<D> operator&(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_and_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator|(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_or_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator^(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_xor_ps(v1.componentRegisters[d], v2.componentRegisters[d]); });
    return out;
}

// Comparison operators
template<unsigned int D>
vec_simd<D> operator>(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_cmp_ps(v1.componentRegisters[d], v2.componentRegisters[d], _CMP_GT_OQ); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator>=(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_cmp_ps(v1.componentRegisters[d], v2.componentRegisters[d], _CMP_GE_OQ); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator<(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_cmp_ps(v1.componentRegisters[d], v2.componentRegisters[d], _CMP_LT_OQ); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator<=(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_cmp_ps(v1.componentRegisters[d], v2.componentRegisters[d], _CMP_LE_OQ); });
    return out;
}
template<unsigned int D>
vec_simd<D> operator==(const vec_simd<D>& v1, const vec_simd<D>& v2)
{
    vec_simd<D> out;
    StaticFor<0, D>::step( [&v1, &v2, &out] (unsigned int d) { out.componentRegisters[d] = _mm256_cmp_ps(v1.componentRegisters[d], v2.componentRegisters[d], _CMP_EQ_OQ); });
    return out;
}

#endif // vec_avx_H
