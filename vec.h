#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <type_traits>

using std::size_t;

template <size_t D, typename T>
class vec
{
public:

    /** Constructors **/

    inline vec();
    template <size_t E = D> inline vec(T x, T y, typename std::enable_if<E == 2, T>::type* = 0) { set(x, y);  }
    template <size_t E = D> inline vec(T x, T y, T z, typename std::enable_if<E == 3, T>::type* = 0) {  set(x, y, z);  }
    template <size_t E = D> inline vec(T x, T y, T z, T w, typename std::enable_if<E == 4, T>::type* = 0) {  set(x, y, z, w);  }

    //! Builds a vector of dimension D by taking two inputs: a vector of dimension D-1 and the missing component.
    template <size_t E> inline vec(vec<E, T> v, T component, typename std::enable_if<E == D - 1, T>::type* = 0);


    /** Getters **/

    template <size_t E = D> typename std::enable_if<E >= 1, T>::type x() const { return element(0); }
    template <size_t E = D> typename std::enable_if<E >= 1, T>::type& x() { return element(0); }
    template <size_t E = D> typename std::enable_if<E >= 2, T>::type y() const { return element(1); }
    template <size_t E = D> typename std::enable_if<E >= 2, T>::type& y() { return element(1); }
    template <size_t E = D> typename std::enable_if<E >= 3, T>::type z() const { return element(2); }
    template <size_t E = D> typename std::enable_if<E >= 3, T>::type& z() { return element(2); }
    template <size_t E = D> typename std::enable_if<E >= 4, T>::type w() const { return element(3); }
    template <size_t E = D> typename std::enable_if<E >= 4, T>::type& w() { return element(3); }

    const T& operator()(size_t d) const { return element(d); }
    T& operator()(size_t d) { return element(d); }

    const T& operator[](size_t d) const { return element(d); } // TODO: just temporary
    T& operator[](size_t d) { return element(d); } // TODO: just temporary

    const T& element(size_t d) const { return m_Data[d]; }
    T& element(size_t d) { return m_Data[d]; }


    /** Setters **/

    template <size_t E = D> typename std::enable_if<E >= 1, void>::type setX(T x) { setElement(0, x); }
    template <size_t E = D> typename std::enable_if<E >= 2, void>::type setY(T y) { setElement(1, y); }
    template <size_t E = D> typename std::enable_if<E >= 3, void>::type setZ(T z) { setElement(2, z); }
    template <size_t E = D> typename std::enable_if<E >= 4, void>::type setW(T w) { setElement(3, w); }

    template <size_t E = D> typename std::enable_if<E == 2, void>::type set(T x, T y) { setX(x); setY(y); } // TODO
    template <size_t E = D> typename std::enable_if<E == 3, void>::type set(T x, T y, T z) { setX(x); setY(y); setZ(z); } // TODO
    template <size_t E = D> typename std::enable_if<E == 4, void>::type set(T x, T y, T z, T w) { setX(x); setY(y); setZ(z); setW(w); } // TODO

    void setElement(size_t d, T value) { m_Data[d] = value; }


    /** Static predefined vectors **/

    static vec<D, T> zeroVector() { return vec<D, T>(); } // TODO: just temporary


    /** Axis **/

    template <size_t E = D> typename std::enable_if<E == 3, vec<D, T>>::type static xAxis() { return vec<D, T>(1, 0, 0); } // TODO: just temporary
    template <size_t E = D> typename std::enable_if<E == 3, vec<D, T>>::type static yAxis() { return vec<D, T>(0, 1, 0); } // TODO: just temporary
    template <size_t E = D> typename std::enable_if<E == 3, vec<D, T>>::type static zAxis() { return vec<D, T>(0, 0, 1); } // TODO: just temporary
    template <size_t E = D> typename std::enable_if<E == 3, vec<D, T>>::type static origin() { return zeroVector(); } // TODO: just temporary


    /** Length **/

    T length() const { return std::sqrt(lengthSquared()); }
    T lengthSquared() const { return dotProduct(*this, *this); }
    T squaredDistance(const vec<D, T>& other) const { return ((*this) - other).lengthSquared(); } // TODO: move it to an auxiliary class


    /** Normalization **/

    vec<D, T> normalized() const;
    void normalize() { *this = normalized(); }

    //! Normalizes the vector, but does not throw an exception if the length of the vector is zero. Zero vectors are not changed by this operation.
    void pseudoNormalize();

    //! Returns the normalized vector, but does not throw an exception if the length of the vector is zero. In this case, a zero vector is returned.
    vec<D, T> pseudoNormalized() const;


    /** Operators **/

    inline vec<D, T>& operator+=(const vec<D, T>& vector);
    inline vec<D, T>& operator-=(const vec<D, T>& vector);
    inline vec<D, T>& operator*=(T factor);
    inline vec<D, T>& operator/=(T divisor);


    /** Dot product **/

    T operator*(const vec<D, T>& vec) const { return dotProduct(*this, vec); }
    static inline T dotProduct(const vec<D, T>& v1, const vec<D, T>& v2);


    /** Cross product **/

    template <size_t E = D>
    static typename std::enable_if<E == 3, vec<D, T>>::type crossProduct(const vec<D, T>& v1, const vec<D, T>& v2);


    /** Orthogonal direction **/

    template <size_t E = D>
    inline typename std::enable_if<E == 3, vec<D, T>>::type orthogonalDirection() const; // TODO: Move it to an auxiliary class? Make it static?


    /** Elementwise operations **/

    inline vec<D, T> elementWiseMultiplication(const vec<D, T>& scale) const;
    inline vec<D, T> elementWiseDivision(const vec<D, T>& scale) const;

    //! The element wise minimum of two vectors.
    static inline vec<D, T> min(const vec<D, T>& v1, const vec<D, T>& v2);
    //! The element wise maximum of two vectors.
    static inline vec<D, T> max(const vec<D, T>& v1, const vec<D, T>& v2);
    //! The element wise absolute value.
    static inline vec<D, T> abs(const vec<D, T>& vector);


    /** Other stuff **/

    size_t indexOfMinComponent() const;
    size_t indexOfMinAbsComponent() const;
    size_t indexOfMaxComponent() const;
    size_t indexOfMaxAbsComponent() const;

    //! Returns the size of this vector-type in bytes.
    static size_t byteSize() { return D * sizeof(T); }

private:
    std::array<T, D> m_Data;
};


/** Overloaded non-member operators **/

template <size_t D, typename T>
inline bool operator==(const vec<D, T>& v1, const vec<D, T>& v2);
template <size_t D, typename T>
inline bool operator!=(const vec<D, T>& v1, const vec<D, T>& v2);

template <size_t D, typename T>
inline vec<D, T> operator+(const vec<D, T>& v1, const vec<D, T>& v2);
template <size_t D, typename T>
inline vec<D, T> operator-(const vec<D, T>& v1, const vec<D, T>& v2);
template <size_t D, typename T>
inline vec<D, T> operator-(const vec<D, T>& v) { return vec<D, T>() - v; } // TODO: maybe declare this operator, and derive two-argument operator from it
template <size_t D, typename T, typename F>
inline vec<D, T> operator*(const F factor, const vec<D, T>& v);
template <size_t D, typename T, typename F>
inline vec<D, T> operator*(const vec<D, T>& v, const F factor) { return factor * v; }
template <size_t D, typename T, typename F>
inline vec<D, T> operator/(vec<D, T> v, const F divisor) { return v /=  divisor; } // TODO: this could negatively influence the performance

/** Approximative equality **/

template <size_t D, typename T>
inline bool qFuzzyCompare(const vec<D, T>& v1, const vec<D, T>& v2);

#include "vec.inl"
