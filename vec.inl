#include "vec.h"

// Constructors

template <size_t D, typename T>
vec<D, T>::vec()
{
    for (size_t d = 0; d < D; d++)
        m_Data[d] = 0;
}

template <size_t D, typename T>
template <size_t E> inline vec<D, T>::vec(vec<E, T> vector, T component, typename std::enable_if<E == D - 1, T>::type*)
{
    for (size_t e = 0; e < E; e++)
        m_Data[e] = vector.element(e);

    m_Data[E] = component;
}


// Normalization

template <size_t D, typename T>
vec<D, T> vec<D, T>::normalized() const
{
    T length = this->length();
#ifdef DEPLOY_MODE
    if (IsZero(length))
        return zeroVector();
#else
//    THROW_EXCEPTION(IsZero(length), std::domain_error("Normalizing a zero vector is undefined"));
#endif
    return *this / length;
}

template <size_t D, typename T>
void vec<D, T>::pseudoNormalize()
{
    T len = length();
    if (len > 0)
        *this /= len;
}

template <size_t D, typename T>
vec<D, T> vec<D, T>::pseudoNormalized() const
{
    vec<D, T> v = *this;
    v.pseudoNormalize();
    return v;
}


// Operators

template <size_t D, typename T>
vec<D, T>& vec<D, T>::operator+=(const vec<D, T>& vector)
{
    for (size_t d = 0; d < D; d++)
        m_Data[d] += vector.m_Data[d];

    return *this;
}

template <size_t D, typename T>
vec<D, T>& vec<D, T>::operator-=(const vec<D, T>& vector)
{
    for (size_t d = 0; d < D; d++)
        m_Data[d] -= vector.m_Data[d];

    return *this;
}

template <size_t D, typename T>
vec<D, T>& vec<D, T>::operator*=(T factor)
{
    for (size_t d = 0; d < D; d++)
        m_Data[d] *= factor;

    return *this;
}

template <size_t D, typename T>
vec<D, T>& vec<D, T>::operator/=(T divisor)
{
    for (size_t d = 0; d < D; d++)
        m_Data[d] /= divisor;

    return *this;
}


// Dot product

template <size_t D, typename T>
T vec<D, T>::dotProduct(const vec<D, T>& v1, const vec<D, T>& v2)
{
    T sum = 0;
    for (size_t d = 0; d < D; d++)
        sum += v1.m_Data[d] * v2.m_Data[d];
    return sum;
}


// Cross product

template <size_t D, typename T>
template <size_t E>
typename std::enable_if<E == 3, vec<D, T>>::type vec<D, T>::crossProduct(const vec<D, T>& v1, const vec<D, T>& v2)
{
    return vec<D, T>(v1.y() * v2.z() - v1.z() * v2.y(),
                     v1.z() * v2.x() - v1.x() * v2.z(),
                     v1.x() * v2.y() - v1.y() * v2.x());
}


// Orthogonal direction

template <size_t D, typename T>
template <size_t E>
inline typename std::enable_if<E == 3, vec<D, T>>::type vec<D, T>::orthogonalDirection() const
{
    vec<D, T> helperVec(0, 0, 0);
    helperVec[indexOfMinAbsComponent()] = 1;
    vec<D, T> orthogonalDir = vec<D, T>::crossProduct(*this, helperVec);
    orthogonalDir.normalize();
    return orthogonalDir;
}


// Elementwise operations

template <size_t D, typename T>
vec<D, T> vec<D, T>::elementWiseMultiplication(const vec<D, T>& scale) const
{
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.m_Data[d] = this->m_Data[d] * scale.m_Data[d];
    return result;
}

template <size_t D, typename T>
vec<D, T> vec<D, T>::elementWiseDivision(const vec<D, T>& scale) const
{
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.m_Data[d] = this->m_Data[d] / scale.m_Data[d];
    return result;
}

template <size_t D, typename T>
vec<D, T> vec<D, T>::min(const vec<D, T>& v1, const vec<D, T>& v2)
{
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.m_Data[d] = std::min(v1.m_Data[d], v2.m_Data[d]);
    return result;
}

template <size_t D, typename T>
vec<D, T> vec<D, T>::max(const vec<D, T>& v1, const vec<D, T>& v2)
{
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.m_Data[d] = std::max(v1.m_Data[d], v2.m_Data[d]);
    return result;
}

template <size_t D, typename T>
vec<D, T> vec<D, T>::abs(const vec<D, T>& vector)
{
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.m_Data[d] = std::abs(vector.m_Data[d]);
    return result;
}


// Other stuff

template <size_t D, typename T>
size_t vec<D, T>::indexOfMinComponent() const
{
    size_t index = 0;
    T currentMin = element(0);
    for (size_t d = 1; d < D; ++d)
    {
        if (element(d) < currentMin)
        {
            index = d;
            currentMin = element(d);
        }
    }
    return index;
}

template <size_t D, typename T>
size_t vec<D, T>::indexOfMinAbsComponent() const
{
    size_t index = 0;
    T currentAbsMin = std::abs(element(0));
    for (size_t d = 1; d < D; ++d)
    {
        if (std::abs(element(d)) < currentAbsMin)
        {
            index = d;
            currentAbsMin = std::abs(element(d));
        }
    }
    return index;
}

template <size_t D, typename T>
size_t vec<D, T>::indexOfMaxComponent() const
{
    size_t index = 0;
    T currentMax = element(0);
    for (size_t d = 1; d < D; ++d)
    {
        if (element(d) > currentMax)
        {
            index = d;
            currentMax = element(d);
        }
    }
    return index;
}

template <size_t D, typename T>
size_t vec<D, T>::indexOfMaxAbsComponent() const
{
    size_t index = 0;
    T currentAbsMax = std::abs(element(0));
    for (size_t d = 1; d < D; ++d)
    {
        if (std::abs(element(d)) > currentAbsMax)
        {
            index = d;
            currentAbsMax = std::abs(element(d));
        }
    }
    return index;
}


// Overloaded non-member operators

template <size_t D, typename T>
bool operator==(const vec<D, T>& v1, const vec<D, T>& v2)
{
    bool result = true;
    for (size_t d = 0; d < D; d++)
        result &= v1.element(d) == v2.element(d);
    return result;
}

template <size_t D, typename T>
bool operator!=(const vec<D, T>& v1, const vec<D, T>& v2)
{
    return !(v1 == v2);
}


template <size_t D, typename T>
vec<D, T> operator+(const vec<D, T>& v1, const vec<D, T>& v2)
{
    /**
     * It would be more convenient to do
     * 		return v1 += v2;
     * Unfortunately, that led to poor performance.
     */
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.setElement(d, v1.element(d) + v2.element(d));

    return result;
}

template <size_t D, typename T>
vec<D, T> operator-(const vec<D, T>& v1, const vec<D, T>& v2)
{
    /**
         * It would be more convenient to do
         * 		return v1 -= v2;
         * Unfortunately, that led to poor performance.
         */
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.setElement(d, v1.element(d) - v2.element(d));

    return result;
}

template <size_t D, typename T, typename F>
vec<D, T> operator*(const F factor, const vec<D, T>& v)
{
    /**
     * It would be more convenient to do
     * 		return v *= factor;
     * Unfortunately, that led to poor performance.
     */
    vec<D, T> result;
    for (size_t d = 0; d < D; d++)
        result.setElement(d, v.element(d) * factor);

    return result;
}