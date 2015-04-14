#ifndef vec_simd_H
#define vec_simd_H

#include "StaticFor.h"
#include "vec.h"
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// Unfortunately, AVX seems to offer no benefit for us at the moment.
// #define USE_AVX

#ifdef USE_AVX
#include "vec_avx.h"
#else
#include "vec_sse.h"
#endif

class SimdUtils
{
public:
    //! Performs mask ? v1 : v2 for each component and returns the result.
    template<unsigned int D>
    Inline static vec_simd<D> choose(const vec_simd<1>& mask, const vec_simd<D>& v1, const vec_simd<D>& v2)
    {
        vec_simd<D> out;
        StaticFor<0, D>::step([&mask, &v1, &v2, &out](unsigned int d)
        {
#ifdef USE_AVX
            out.componentRegisters[d] = _mm256_blendv_ps(v2.componentRegisters[d], v1.componentRegisters[d], mask.componentRegisters[0]);
#else
            out.componentRegisters[d] = _mm_blendv_ps(v2.componentRegisters[d], v1.componentRegisters[d], mask.componentRegisters[0]);
            // out.componentRegisters[d] = _mm_or_ps(_mm_and_ps(mask.componentRegisters[0], v1.componentRegisters[d]),
            //                                      _mm_andnot_ps(mask.componentRegisters[0], v2.componentRegisters[d]));
#endif
        });
        return out;
    }

    template<class Mapper, unsigned int D>
    Inline static void mapVectors(Mapper& mapper, unsigned int numElements)
    {
        vec_simd<D> currVector;
        unsigned int restElements = numElements % vec_simd<1>::REGISTER_SIZE;
        unsigned int i = 0;
        for (; i < numElements - restElements; i += vec_simd<1>::REGISTER_SIZE)
        {
            currVector.setVecs([&](unsigned int offset) -> vec<D, float>& { return mapper.getIn(i + offset); });
            mapper.map(currVector);
            currVector.storeVecs([&](unsigned int offset) -> vec<D, float>& { return mapper.getOut(i + offset); });
        }

        if (restElements > 0)
        {
            // process the remaining 1-3 vectors
            vec<D, float> dummy;
            currVector.setVecs([&](unsigned int offset) -> vec<D, float>& { return offset < numElements ? mapper.getIn(i + offset) : dummy; });
            mapper.map(currVector);
            currVector.storeVecs([&](unsigned int offset) -> vec<D, float>& { return offset < numElements ? mapper.getOut(i + offset) : dummy; });
        }
    }

    //! Example mapper class that normalizes all 3d vectors stored in a std vector.
    struct Normalize3DMapper
    {
        vec<3, float>* vectorsIn;
        vec<3, float>* vectorsOut;
        Normalize3DMapper(vec<3, float>* vectorsIn, vec<3, float>* vectorsOut) : vectorsIn(vectorsIn), vectorsOut(vectorsOut) {}

        void map(vec_simd<3>& v)
        {
            v /= v.length();
        }

        vec<3, float>& getIn(unsigned int i) { return *(vectorsIn + i); }
        vec<3, float>& getOut(unsigned int i) { return *(vectorsOut + i); }
    };

    template<class Accumulator>
    Inline static void accumulate(Accumulator& accu, unsigned int numElements)
    {
        unsigned int restElements = numElements % vec_simd<1>::REGISTER_SIZE;
        unsigned int i = 0;
        if (numElements >= vec_simd<1>::REGISTER_SIZE)
        {
            for (i = 0; i < numElements - restElements; i += vec_simd<1>::REGISTER_SIZE)
            {
                accu.loadData(i);
                accu.accumulate(i);
            }
        }
        if (restElements > 0)
        {
            accu.loadRemainingData(i, restElements);
            accu.accumulate(i);
        }
    }
};

#endif // vec_simd_H
