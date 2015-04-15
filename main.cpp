
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <smmintrin.h>
#include <random>
#include <vector>
#include "BenchmarkHelper.h"
#include "vec_simd.h"

using std::vector;

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

typedef vec<3, float> vec3f;
typedef vector<vec3f> vec3fArray;

void fillFloatArrayRandom(float* values, size_t numElements, float min = 0, float max = 10000.0f)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(min, max);

    for (size_t i = 0; i < numElements; i++)
    {
        values[i] = distribution(generator);
    }
}

float computeAverage(float* valuesIn, size_t numElements)
{
    double sum = 0;
    for (size_t i = 0; i < numElements; i++)
        sum += (double)valuesIn[i];
    return sum / numElements;
}

struct ComputeAverageAccu
{
	vec_simd<1> sum = vec_simd<1>(0.0f);
	vec_simd<1> offset;
	float* values;
	ComputeAverageAccu(float* values) : values(values) {}
	
	void loadData(unsigned int i) { offset.load(values + i); }
	void loadRemainingData(unsigned int i, unsigned int numElements)
	{
		offset.setScalars(
			[&](unsigned int x) -> float& { return values[i + x]; },
			numElements, 0.0f);
	}
	
	void accumulate(int) { sum += offset; }
};

float computeAverageSimd(float* valuesIn, size_t numElements)
{
	ComputeAverageAccu accu(valuesIn);
	SimdUtils::accumulate(accu, numElements);
	accu.sum /= vec_simd<1>(numElements);
    return accu.sum.horizontalAdd()[0];
}

void computeSquareRootsSimd(vector<float>& in, vector<float>& out, size_t n)
{
    for (size_t i = 0; i < n; i += 4)
    {
        __m128 tmp = _mm_loadu_ps(&in[i]);
        tmp = _mm_sqrt_ps(tmp);
        _mm_storeu_ps(&out[i], tmp);
    }
}

void computeSquareRoots(vector<float>& in, vector<float>& out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = std::sqrt(in[i]);
}

void computeSqrtsConditional(vector<float>& in, vector<float>& out, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (in[i] >= 0)
            out[i] = std::sqrt(in[i]);
        else
            out[i] = 0.0f;
    }
}

void computeSqrtsConditionalSimd(vector<float>& in, vector<float>& out, size_t n)
{
    __m128 zero = _mm_setzero_ps();
    for (size_t i = 0; i < n; i += 4)
    {
        __m128 tmp = _mm_loadu_ps(&in[i]);
        __m128 mask = _mm_cmpge_ps(tmp, zero);
        tmp = _mm_or_ps(
            _mm_and_ps(mask, _mm_sqrt_ps(tmp)),
            _mm_andnot_ps(mask, zero));
        _mm_storeu_ps(&out[i], tmp);
    }
}

void computeSqrtsCondSimdNice(const vector<float>& in, vector<float>& out, size_t n)
{
    vec_simd<1> v;
    vec_simd<1> zero(0.0f);
    for (size_t i = 0; i < n; i += 4)
    {
        v.load(&in[i]);
        v = SimdUtils::choose(v >= zero,
            vec_simd<1>::sqrt(v), zero);
        v.store(&out[i]);
    }
}

void computeSquareRootsNoAutoVectorization(vector<float>& in, vector<float>& out, size_t n)
{
// #pragma loop(no_vector)
    for (size_t i = 0; i < n; i++)
        out[i] = std::sqrt(in[i]);
}

void normalizeVecs(vec3fArray& in, vec3fArray& out, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        float length = in[i].length();
        out[i] = in[i] / length;    // in[i].length();
    }
}

void normalizeVecsSimd(const vec3fArray& in, vec3fArray& out, size_t n)
{
    vec_simd<3> v;
    for (size_t i = 0; i < n; i += 4)
    {
        v.setVecs([&](size_t o) -> const vec3f& { return in[i + o]; });
        v /= v.length();
        v.storeVecs([&](size_t o) -> vec3f& { return out[i + o]; });
    }
}

void normalizeVecsSimdCond(const vec3fArray& in, vec3fArray& out, size_t n)
{
    vec_simd<3> v;
    for (size_t i = 0; i < n; i += 4)
    {
        v.setVecs([&](size_t o) -> const vec3f& { return in[i + o]; });
        vec_simd<1> length = v.length();
        v = SimdUtils::choose(length > vec_simd<1>(0.0f), v / length, v);
        v.storeVecs([&](size_t o) -> vec3f& { return out[i + o]; });
    }
}

void normalizeVecsSimdMap(vec3fArray& in, vec3fArray& out, size_t n)
{
    SimdUtils::Normalize3DMapper mapper(in.data(), out.data());
    SimdUtils::mapVectors<SimdUtils::Normalize3DMapper, 3>(mapper, n);
}

void benchmarkComputeSqrts()
{
    const size_t numElements = 8 * 10000000;
    vector<float> valuesIn(numElements);
    vector<float> valuesOut(numElements);
    memset(valuesOut.data(), 0, numElements * sizeof(float));
    fillFloatArrayRandom(valuesIn.data(), numElements);

    std::cout << "Benchmarking computeSquareRoots..." << std::endl;
    benchmarkhelper::benchmark(5, computeSquareRoots, valuesIn, valuesOut, numElements);
    std::cout << computeAverage(valuesOut.data(), numElements) << std::endl;

    memset(valuesOut.data(), 0, numElements * sizeof(float));

    std::cout << "Benchmarking computeSquareRootsSimd..." << std::endl;
    benchmarkhelper::benchmark(5, computeSquareRootsSimd, valuesIn, valuesOut, numElements);
    std::cout << computeAverage(valuesOut.data(), numElements) << std::endl;

    memset(valuesOut.data(), 0, numElements * sizeof(float));

    std::cout << "Benchmarking computeSquareRootsNoAutoVectorization..." << std::endl;
    benchmarkhelper::benchmark(5, computeSquareRootsNoAutoVectorization, valuesIn, valuesOut, numElements);
    std::cout << computeAverage(valuesOut.data(), numElements) << std::endl;

    memset(valuesOut.data(), 0, numElements * sizeof(float));

    fillFloatArrayRandom(valuesIn.data(), numElements, -100.0f);

    std::cout << "Benchmarking computeSqrtsConditional..." << std::endl;
    benchmarkhelper::benchmark(5, computeSqrtsConditional, valuesIn, valuesOut, numElements);
    std::cout << computeAverage(valuesOut.data(), numElements) << std::endl;

    memset(valuesOut.data(), 0, numElements * sizeof(float));

    std::cout << "Benchmarking computeSqrtsConditionalSimd..." << std::endl;
    benchmarkhelper::benchmark(5, computeSqrtsConditionalSimd, valuesIn, valuesOut, numElements);
    std::cout << computeAverage(valuesOut.data(), numElements) << std::endl;

    std::cout << "Benchmarking computeSqrtsCondSimdNice..." << std::endl;
    benchmarkhelper::benchmark(5, computeSqrtsCondSimdNice, valuesIn, valuesOut, numElements);
    std::cout << computeAverage(valuesOut.data(), numElements) << std::endl;
}

void benchmarkVectorNormalizationAoS()
{
    const size_t numElements = 4 * 10000000;
    vec3fArray vecsIn(numElements);
    vec3fArray vecsOut(numElements);
    memset(vecsOut.data(), 0, numElements * sizeof(vec3f));
    fillFloatArrayRandom(&vecsIn[0][0], numElements * 3);

    std::cout << "Benchmarking normalizeVecs..." << std::endl;
    benchmarkhelper::benchmark(5, normalizeVecs, vecsIn, vecsOut, numElements);

    memset(vecsOut.data(), 0, numElements * sizeof(vec3f));

    std::cout << "Benchmarking normalizeVecsSimd..." << std::endl;
    benchmarkhelper::benchmark(5, normalizeVecsSimd, vecsIn, vecsOut, numElements);

    memset(vecsOut.data(), 0, numElements * sizeof(vec3f));

    std::cout << "Benchmarking normalizeVecsSimd2..." << std::endl;
    benchmarkhelper::benchmark(5, normalizeVecsSimdMap, vecsIn, vecsOut, numElements);

    memset(vecsOut.data(), 0, numElements * sizeof(vec3f));

    std::cout << "Benchmarking normalizeVecsSimdCond..." << std::endl;
    benchmarkhelper::benchmark(5, normalizeVecsSimdCond, vecsIn, vecsOut, numElements);
}

int main()
{
    benchmarkComputeSqrts();

    while (true) {}

    return 0;
}
