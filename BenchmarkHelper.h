// Copyright 2015 Philipp Bausch <bauschp@informatik.uni-freiburg.de>
#ifndef BENCHMARKHELPER_H_
#define BENCHMARKHELPER_H_
#include <cassert>
// Needed for clock() to count CPU time.
#include <ctime>
// Needed for high_resolution_clock that should calculate the real time.
#include <chrono>
// Needed for fprintf.
#include <cstdio>
// Needed for std::bind
#include <functional>
// Needed for std::forward
#include <utility>


namespace placeholderhelper {

template<int T>
struct placeholder {};

template <std::size_t... Is>
struct indices {};

template <std::size_t N, std::size_t... Is>
struct build_indices
  : build_indices<N-1, N-1, Is...> {};

template <std::size_t... Is>
struct build_indices<0, Is...>
  : indices<Is...> {};
template<std::size_t... Is, typename F, typename C>
auto easy_bind(indices<Is...>, F const& f, C* c)
    -> decltype(std::bind(f, c, placeholder<Is + 1>{}...)) {
  return std::bind(f, c, placeholder<Is + 1>{}...);
}
}  // namespace placeholderhelper

namespace std {
template<int I>
struct is_placeholder<placeholderhelper::placeholder<I> >
    : std::integral_constant<int, I> {};
}  // namespace std


namespace benchmarkhelper {
/// Benchmarks the given function with given arguments n times.
/// Then it returns the result of the last argument.
/// NOTE: function will be called at least once. (Therefore 0 = 1, :-))
template <typename Func, typename... Args>
auto benchmark(const size_t& n, Func&& f, Args&&... args)
    -> decltype(f(std::forward<Args>(args)...));
/// Benchmark function for member functions.
template <typename R, typename C, typename... Types, typename... Args>
auto benchmark(const size_t& n, R(C::*f)(Types...), C* c, Args&&... args)
    -> R;
/// Benchmark function for const member functions.
template <typename R, typename C, typename... Types, typename... Args>
auto benchmark(const size_t& n, R(C::*f)(Types...) const, C* c, Args&&... args)
    -> R;
// Fallback with reasonable message if class is provided by value.
template <typename R, typename C, typename... Types, typename... Args>
auto benchmark(const size_t& n, R(C::*f)(Types...) const, C c, Args&&... args)
    -> R;
// Fallback with reasonable message if class is provided by value.
template <typename R, typename C, typename... Types, typename... Args>
auto benchmark(const size_t& n, R(C::*f)(Types...), C c, Args&&... args)
    -> R;
FILE* benchmarkStream = stdout;

template<typename... Args>
struct Callback {
 public:
  static std::function<void(Args...)> mCallback;
};
}  // namespace benchmarkhelper


// TEMPLATE IMPLEMENTATIONS:
// _____________________________________________________________________________

template<typename... Args>
std::function<void(Args...)>
  benchmarkhelper::Callback<Args...>::mCallback = nullptr;

template <typename Func, typename... Args>
auto benchmarkhelper::benchmark(
    const size_t& n,
    Func&& f,
    Args&&... args) -> decltype(f(std::forward<Args>(args)...)) {
  // A trick from stack overflow:
  struct BenchmarkTimer {
   public:
    explicit BenchmarkTimer(const std::size_t& n) :
      mClockTime(clock()),
      mNumRuns(n),
      mWallTime(std::chrono::high_resolution_clock::now()) {
    }
    ~BenchmarkTimer() {
      // Stop time.
      std::clock_t end = clock();
      auto endTime = std::chrono::high_resolution_clock::now();
      // Print the result.
      assert(benchmarkStream);
      fprintf(benchmarkStream, "RUNS:\t%lu\nCPU:\t%.2fms\nWALL:\t%.2fms\n",
        mNumRuns,
        1e3 * static_cast<double>(end - mClockTime) / CLOCKS_PER_SEC,
        std::chrono::duration<double, std::milli>(endTime - mWallTime).count());
    }
   private:
    const std::clock_t mClockTime;
    const std::size_t mNumRuns;
    const std::chrono::high_resolution_clock::time_point mWallTime;
  } timer(n);
  assert(n != 0);
  for (size_t i = 0; i < n - 1; ++i) {
    if (Callback<Args...>::mCallback)
      Callback<Args...>::mCallback(std::forward<Args>(args)...);
    f(std::forward<Args>(args)...);
  }
  return f(std::forward<Args>(args)...);
}

// _____________________________________________________________________________
template <typename R, typename C, typename... Types, typename... Args>
auto benchmarkhelper::benchmark(
    const size_t& n,
    R(C::*f)(Types...) const,
    C* c, Args&&... args) -> R {
  // Call the basic function after binding the parameters.
  auto func = placeholderhelper::easy_bind(
      placeholderhelper::build_indices<sizeof...(Args)>{}, f, c);
  return benchmark(n, func, std::forward<Args>(args)...);
}
// _____________________________________________________________________________
template <typename R, typename C, typename... Types, typename... Args>
auto benchmarkhelper::benchmark(
    const size_t& n,
    R(C::*f)(Types...),
    C* c, Args&&... args) -> R {
  // Call the basic function after binding the parameters.
  auto func = placeholderhelper::easy_bind(
       placeholderhelper::build_indices<sizeof...(Args)>{}, f, c);
  return benchmark(n, func, std::forward<Args>(args)...);
}
// Fallback with reasonable message if class is provided by value.
template <typename R, typename C, typename... Types, typename... Args>
auto benchmarkhelper::benchmark(
    const size_t& n,
    R(C::*f)(Types...) const,
    C c, Args&&... args) -> R {
  static_assert(sizeof(C) != sizeof(C) || n != n,
      "Make sure you provide the Class as pointer.");
  return (c.*f)(std::forward<Args>(args)...);
}
template <typename R, typename C, typename... Types, typename... Args>
auto benchmarkhelper::benchmark(
    const size_t& n,
    R(C::*f)(Types...),
    C c, Args&&... args) -> R {
  static_assert(sizeof(C) != sizeof(C) || n != n,
      "Make sure you provide the Class as pointer.");
  return (c.*f)(std::forward<Args>(args)...);
}

#endif  // BENCHMARKHELPER_H_
