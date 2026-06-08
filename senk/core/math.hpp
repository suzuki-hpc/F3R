#ifndef SENK_CORE_MATH_HPP
#define SENK_CORE_MATH_HPP

#include "senk/core/tools.hpp"

#if defined(SENK_WITH_CUDA)
#include <cuda/std/cmath>
#include <cuda/std/complex>
#else
#include <cmath>
#include <complex>
#endif

namespace senk {

template <typename T>
SENK_LOC inline auto abs(T val) {
#if defined(SENK_WITH_CUDA)
  return cuda::std::abs(val);
#else
  return std::abs(val);
#endif
}

template <typename T>
SENK_LOC inline auto sqrt(T val) {
#if defined(SENK_WITH_CUDA)
  return cuda::std::sqrt(val);
#else
  return std::sqrt(val);
#endif
}

#if defined(SENK_HAS_FP16)
template <>
SENK_LOC inline auto abs(half val) {
#if defined(SENK_WITH_CUDA)
  return static_cast<half>(std::abs(static_cast<float>(val)));
#else
  return static_cast<half>(std::abs(static_cast<float>(val)));
#endif
}
template <>
SENK_LOC inline auto sqrt(half val) {
#if defined(SENK_WITH_CUDA)
  return static_cast<half>(cuda::std::sqrt(static_cast<float>(val)));
#else
  return static_cast<half>(std::sqrt(static_cast<float>(val)));
#endif
}
// template <typename T, SENK_ENABULER(std::is_arithmetic_v<T>)>
// SENK_LOC inline T operator*(const T &lh, const senk::half &rh) {
//   return lh * static_cast<T>(rh);
// };
// template <typename T, SENK_ENABULER(std::is_arithmetic_v<T>)>
// SENK_LOC inline T operator*(const senk::half &lh, const T &rh) {
//   return static_cast<T>(lh) * rh;
// };
#endif // defined(SENK_HAS_FP16)

#if defined(SENK_WITH_CUDA) && __has_include(<cuda/std/complex>)
template <typename T>
using complex = cuda::std::complex<T>;
template <typename T>
SENK_LOC inline auto conj(T val) {
  return cuda::std::conj(val);
}
#else
template <typename T>
using complex = std::complex<T>;
template <typename T>
SENK_LOC auto inline conj(T val) {
  return std::conj(val);
}
#endif // defined(SENK_WITH_CUDA) && __has_include(<cuda/std/complex>)

} // namespace senk

#if defined(SENK_HAS_FP16)
template <typename T, SENK_ENABULER(std::is_arithmetic_v<T>)>
SENK_LOC inline T operator*(const T &lh, const senk::half &rh) {
  return lh * static_cast<T>(rh);
};
template <typename T, SENK_ENABULER(std::is_arithmetic_v<T>)>
SENK_LOC inline T operator*(const senk::half &lh, const T &rh) {
  return static_cast<T>(lh) * rh;
};
#endif // defined(SENK_HAS_FP16)

#if defined(__CUDACC__) && __has_include(<cuda/std/complex>)
#pragma omp declare reduction(+ : cuda::std::complex<double> : omp_out +=      \
                                  omp_in) initializer(omp_priv = omp_orig)
#pragma omp declare reduction(+ : cuda::std::complex<float> : omp_out +=       \
                                  omp_in) initializer(omp_priv = omp_orig)
#else
#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in)    \
    initializer(omp_priv = omp_orig)
#pragma omp declare reduction(+ : std::complex<float> : omp_out += omp_in)     \
    initializer(omp_priv = omp_orig)
#endif

#endif // SENK_CORE_MATH_HPP