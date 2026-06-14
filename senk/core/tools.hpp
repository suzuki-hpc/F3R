#ifndef SENK_CORE_TOOLS_HPP
#define SENK_CORE_TOOLS_HPP

#include <iostream>
#include <type_traits>

#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#elif __has_include(<hip/hip_fp16.h>)
#include <hip/hip_fp16.h>
#endif

#define SENK_ENABULER(cond) std::enable_if_t<cond, std::nullptr_t> = nullptr
#define SENK_RET(cond, ret) std::enable_if_t<cond, ret>
#define SENK_IS_SAME_V(cond1, cond2) std::is_same_v<cond1, cond2>
#define SENK_IS_CONV_V(cond1, cond2) std::is_convertible_v<cond1, cond2>

#if defined(__CUDACC__)
#define SENK_WITH_CUDA
#include <cuda_runtime.h>
#define SENK_DEVICE_CHECK(call)                                                \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__,              \
          cudaGetErrorString(err));                                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#elif defined(__HIPCC__)
#define SENK_WITH_HIP
#include <hip/hip_runtime.h>
#define SENK_DEVICE_CHECK(call)                                                \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (hipSuccess != err) {                                                   \
      printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__,              \
          hipGetErrorString(err));                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

#if defined(SENK_WITH_CUDA) | defined(SENK_WITH_HIP)
#define SENK_LOC __host__ __device__
#else
#define SENK_LOC
#endif

#if defined(SENK_WITH_CUDA) && __has_include(<cuda_fp16.h>)
#define SENK_HAS_FP16
#elif defined(SENK_WITH_HIP) && __has_include(<hip/hip_fp16.h>)
#define SENK_HAS_FP16
#elif defined(__FLT16_MAX__)
#define SENK_HAS_FP16
#endif

namespace senk {

#if defined(SENK_WITH_CUDA) && __has_include(<cuda_fp16.h>)
using half = __half;
#elif defined(SENK_WITH_HIP) && __has_include(<hip/hip_fp16.h>)
using half = __half;
#elif defined(__FLT16_MAX__)
using half = _Float16;
#else
using half = float;
#endif

struct host;
struct device;

template <class>
struct is_locator {
  constexpr static bool value = false;
};
template <class T>
inline constexpr bool is_locator_v = is_locator<T>::value;

#define SENK_SET_LOCATOR(TAG)                                                  \
  template <>                                                                  \
  struct is_locator<TAG> {                                                     \
    constexpr static bool value = true;                                        \
  }
SENK_SET_LOCATOR(host);
SENK_SET_LOCATOR(device);
#undef SENK_SET_LOCATOR

template <typename... Args>
constexpr bool all_integral_v = (std::is_integral_v<Args> && ...);

template <typename T>
struct has_val_t {
  using val_t = T;
};

template <class L>
struct has_loc_t {
  using loc_t = L;
};

} // namespace senk

#endif // SENK_CORE_TOOLS_HPP