#ifndef UMINEKO_CORE_INTERFACE_HPP
#define UMINEKO_CORE_INTERFACE_HPP

#include <cstdint>

namespace kmm {

using idx_t = int32_t;

struct host;
struct device;

template <class> struct is_locator {
  constexpr static bool value = false;
};
template <class T> inline constexpr bool is_locator_v = is_locator<T>::value;
template <> struct is_locator<host> {
  constexpr static bool value = true;
};
template <> struct is_locator<device> {
  constexpr static bool value = true;
};

#if defined(__clang__) && defined(__CUDA__) || defined(__CUDACC__)
#define KMM_WITH_CUDA
#endif

#ifndef __is_identifier      // Optional of course.
#define __is_identifier(x) 1 // Compatibility with non-clang compilers.
#endif

// More sensible macro for keyword detection
#define __has_keyword(__x) !(__is_identifier(__x))

// map a half float type, if available, to _OptionalHalfFloatType
// #if __has_keyword(_Float16)
//     typedef _Float16    __half;
// #elif __has_keyword(__fp16)
//     typedef __fp16      __half;
// #else
//     typedef int        __half;
// #endif

} // namespace kmm

#endif