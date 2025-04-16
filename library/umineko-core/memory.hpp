#ifndef UMINEKO_CORE_MEMORY_HPP
#define UMINEKO_CORE_MEMORY_HPP

#include <cstdlib>
#include <cstring>
#include <type_traits>

#if __has_include("cuda_runtime_api.h")
#include "cuda_runtime_api.h"
#endif

#include "umineko-core/interface.hpp"

namespace kmm {

namespace impl {} // namespace impl

template <class L, std::enable_if_t<is_locator_v<L>, std::nullptr_t> = nullptr>
struct memory {};

template <> struct memory<host> {
  template <typename T> static T *alloc(size_t size) {
    return static_cast<T *>(std::calloc(size, sizeof(T)));
  }
  static void free(void *ptr) { std::free(ptr); }
  template <class L2, typename T>
  static void copy(T *src, T *dst, size_t size) {
    if constexpr (std::is_same_v<L2, host>) {
      memcpy(dst, src, sizeof(T) * size);
    } else {
#if __has_include("cuda_runtime_api.h")
      cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice);
#endif
    }
  }
};

#if __has_include("cuda_runtime_api.h")
template <> struct memory<device> {
  template <typename T> static T *alloc(size_t size) {
    T *ptr;
    cudaMalloc((void **)&ptr, sizeof(T) * size);
    cudaMemset((void **)&ptr, 0, sizeof(T) * size);
    return ptr;
  }
  static void free(void *ptr) { cudaFree(ptr); }
  template <class L2, typename T>
  static void copy(T *src, T *dst, size_t size) {
    if constexpr (std::is_same_v<L2, host>)
      cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToHost);
    else
      cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToDevice);
  }
};
#endif

} // namespace kmm

#endif // UMINEKO_CORE_MEMORY_HPP
