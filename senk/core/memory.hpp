#ifndef SENK_CORE_MEMORY_HPP
#define SENK_CORE_MEMORY_HPP

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <unordered_map>

#include "senk/core/tools.hpp"

namespace senk {

template <class L, SENK_ENABULER(is_locator_v<L>)>
struct memory {
  inline static std::unordered_map<void *, size_t> map =
      std::unordered_map<void *, size_t>();
  inline static double required = 0;
  inline static double peak = 0;
  template <typename T>
  static T *alloc(size_t size);
  static void free(void *ptr);
};

template <class Lsrc, class Ldst, SENK_ENABULER(is_locator_v<Lsrc>),
    SENK_ENABULER(is_locator_v<Ldst>)>
struct communicator {
  template <typename T>
  static void to(T *dst, const T *src, size_t size);
  template <typename T>
  static void from(T *dst, const T *src, size_t size);
};

template <>
template <typename T>
inline T *memory<host>::alloc(size_t size) {
  auto ptr = static_cast<T *>(std::calloc(size, sizeof(T)));
  required += size * sizeof(T);
  peak = (peak < required) ? required : peak;
  map[ptr] = size * sizeof(T);
  return ptr;
}

template <>
inline void memory<host>::free(void *ptr) {
  if (ptr) {
    auto size = memory<host>::map[ptr];
    map.erase(ptr);
    required -= size;
    std::free(ptr);
  }
}

template <>
template <typename T>
inline void communicator<host, host>::to(T *dst, const T *src, size_t size) {
  std::memcpy(dst, src, sizeof(T) * size);
}

template <>
template <typename T>
inline void communicator<host, host>::from(T *dst, const T *src, size_t size) {
  std::memcpy(dst, src, sizeof(T) * size);
}

template <class Ldst, class Lsrc, typename T, SENK_ENABULER(is_locator_v<Ldst>),
    SENK_ENABULER(is_locator_v<Lsrc>)>
inline void memcpy(T *dst, const T *src, size_t size) {
  if constexpr (std::is_same_v<Ldst, host> && std::is_same_v<Lsrc, host>) {
    std::memcpy(dst, src, sizeof(T) * size);
  }
}

#if defined(SENK_WITH_CUDA)

template <>
template <typename T>
inline T *memory<device>::alloc(size_t size) {
  T *ptr;
  cudaMalloc((void **)&ptr, sizeof(T) * size);
  cudaMemset(ptr, 0, sizeof(T) * size);
  required += size * sizeof(T);
  peak = (peak < required) ? required : peak;
  map[ptr] = size * sizeof(T);
  return ptr;
}

template <>
inline void memory<device>::free(void *ptr) {
  auto size = memory<host>::map[ptr];
  map.erase(ptr);
  required -= size;
  cudaFree(ptr);
}

template <>
template <typename T>
inline void communicator<device, device>::to(
    T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToDevice);
}

template <>
template <typename T>
inline void communicator<device, device>::from(
    T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToDevice);
}

template <>
template <typename T>
inline void communicator<host, device>::to(T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice);
}

template <>
template <typename T>
inline void communicator<host, device>::from(
    T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToHost);
}

template <>
template <typename T>
inline void communicator<device, host>::to(T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToHost);
}

template <>
template <typename T>
inline void communicator<device, host>::from(
    T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice);
}

#endif

#if defined(SENK_WITH_HIP)

template <>
template <typename T>
inline T *memory<device>::alloc(size_t size) {
  T *ptr;
  SENK_DEVICE_CHECK(hipMalloc((void **)&ptr, sizeof(T) * size));
  SENK_DEVICE_CHECK(hipMemset(ptr, 0, sizeof(T) * size));
  required += size * sizeof(T);
  peak = (peak < required) ? required : peak;
  map[ptr] = size * sizeof(T);
  return ptr;
}

template <>
inline void memory<device>::free(void *ptr) {
  auto size = memory<host>::map[ptr];
  map.erase(ptr);
  required -= size;
  SENK_DEVICE_CHECK(hipFree(ptr));
}

template <>
template <typename T>
inline void communicator<device, device>::to(
    T *dst, const T *src, size_t size) {
  SENK_DEVICE_CHECK(
      hipMemcpy(dst, src, sizeof(T) * size, hipMemcpyDeviceToDevice));
}

template <>
template <typename T>
inline void communicator<device, device>::from(
    T *dst, const T *src, size_t size) {
  SENK_DEVICE_CHECK(
      hipMemcpy(dst, src, sizeof(T) * size, hipMemcpyDeviceToDevice));
}

template <>
template <typename T>
inline void communicator<host, device>::to(T *dst, const T *src, size_t size) {
  SENK_DEVICE_CHECK(
      hipMemcpy(dst, src, sizeof(T) * size, hipMemcpyHostToDevice));
}

template <>
template <typename T>
inline void communicator<host, device>::from(
    T *dst, const T *src, size_t size) {
  SENK_DEVICE_CHECK(
      hipMemcpy(dst, src, sizeof(T) * size, hipMemcpyDeviceToHost));
}

template <>
template <typename T>
inline void communicator<device, host>::to(T *dst, const T *src, size_t size) {
  SENK_DEVICE_CHECK(
      hipMemcpy(dst, src, sizeof(T) * size, hipMemcpyDeviceToHost));
}

template <>
template <typename T>
inline void communicator<device, host>::from(
    T *dst, const T *src, size_t size) {
  SENK_DEVICE_CHECK(
      hipMemcpy(dst, src, sizeof(T) * size, hipMemcpyHostToDevice));
}

#endif

} // namespace senk

#endif // SENK_CORE_MEMORY_HPP
