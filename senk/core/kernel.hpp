#ifndef SENK_CORE_KERNEL_HPP
#define SENK_CORE_KERNEL_HPP

#include <array>

#include "senk/core/memory.hpp"
#include "senk/core/tools.hpp"

namespace senk {

template <class L>
struct kernel;

template <>
struct kernel<host> {
  kernel() = delete;
  template <typename F>
  static void single(F func) {
    func();
  }
  template <typename F>
  static void parallel(size_t num, F func) {
#pragma omp parallel for
    for (size_t i = 0; i < num; i++)
      func(i);
  }
  template <size_t N, typename F>
  static void parallel(const std::array<size_t, N> &num, F func) {
    if constexpr (N == 1) {
#pragma omp parallel for
      for (size_t i = 0; i < num[0]; i++)
        func(i);
    } else if constexpr (N == 2) {
#pragma omp parallel for collapse(2)
      for (size_t i = 0; i < num[0]; i++)
        for (size_t j = 0; j < num[1]; j++)
          func(i, j);
    } else if constexpr (N == 3) {
#pragma omp parallel for collapse(3)
      for (size_t i = 0; i < num[0]; i++)
        for (size_t j = 0; j < num[1]; j++)
          for (size_t k = 0; k < num[2]; k++)
            func(i, j, k);
    } else {
      static_assert([]() { return false; }(), "direct must be on host");
    }
  }
  template <typename T, typename F>
  static void reduce_add(
      size_t num, T *res, F func, [[maybe_unused]] T *buffer = nullptr) {
    res[0] = static_cast<T>(0);
#pragma omp parallel for reduction(+ : res[0])
    for (size_t i = 0; i < num; i++)
      res[0] += func(i);
  }
};

template <class Ldst, class Lsrc, typename Tdst, typename Tsrc>
inline void copy(Tdst *dst, const Tsrc *src, size_t size) {
  if constexpr (std::is_same_v<Ldst, Lsrc>) {
    if (size == 1) {
      kernel<Ldst>::single(
          [=] SENK_LOC() { dst[0] = static_cast<Tdst>(src[0]); });
    } else {
      kernel<Ldst>::parallel(
          size, [=] SENK_LOC(size_t i) { dst[i] = static_cast<Tdst>(src[i]); });
    }
  } else if constexpr (std::is_same_v<Tdst, Tsrc>) {
    communicator<Ldst, Lsrc>::from(dst, src, size);
  }
#if defined(SENK_WITH_CUDA)
  else if constexpr (std::is_same_v<Lsrc, host>) {
    const auto s = 4096;
    Tdst buff[s];
    for (size_t c = 0; c < size; c += s) {
      auto t_s = (c + s < size) ? s : size - c;
      for (size_t i = 0; i < t_s; i++)
        buff[i] = static_cast<Tdst>(src[c + i]);
      cudaMemcpy(dst + c, buff, sizeof(Tdst) * t_s, cudaMemcpyHostToDevice);
    }
  } else if constexpr (std::is_same_v<Lsrc, device>) {
    const auto s = 4096;
    Tsrc buff[s];
    for (size_t c = 0; c < size; c += s) {
      auto t_s = (c + s < size) ? s : size - c;
      cudaMemcpy(buff, src + c, sizeof(Tsrc) * t_s, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < t_s; i++)
        dst[c + i] = static_cast<Tdst>(buff[i]);
    }
  }
#endif
#if defined(SENK_WITH_HIP)
  else if constexpr (std::is_same_v<Lsrc, host>) {
    const auto s = 4096;
    Tdst buff[s];
    for (size_t c = 0; c < size; c += s) {
      auto t_s = (c + s < size) ? s : size - c;
      for (size_t i = 0; i < t_s; i++)
        buff[i] = static_cast<Tdst>(src[c + i]);
      SENK_DEVICE_CHECK(
          hipMemcpy(dst + c, buff, sizeof(Tdst) * t_s, hipMemcpyHostToDevice));
    }
  } else if constexpr (std::is_same_v<Lsrc, device>) {
    const auto s = 4096;
    Tsrc buff[s];
    for (size_t c = 0; c < size; c += s) {
      auto t_s = (c + s < size) ? s : size - c;
      SENK_DEVICE_CHECK(
          hipMemcpy(buff, src + c, sizeof(Tsrc) * t_s, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < t_s; i++)
        dst[c + i] = static_cast<Tdst>(buff[i]);
    }
  }
#endif
}

#if defined(SENK_WITH_CUDA)

namespace impl {

template <typename F>
__global__ void cuda_wrapper(F func) {
  func();
}

template <typename T, typename Expl>
__global__ void add_reduce1(Expl expl, T *out, size_t n);
template <typename T>
__global__ void add_reduce2(const T *in, T *out, size_t n);

} // namespace impl

template <>
struct kernel<device> {
  kernel() = delete;
  template <typename F>
  static void single(F func) {
    impl::cuda_wrapper<<<1, 1>>>([=] __device__() mutable { func(); });
  }
  template <typename F>
  static void parallel(size_t num, F func) {
    impl::cuda_wrapper<<<(num + 255) / 256, 256>>>([=] __device__() mutable {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < num)
        func(i);
    });
  }
  template <size_t N, typename F>
  static void parallel(const std::array<size_t, N> &nums, F func) {
    size_t num[N];
    std::copy(nums.begin(), nums.end(), num);
    if constexpr (N == 1) {
      impl::cuda_wrapper<<<(num[0] + 255) / 256, 256>>>(
          [=] __device__() mutable {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < num[0])
              func(i);
          });
    } else if constexpr (N == 2) {
      auto size = num[0] * num[1];
      impl::cuda_wrapper<<<(size + 255) / 256, 256>>>([=] __device__() mutable {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i = tid % num[0];
        size_t j = tid / num[0];
        if (tid < size)
          func(i, j);
      });
      // } else if constexpr (N == 3) {
    } else {
      static_assert([]() { return false; }(), "direct must be on host");
    }
  }
  template <typename F>
  static void parallel2(size_t num, F func) {
    impl::cuda_wrapper<<<(num + 511) / 512, 256>>>([=] __device__() mutable {
      size_t i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
      if (i < num)
        func(i);
      if (i + 256 < num)
        func(i + 256);
    });
  }
  template <uint8_t p, typename F>
  static void parallel_p(size_t num, F func) {
    const uint16_t tpb = 256;
    const auto tc = tpb / p;
    const dim3 block(p, tc);
    const dim3 grid((num + tc - 1) / tc, 1);

    impl::cuda_wrapper<<<grid, block>>>([=] __device__() mutable {
      size_t i = blockIdx.x * blockDim.y + threadIdx.y;
      size_t j = threadIdx.x;
      if (i < num)
        func(i, j);
    });
  }
  // template <typename F> static void parallel2d(size_t num[2], F func);
  template <typename T, typename F>
  static void reduce_add(
      size_t num, T *res, F func, [[maybe_unused]] T *buffer = nullptr) {
    const auto step = 4 * 1024;
    if (num < step) {
      impl::add_reduce1<<<1, 1024>>>(func, res, num);
    } else {
      auto nn = (num + step - 1) / step;
      impl::add_reduce1<<<nn, 1024>>>(func, buffer, num);
      while (nn > step) {
        auto size = nn;
        nn = (nn + step - 1) / step;
        impl::add_reduce2<<<nn, 1024>>>(buffer, buffer + size, size);
        buffer = buffer + size;
      }
      impl::add_reduce2<<<1, 1024>>>(buffer, res, nn);
    }
  }
};

namespace impl {

#define UMNK_WARP_SHFL(VAR)                                                    \
  auto mask = __activemask();                                                  \
  VAR += __shfl_down_sync(mask, VAR, 16, 32);                                  \
  VAR += __shfl_down_sync(mask, VAR, 8, 16);                                   \
  VAR += __shfl_down_sync(mask, VAR, 4, 8);                                    \
  VAR += __shfl_down_sync(mask, VAR, 2, 4);                                    \
  VAR += __shfl_down_sync(mask, VAR, 1, 2);

template <typename T, typename Expl>
__global__ void add_reduce1(Expl expl, T *out, size_t n) {
  __shared__ T sdata[32];
  size_t wid = threadIdx.x / 32;
  size_t tid = threadIdx.x % 32;

  size_t off = blockIdx.x * (blockDim.x * 4) + wid * 128;
  T mySum = (off + tid < n) ? expl(off + tid) : static_cast<T>(0.);
  if (off + tid + 32 < n)
    mySum += expl(off + tid + 32);
  if (off + tid + 64 < n)
    mySum += expl(off + tid + 64);
  if (off + tid + 96 < n)
    mySum += expl(off + tid + 96);
  UMNK_WARP_SHFL(mySum)
  if (tid == 0)
    sdata[wid] = mySum;
  __syncthreads();

  if (wid == 0) {
    T warpSum = sdata[tid];
    UMNK_WARP_SHFL(warpSum)
    if (threadIdx.x == 0)
      out[blockIdx.x] = warpSum;
  }
}

template <typename T>
__global__ void add_reduce2(const T *in, T *out, size_t n) {
  __shared__ T sdata[32];
  size_t wid = threadIdx.x / 32;
  size_t tid = threadIdx.x % 32;

  size_t off = blockIdx.x * (blockDim.x * 4) + wid * 128;
  T mySum = (off + tid < n) ? in[off + tid] : static_cast<T>(0.);
  if (off + tid + 32 < n)
    mySum += in[off + tid + 32];
  if (off + tid + 64 < n)
    mySum += in[off + tid + 64];
  if (off + tid + 96 < n)
    mySum += in[off + tid + 96];
  UMNK_WARP_SHFL(mySum)
  if (tid == 0)
    sdata[wid] = mySum;
  __syncthreads();

  if (wid == 0) {
    T warpSum = sdata[tid];
    UMNK_WARP_SHFL(warpSum)
    if (threadIdx.x == 0)
      out[blockIdx.x] = warpSum;
  }
}

#undef UMNK_WARP_SHFL

} // namespace impl

#endif

#if defined(SENK_WITH_HIP)

namespace impl {

template <typename F>
__global__ void hip_wrapper(F func) {
  func();
}

template <typename T, typename Expl>
__global__ void add_reduce1(Expl expl, T *out, size_t n);
template <typename T>
__global__ void add_reduce2(const T *in, T *out, size_t n);

} // namespace impl

template <>
struct kernel<device> {
  kernel() = delete;
  template <typename F>
  static void single(F func) {
    impl::hip_wrapper<<<1, 1>>>([=] __device__() mutable { func(); });
  }
  template <typename F>
  static void parallel(size_t num, F func) {
    impl::hip_wrapper<<<(num + 255) / 256, 256>>>([=] __device__() mutable {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < num)
        func(i);
    });
  }
  template <size_t N, typename F>
  static void parallel(const std::array<size_t, N> &nums, F func) {
    size_t num[N];
    std::copy(nums.begin(), nums.end(), num);
    if constexpr (N == 1) {
      impl::hip_wrapper<<<(num[0] + 255) / 256, 256>>>(
          [=] __device__() mutable {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < num[0])
              func(i);
          });
    } else if constexpr (N == 2) {
      auto size = num[0] * num[1];
      impl::hip_wrapper<<<(size + 255) / 256, 256>>>([=] __device__() mutable {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i = tid % num[0];
        size_t j = tid / num[0];
        if (tid < size)
          func(i, j);
      });
      // } else if constexpr (N == 3) {
    } else {
      static_assert([]() { return false; }(), "direct must be on host");
    }
  }
  template <typename F>
  static void parallel2(size_t num, F func) {
    impl::hip_wrapper<<<(num + 511) / 512, 256>>>([=] __device__() mutable {
      size_t i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
      if (i < num)
        func(i);
      if (i + 256 < num)
        func(i + 256);
    });
  }
  template <uint8_t p, typename F>
  static void parallel_p(size_t num, F func) {
    const uint16_t tpb = 256;
    const auto tc = tpb / p;
    const dim3 block(p, tc);
    const dim3 grid((num + tc - 1) / tc, 1);

    impl::hip_wrapper<<<grid, block>>>([=] __device__() mutable {
      size_t i = blockIdx.x * blockDim.y + threadIdx.y;
      size_t j = threadIdx.x;
      if (i < num)
        func(i, j);
    });
  }
  // template <typename F> static void parallel2d(size_t num[2], F func);
  template <typename T, typename F>
  static void reduce_add(
      size_t num, T *res, F func, [[maybe_unused]] T *buffer = nullptr) {
    const auto step = 4 * 1024;
    if (num < step) {
      impl::add_reduce1<<<1, 1024>>>(func, res, num);
    } else {
      auto nn = (num + step - 1) / step;
      impl::add_reduce1<<<nn, 1024>>>(func, buffer, num);
      while (nn > step) {
        auto size = nn;
        nn = (nn + step - 1) / step;
        impl::add_reduce2<<<nn, 1024>>>(buffer, buffer + size, size);
        buffer = buffer + size;
      }
      impl::add_reduce2<<<1, 1024>>>(buffer, res, nn);
    }
  }
};

namespace impl {

#define UMNK_WARP_SHFL(VAR)                                                    \
  auto mask = __activemask();                                                  \
  VAR += __shfl_down_sync(mask, VAR, 16, 32);                                  \
  VAR += __shfl_down_sync(mask, VAR, 8, 16);                                   \
  VAR += __shfl_down_sync(mask, VAR, 4, 8);                                    \
  VAR += __shfl_down_sync(mask, VAR, 2, 4);                                    \
  VAR += __shfl_down_sync(mask, VAR, 1, 2);

template <typename T, typename Expl>
__global__ void add_reduce1(Expl expl, T *out, size_t n) {
  __shared__ T sdata[32];
  size_t wid = threadIdx.x / 32;
  size_t tid = threadIdx.x % 32;

  size_t off = blockIdx.x * (blockDim.x * 4) + wid * 128;
  T mySum = (off + tid < n) ? expl(off + tid) : static_cast<T>(0.);
  if (off + tid + 32 < n)
    mySum += expl(off + tid + 32);
  if (off + tid + 64 < n)
    mySum += expl(off + tid + 64);
  if (off + tid + 96 < n)
    mySum += expl(off + tid + 96);
  UMNK_WARP_SHFL(mySum)
  if (tid == 0)
    sdata[wid] = mySum;
  __syncthreads();

  if (wid == 0) {
    T warpSum = sdata[tid];
    UMNK_WARP_SHFL(warpSum)
    if (threadIdx.x == 0)
      out[blockIdx.x] = warpSum;
  }
}

template <typename T>
__global__ void add_reduce2(const T *in, T *out, size_t n) {
  __shared__ T sdata[32];
  size_t wid = threadIdx.x / 32;
  size_t tid = threadIdx.x % 32;

  size_t off = blockIdx.x * (blockDim.x * 4) + wid * 128;
  T mySum = (off + tid < n) ? in[off + tid] : static_cast<T>(0.);
  if (off + tid + 32 < n)
    mySum += in[off + tid + 32];
  if (off + tid + 64 < n)
    mySum += in[off + tid + 64];
  if (off + tid + 96 < n)
    mySum += in[off + tid + 96];
  UMNK_WARP_SHFL(mySum)
  if (tid == 0)
    sdata[wid] = mySum;
  __syncthreads();

  if (wid == 0) {
    T warpSum = sdata[tid];
    UMNK_WARP_SHFL(warpSum)
    if (threadIdx.x == 0)
      out[blockIdx.x] = warpSum;
  }
}

#undef UMNK_WARP_SHFL

} // namespace impl

#endif

} // namespace senk

#endif // SENK_CORE_KERNEL_HPP
