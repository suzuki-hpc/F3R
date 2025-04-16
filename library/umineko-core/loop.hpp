#ifndef UMINEKO_CORE_LOOP_HPP
#define UMINEKO_CORE_LOOP_HPP

#include <cstddef>
#include <type_traits>

#include "umineko-core/interface.hpp"

namespace kmm {

template <class L, std::enable_if_t<is_locator_v<L>, std::nullptr_t> = nullptr>
struct exec {};

template <> struct exec<host> {
  template <typename F> static void seq(F func) { func(); }
  template <typename F> static void seq_for(size_t size, F func) {
    for (size_t i = 0; i < size; i++) {
      func(i);
    }
  }
  template <typename F> static void para_for(size_t size, F func) {
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      func(i);
    }
  }
  template <typename F> static void para_for_2d(size_t n, size_t m, F func) {
// #pragma omp parallel
//     {
//       for (size_t j = 0; j < m; j++) {
// #pragma omp for
//         for (size_t i = 0; i < n; i++) {
//           func(i, j);
//         }
//       }
//     }
#pragma omp parallel for
    for (size_t i = 0; i < n * m; i++) {
      func(i % n, i / n);
    }
  }
  template <typename T, typename T2, typename F>
  static void para_reduce(size_t size, T *res, T2 init, F func,
                          [[maybe_unused]] T *buff = nullptr) {
    res[0] = T(init);
#pragma omp parallel for reduction(+ : res[0])
    for (size_t i = 0; i < size; i++) {
      res[0] += func(i);
    }
  }
};

#if __has_include("cuda_runtime_api.h")

namespace impl {

template <typename F> static __global__ void _exec(F func) { func(); }

#define YAMAME_REDUCTION_UNROLL(a, b)                                          \
  if ((blockSize >= a) && (tid < b)) {                                         \
    sdata[tid] = mySum = mySum + sdata[tid + b];                               \
    __syncthreads();                                                           \
  }

template <uint16_t blockSize, typename T, typename Expl>
__global__ static void reduce1(Expl expl, T *out, size_t n) {
  // thread_block cta = this_thread_block();
  extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
  T *sdata = reinterpret_cast<T *>(_sdata);
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  // T mySum = (i < n) ? expl(i) : static_cast<T>(0.);
  // if (i + blockSize < n)
  //   mySum += expl(i + blockDim.x);
  int grid_size = 2 * blockDim.x * gridDim.x;
  T mySum = static_cast<T>(0.);
  while (i < n) {
    mySum += expl(i);
    if ((i + blockSize) < n)
      mySum += expl(i + blockSize);
    i += grid_size;
  }

  sdata[tid] = mySum;
  // sync(cta);
  __syncthreads();
  YAMAME_REDUCTION_UNROLL(512, 256)
  YAMAME_REDUCTION_UNROLL(256, 128)
  YAMAME_REDUCTION_UNROLL(128, 64)
  YAMAME_REDUCTION_UNROLL(64, 32)
  YAMAME_REDUCTION_UNROLL(32, 16)
  YAMAME_REDUCTION_UNROLL(16, 8)
  YAMAME_REDUCTION_UNROLL(8, 4)
  YAMAME_REDUCTION_UNROLL(4, 2)
  YAMAME_REDUCTION_UNROLL(2, 1)
  // if (cta.thread_rank() == 0) out[blockIdx.x] = mySum;
  if (tid == 0)
    out[blockIdx.x] = mySum;
}

template <uint16_t blockSize, typename T>
__global__ static void reduce2(const T *in, T *out, size_t n) {
  // thread_block cta = this_thread_block();
  extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
  T *sdata = reinterpret_cast<T *>(_sdata);
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? in[i] : static_cast<T>(0.);
  if (i + blockSize < n)
    mySum += in[i + blockDim.x];
  // int grid_size = 2 * blockDim.x * gridDim.x;
  // T mySum = static_cast<T>(0.);
  // while (i < n) {
  //   mySum += in[i];
  //   if ((i + blockSize) < n)
  //     mySum += in[i + blockSize];
  //   i += grid_size;
  // }

  sdata[tid] = mySum;
  // sync(cta);
  __syncthreads();
  YAMAME_REDUCTION_UNROLL(512, 256)
  YAMAME_REDUCTION_UNROLL(256, 128)
  YAMAME_REDUCTION_UNROLL(128, 64)
  YAMAME_REDUCTION_UNROLL(64, 32)
  YAMAME_REDUCTION_UNROLL(32, 16)
  YAMAME_REDUCTION_UNROLL(16, 8)
  YAMAME_REDUCTION_UNROLL(8, 4)
  YAMAME_REDUCTION_UNROLL(4, 2)
  YAMAME_REDUCTION_UNROLL(2, 1)
  // if (cta.thread_rank() == 0) out[blockIdx.x] = mySum;
  if (tid == 0)
    out[blockIdx.x] = mySum;
}
#undef YAMAME_REDUCTION_UNROLL

} // namespace impl

template <> struct exec<device> {
  template <typename F> static void seq(F func) {
    impl::_exec<<<1, 1>>>([=] __device__() mutable { func(); });
  }
  template <typename F> static void seq_for(size_t size, F func) {
    impl::_exec<<<1, 1>>>([=] __device__() mutable {
      for (size_t i = 0; i < size; i++) {
        func(i);
      }
    });
  };
  template <typename F, uint16_t Nt = 256>
  static void para_for(size_t size, F func) {
    impl::_exec<<<(size + Nt - 1) / Nt, Nt>>>([=] __device__() mutable {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < size) {
        func(i);
      }
    });
  };
  template <typename F, uint16_t Nt = 256>
  static void para_for_2d(size_t n, size_t m, F func) {
    impl::_exec<<<(n * m + Nt - 1) / Nt, Nt>>>([=] __device__() mutable {
      size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n * m) {
        func(i % n, i / n);
      }
    });
  };
  template <typename T, typename T2, typename F>
  static void para_reduce(size_t size, T *res, T2 init, F func, T *buff) {
    const auto SM = 512 * sizeof(T);
    const auto n1 = (size + 1024 - 1) / 1024;
    impl::reduce1<512, T><<<128, 512, SM>>>(func, buff, size);
    const auto n2 = (n1 + 1024 - 1) / 1024;
    impl::reduce2<512, T><<<n2, 512, SM>>>(buff, buff + n1, n1);
    const auto n3 = (n2 + 1024 - 1) / 1024;
    impl::reduce2<512, T><<<n3, 512, SM>>>(buff + n1, res, n2);
  };
};

#endif

} // namespace kmm

#endif
