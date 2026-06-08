#ifndef SENK_MATRIX_SIGMA_HPP
#define SENK_MATRIX_SIGMA_HPP

#include "senk/core/sort.hpp"
#include "senk/core/tensor.hpp"
#include "senk/matrix/base.hpp"

namespace senk {

template <typename T, class L, typename I, typename S>
struct CSR;

template <class V, class L, typename I, typename S>
struct PackCSR;

template <uint64_t S, class B>
struct SIGMA;
template <class B>
using SIGMA15 = SIGMA<32768, B>;

namespace impl {

template <std::uint64_t N>
constexpr auto sigma_uint_impl() {
  if constexpr (N <= UINT8_MAX + 1)
    return std::uint8_t{};
  else if constexpr (N <= UINT16_MAX + 1)
    return std::uint16_t{};
  else if constexpr (N <= UINT32_MAX + 1)
    return std::uint32_t{};
  else
    return std::uint64_t{};
}

template <uint64_t N>
using sigma_uint_t = decltype(sigma_uint_impl<N>());

template <uint64_t s, typename T, typename I, typename S>
std::tuple<vector<sigma_uint_t<s>, host>, CSR<T, host, I, S>> sigma(
    const CSR<T, host, I, S> &in) {
  using sig_t = sigma_uint_t<s>;
  using idx_t = typename CSR<T, host, I, S>::idx_t;
  auto p = vector<sig_t, host>(in.nrows());
  for (idx_t i = 0; i < in.nrows(); ++i)
    p[i] = i % s;
  auto key = vector<idx_t, host>(in.nrows());
  for (idx_t i = 0; i < in.nrows(); i++)
    key[i] = in.rptr[i + 1] - in.rptr[i];
  for (idx_t i = 0; i < in.nrows(); i += s) {
    auto e = std::cmp_greater(i + s, in.nrows()) ? in.nrows() : i + s;
    sort::pack_sort<sort::order::desc>(i, e, key.raw(), p.raw());
  }
  auto res = CSR<T, host, I, S>(in.shape, in.val.shape(0), in.attr);
  idx_t nnz = 0;
  res.rptr[0] = nnz;
  for (idx_t i = 0; i < in.nrows(); i++) {
    auto sig = i / s * s;
    idx_t row = sig + p[i];
    for (auto j = in.rptr[row]; j < in.rptr[row + 1]; j++) {
      res.val[nnz] = in.val[j];
      res.col[nnz++] = in.col[j];
    }
    res.rptr[i + 1] = nnz;
  }
  return {p, res};
}

template <class V, typename I, typename S>
std::tuple<vector<uint8_t, host>, PackCSR<V, host, I, S>> sigma(
    const PackCSR<V, host, I, S> &in, uint16_t s = 256) {
  using idx_t = typename PackCSR<V, host, I, S>::idx_t;
  auto p = vector<uint8_t, host>(in.nrows());
#pragma omp parallel for
  for (idx_t i = 0; i < in.nrows(); ++i)
    p[i] = i % s;
  auto key = vector<idx_t, host>(in.nrows());
#pragma omp parallel for
  for (idx_t i = 0; i < in.nrows(); i++)
    key[i] = in.rptr[i + 1] - in.rptr[i];
#pragma omp parallel for
  for (idx_t i = 0; i < in.nrows(); i += s) {
    auto e = (i + s > in.nrows()) ? in.nrows() : i + s;
    sort::pack_sort<sort::order::desc>(i, e, key.raw(), p.raw());
  }

  auto res = PackCSR<V, host, I, S>(
      in.shape, in.val.shape(0), in.attr, in.bit, in.left);
  idx_t nnz = 0;
  res.rptr[0] = nnz;
  for (idx_t i = 0; i < in.nrows(); i++) {
    auto sig = i / s * s;
    idx_t row = sig + p[i];
    for (auto j = in.rptr[row]; j < in.rptr[row + 1]; j++)
      res.val[nnz++] = in.val[j];
    res.rptr[i + 1] = nnz;
  }
  return {p, res};
}

} // namespace impl

} // namespace senk

#endif
