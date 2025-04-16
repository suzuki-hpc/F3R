#ifndef UMINEKO_SPARSE_MATRIX_BCSR_HPP
#define UMINEKO_SPARSE_MATRIX_BCSR_HPP

#include "umineko-core/tensor.hpp"
#include "umineko-sparse/matrix/base.hpp"

#include "umineko-sparse/matrix/csr.hpp"

namespace kmm {

namespace impl {

template <typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<idx_t, host>, vector<idx_t, host>>
csr_to_bcsr(const CSR<T2, host> &in, uint16_t bnl, uint16_t bnw);

} // namespace impl

template <uint16_t bnl, uint16_t bnw, typename T, class L> struct BCSR : spmat {
  vector<T, L> val;
  vector<idx_t, L> idx;
  vector<idx_t, L> rptr;

  template <typename T2>
  explicit BCSR(const CSR<T2, host> &in) : BCSR(impl::csr_to_bcsr(in, bnl, bnw)) {
    spmat::copy_attrs(in);
  }

  BCSR(const BCSR &in) = default;
  template <typename T2, class L2>
  explicit BCSR(const BCSR<bnl, bnw, T2, L2> &in)
      : spmat(in), val(in.val), idx(in.idx), rptr(in.rptr) {}

  template <typename in_t, typename out_t>
  void operate(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_BCSR_SIMD(col)                                                          \
  _Pragma("omp simd simdlen(bnl)") for (uint16_t k = 0; k < bnl; ++k) {             \
    t[k] += val[j * area + bnl * (col) + k] * in[x_ind + (col)];                    \
  }
    exec<L>::para_for(out.size(0) / bnl, [=, *this](idx_t bidx) mutable {
      const auto i = bidx * bnl;
      const auto area = bnl * bnw;
      decltype(std::declval<T &>() * std::declval<in_t &>()) t[bnl] = {0};
      for (auto j = rptr[bidx]; j < rptr[bidx + 1]; ++j) {
        auto x_ind = idx[j] * bnw;
        KMM_BCSR_SIMD(0)
        if constexpr (bnw == 2 || bnw == 4 || bnw == 8) {
          KMM_BCSR_SIMD(1)
        }
        if constexpr (bnw == 4 || bnw == 8) {
          KMM_BCSR_SIMD(2)
          KMM_BCSR_SIMD(3)
        }
        if constexpr (bnw == 8) {
          KMM_BCSR_SIMD(4)
          KMM_BCSR_SIMD(5)
          KMM_BCSR_SIMD(6)
          KMM_BCSR_SIMD(7)
        }
      }
#pragma omp simd simdlen(bnl)
      for (uint16_t j = 0; j < bnl; ++j)
        out[i + j] = t[j];
    });
#undef KMM_BCSR_SIMD
  }
  template <typename rhs_t, typename in_t, typename res_t>
  void compute_residual(const vector<rhs_t, L> &rhs, const vector<in_t, L> &in,
      vector<res_t, L> res) const {
#define KMM_BCSR_SIMD(col)                                                          \
  _Pragma("omp simd simdlen(bnl)") for (uint16_t k = 0; k < bnl; ++k) {             \
    t[k] += val[j * area + bnl * (col) + k] * in[x_ind + (col)];                    \
  }
    exec<L>::para_for(res.size(0) / bnl, [=, *this](idx_t bidx) mutable {
      const auto i = bidx * bnl;
      const auto area = bnl * bnw;
      decltype(std::declval<T &>() * std::declval<in_t &>()) t[bnl] = {0};
      for (auto j = rptr[bidx]; j < rptr[bidx + 1]; ++j) {
        auto x_ind = idx[j] * bnw;
        KMM_BCSR_SIMD(0)
        if constexpr (bnw == 2 || bnw == 4 || bnw == 8) {
          KMM_BCSR_SIMD(1)
        }
        if constexpr (bnw == 4 || bnw == 8) {
          KMM_BCSR_SIMD(2)
          KMM_BCSR_SIMD(3)
        }
        if constexpr (bnw == 8) {
          KMM_BCSR_SIMD(4)
          KMM_BCSR_SIMD(5)
          KMM_BCSR_SIMD(6)
          KMM_BCSR_SIMD(7)
        }
      }
#pragma omp simd simdlen(bnl)
      for (uint16_t j = 0; j < bnl; ++j)
        res[i + j] = rhs[i + j] - t[j];
    });
#undef KMM_BCSR_SIMD
  }

private:
  template <typename T2>
  explicit BCSR(std::tuple<impl::shape, idx_t, vector<T2, host>, vector<idx_t, host>,
      vector<idx_t, host>> &&t)
      : spmat(std::get<0>(t), std::get<1>(t)), val(std::get<2>(t)),
        idx(std::get<3>(t)), rptr(std::get<4>(t)) {}
};

namespace impl {

template <typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<idx_t, host>, vector<idx_t, host>>
csr_to_bcsr(const CSR<T2, host> &in, uint16_t bnl, uint16_t bnw) {
  auto nrows = in.nrows();
  // auto ncols = in.ncols();
  auto brptr = vector<idx_t, host>(nrows / bnl + 1);
  const auto area = bnl * bnw;
  idx_t cnt = 0;
  brptr[0] = 0;
  auto ptr = vector<idx_t, host>(bnl);
  for (idx_t i = 0; i < nrows; i += bnl) { // Count the number of block
    for (uint16_t j = 0; j < bnl; j++)     // Initialize "ptr"
      ptr[j] = in.rptr[i + j];
    while (true) { // Find minimum col value
      idx_t min = nrows;
      for (uint16_t j = 0; j < bnl; j++) {
        if (ptr[j] != -1 && in.idx[ptr[j]] < min)
          min = in.idx[ptr[j]];
      }
      if (min == nrows)
        break;
      // Increment ptr[j] whose col-idx is in min block
      for (uint16_t j = 0; j < bnl; j++) {
        if (ptr[j] == -1)
          continue;
        while (in.idx[ptr[j]] / bnw == min / bnw) {
          ++ptr[j];
          if (ptr[j] >= in.rptr[i + j + 1]) {
            ptr[j] = -1;
            break;
          }
        }
      }
      cnt++;
    }
    brptr[i / bnl + 1] = cnt;
  }
  auto bval = vector<T2, host>(cnt * area);
  auto bidx = vector<idx_t, host>(cnt);
  cnt = 0;
  for (idx_t i = 0; i < nrows; i += bnl) { // Assign val to bval
    for (uint16_t j = 0; j < bnl; j++)     // Initialize "ptr"
      ptr[j] = in.rptr[i + j];
    while (true) {
      int min = nrows;
      for (uint16_t j = 0; j < bnl; j++) {
        if (ptr[j] != -1 && in.idx[ptr[j]] < min)
          min = in.idx[ptr[j]];
      }
      if (min == nrows)
        break;
      for (uint16_t j = 0; j < bnl; j++) {
        if (ptr[j] == -1)
          continue;
        while (in.idx[ptr[j]] / bnw == min / bnw) {
          const idx_t off = in.idx[ptr[j]] % bnw;
          bval[cnt * area + off * bnl + j] = in.val[ptr[j]];
          ++ptr[j];
          if (ptr[j] >= in.rptr[i + j + 1]) {
            ptr[j] = -1;
            break;
          }
        }
      }
      bidx[cnt] = min / bnw;
      cnt++;
    }
  }
  return {in.shp, cnt * bnl * bnw, bval, bidx, brptr};
}

template <typename T2, uint16_t bnl, uint16_t bnw>
std::tuple<shape, idx_t, vector<T2, host>, vector<idx_t, host>, vector<idx_t, host>>
bcsr_to_csr(const BCSR<bnl, bnw, T2, host> &B) {
  auto num_block = B.rptr[B.nrows() / bnl];
  const auto area = bnl * bnw;
  auto val = vector<T2, host>(num_block * bnl * bnw);
  auto idx = vector<idx_t, host>(num_block * bnl * bnw);
  auto rptr = vector<idx_t, host>(B.nrows() + 1);
  idx_t nnz = 0;
  rptr[0] = nnz;
  for (idx_t i = 0; i < B.nrows(); i++) {
    idx_t bid = i / bnl;
    idx_t id = i % bnl;
    for (auto bj = B.rptr[bid]; bj < B.rptr[bid + 1]; ++bj) {
      for (idx_t j = 0; j < bnw; j++) {
        idx[nnz] = B.idx[bj] * bnw + j;
        val[nnz] = B.val[bj * area + j * bnl + id];
        nnz++;
      }
    }
    rptr[i + 1] = nnz;
  }
  return {B.get_shape(), nnz, val, idx, rptr};
}

} // namespace impl

} // namespace kmm

#endif // UMINEKO_SPARSE_MATRIX_BCSR_HPP
