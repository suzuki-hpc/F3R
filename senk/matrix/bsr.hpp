#ifndef SENK_MATRIX_BSR_HPP
#define SENK_MATRIX_BSR_HPP

#include "senk/core/tools.hpp"

#include "senk/core/tensor.hpp"
#include "senk/matrix/base.hpp"
#include "senk/matrix/csr.hpp"

namespace senk {

namespace impl {

template <typename T, typename I, typename S>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> csr_to_bcsr(
    const CSR<T, host, I, S> &in, uint16_t bnl, uint16_t bnw);

template <typename T, typename I, typename S>
auto csr_to_cubcsr(const CSR<T, host, I, S> &in, uint16_t bnl, uint16_t bnw);

template <uint16_t bnl, uint16_t bnw, typename loc_t, typename val_t,
    typename idx_t, typename srl_t, typename in_t, typename out_t>
void bsr_apply_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out);
template <uint16_t bnl, uint16_t bnw, typename loc_t, typename val_t,
    typename idx_t, typename srl_t, typename rhs_t, typename in_t,
    typename out_t>
void bsr_residual_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const rhs_t *rhs, const in_t *in,
    out_t *out);

} // namespace impl

template <uint16_t bnl, uint16_t bnw, typename T, class L, typename I = index_t,
    typename S = serial_t>
struct BSR : public has_val_t<T>,
             public has_loc_t<L>,
             public has_idx_t<I>,
             public has_srl_t<S>,
             public model::is_operator<BSR<bnl, bnw, T, L, I, S>> {
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  using typename has_idx_t<I>::idx_t;
  using typename has_srl_t<S>::srl_t;

  attribute attr;
  std::array<idx_t, 2> shape;
  vector<val_t, loc_t> val;
  vector<idx_t, loc_t> col;
  vector<srl_t, loc_t> rptr;

  template <typename T2>
  explicit BSR(const CSR<T2, host, I, S> &in)
      : BSR(in.attr, in.shape, impl::csr_to_bcsr(in, bnl, bnw)) {}

  BSR(const BSR &in) = default;
  template <typename T2, class L2, typename I2, typename S2>
  explicit BSR(const BSR<bnl, bnw, T2, L2, I2, S2> &in)
      : attr(in.attr), shape(in.shape), val(in.val), col(in.col),
        rptr(in.rptr) {}

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::bsr_apply_impl<bnl, bnw, loc_t>(nrows_impl(), ncols_impl(), val.raw(),
        col.raw(), rptr.raw(), in.raw(), out.raw());
  }
  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::bsr_residual_impl<bnl, bnw, loc_t>(nrows_impl(), ncols_impl(),
        val.raw(), col.raw(), rptr.raw(), rhs.raw(), in.raw(), out.raw());
  }
  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }

  friend struct model::is_operator<BSR<bnl, bnw, T, L, I, S>>;

private:
  template <typename T2>
  explicit BSR(const attribute &attr, const std::array<I, 2> &shape,
      std::tuple<vector<T2, host>, vector<I, host>, vector<S, host>> &&t)
      : attr(attr), shape(shape), val(std::get<0>(t)), col(std::get<1>(t)),
        rptr(std::get<2>(t)) {}
};

template <uint16_t bnl, uint16_t bnw, typename T, class L, typename I = index_t,
    typename S = serial_t>
struct cuBSR : public has_val_t<T>,
               public has_loc_t<L>,
               public has_idx_t<I>,
               public has_srl_t<S>,
               public model::is_operator<cuBSR<bnl, bnw, T, L, I, S>> {
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  using typename has_idx_t<I>::idx_t;
  using typename has_srl_t<S>::srl_t;

  attribute attr;
  std::array<idx_t, 2> shape;
  vector<val_t, loc_t> val;
  vector<idx_t, loc_t> col;
  vector<srl_t, loc_t> rptr;

  template <typename T2>
  explicit cuBSR(const CSR<T2, host, I, S> &in)
      : cuBSR(in.attr, impl::csr_to_cubcsr(in, bnl, bnw)) {}

  cuBSR(const cuBSR &in) = default;
  template <typename T2, class L2>
  explicit cuBSR(const cuBSR<bnl, bnw, T2, L2> &in)
      : attr(in.attr), shape(in.shape), val(in.val), col(in.col),
        rptr(in.rptr) {}
  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }
  friend struct model::is_operator<BSR<bnl, bnw, T, L, I, S>>;

private:
  explicit cuBSR(const attribute &attr,
      std::tuple<I, I, vector<T, host>, vector<I, host>, vector<S, host>> &&t)
      : attr(attr), shape({std::get<0>(t), std::get<1>(t)}),
        val(std::get<2>(t)), col(std::get<3>(t)), rptr(std::get<4>(t)) {}
};

namespace impl {

template <typename T, typename I, typename S>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> csr_to_bcsr(
    const CSR<T, host, I, S> &in, uint16_t bnl, uint16_t bnw) {
  using idx_t = I;
  using val_t = T;
  using srl_t = S;
  const auto area = bnl * bnw;
  auto nrows = in.nrows();
  auto ncols = in.ncols();
  auto bnrows = (nrows + bnl - 1) / bnl;

  auto brptr = vector<srl_t, host>(bnrows + 1);

  srl_t cnt = 0;
  brptr[0] = cnt;
  auto ptr = vector<idx_t, host>(bnl);
  for (idx_t i = 0; i < nrows; i += bnl) {
    for (uint16_t j = 0; j < bnl; j++)
      ptr[j] = (i + j < nrows) ? in.rptr[i + j] : -1;
    while (true) {
      idx_t min = ncols;
      for (uint16_t j = 0; j < bnl; j++)
        if (ptr[j] != -1 && in.col[ptr[j]] < min)
          min = in.col[ptr[j]];
      if (min == ncols)
        break;
      for (uint16_t j = 0; j < bnl; j++) {
        if (ptr[j] == -1)
          continue;
        while (in.col[ptr[j]] / bnw == min / bnw) {
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
  auto bval = vector<val_t, host>(cnt * area);
  auto bidx = vector<idx_t, host>(cnt);
  cnt = 0;
  for (idx_t i = 0; i < nrows; i += bnl) {
    for (uint16_t j = 0; j < bnl; j++)
      ptr[j] = (i + j < nrows) ? in.rptr[i + j] : -1;
    while (true) {
      int min = ncols;
      for (uint16_t j = 0; j < bnl; j++) {
        if (ptr[j] != -1 && in.col[ptr[j]] < min)
          min = in.col[ptr[j]];
      }
      if (min == ncols)
        break;
      for (uint16_t j = 0; j < bnl; j++) {
        if (ptr[j] == -1)
          continue;
        while (in.col[ptr[j]] / bnw == min / bnw) {
          const idx_t off = in.col[ptr[j]] % bnw;
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
  return {bval, bidx, brptr};
}

template <typename T, typename I, typename S>
auto csr_to_cubcsr(const CSR<T, host, I, S> &in, uint16_t bnl, uint16_t bnw) {
  using idx_t = I;
  using val_t = T;
  using srl_t = S;
  const auto area = bnl * bnw;
  auto nrows = in.nrows();
  auto ncols = in.ncols();
  auto bnrows = (nrows + bnl - 1) / bnl;
  auto bncols = (ncols + bnl - 1) / bnw;
  auto _nrows = bnrows * bnl;
  auto _ncols = bncols * bnw;

  auto brptr = vector<srl_t, host>(bnrows + 1);

  srl_t cnt = 0;
  brptr[0] = cnt;
  auto ptr = vector<srl_t, host>(bnl);
  for (idx_t i = 0; i < nrows; i += bnl) {
    auto _bnl = (i + bnl < nrows) ? bnl : nrows - i;
    for (uint16_t j = 0; j < _bnl; j++)
      ptr[j] = in.rptr[i + j];
    while (true) {
      idx_t min = _ncols;
      for (uint16_t j = 0; j < _bnl; j++)
        if (ptr[j] != -1 && in.col[ptr[j]] < min)
          min = in.col[ptr[j]];
      if (min == _ncols)
        break;
      for (uint16_t j = 0; j < _bnl; j++) {
        if (ptr[j] == -1)
          continue;
        while (in.col[ptr[j]] / bnw == min / bnw) {
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
  auto bval = vector<val_t, host>(cnt * area);
  auto bidx = vector<idx_t, host>(cnt);
  cnt = 0;
  for (idx_t i = 0; i < nrows; i += bnl) {
    auto _bnl = (i + bnl < nrows) ? bnl : nrows - i;
    for (uint16_t j = 0; j < _bnl; j++)
      ptr[j] = in.rptr[i + j];
    while (true) {
      int min = _ncols;
      for (uint16_t j = 0; j < _bnl; j++) {
        if (ptr[j] != -1 && in.col[ptr[j]] < min)
          min = in.col[ptr[j]];
      }
      if (min == _ncols)
        break;
      for (uint16_t j = 0; j < _bnl; j++) {
        if (ptr[j] == -1)
          continue;
        while (in.col[ptr[j]] / bnw == min / bnw) {
          const idx_t off = in.col[ptr[j]] % bnw;
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
  return std::make_tuple(_nrows, _ncols, bval, bidx, brptr);
}

#define KMM_BCSR_SIMD(col)                                                     \
  _Pragma("omp simd simdlen(bnl)") for (uint16_t k = 0; k < bnl; ++k) {        \
    t[k] += val[j * area + bnl * (col) + k] * x[(col)];                        \
  }

template <uint16_t bnl, uint16_t bnw, typename loc_t, typename val_t,
    typename idx_t, typename srl_t, typename in_t, typename out_t>
void bsr_apply_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out) {
  const auto bncols = (ncols + bnw - 1) / bnw;
  kernel<loc_t>::parallel((nrows + bnl - 1) / bnl, [=](idx_t bidx) mutable {
    const auto i = bidx * bnl;
    const auto area = bnl * bnw;
    using tmp_t = decltype(std::declval<val_t>() * std::declval<in_t>());
    tmp_t t[bnl] = {0};
    in_t x[bnw] = {0};
    for (auto j = rptr[bidx]; j < rptr[bidx + 1]; ++j) {
      auto x_ind = col[j] * bnw;
      if (col[j] != bncols - 1) {
        for (int k = 0; k < bnw; k++)
          x[k] = in[x_ind + k];
      } else {
        for (int k = 0; k < (ncols - 1) % bnw + 1; k++)
          x[k] = in[x_ind + k];
        for (int k = (ncols - 1) % bnw + 1; k < bnw; k++)
          x[k] = static_cast<in_t>(0);
      }
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
    if (i < nrows - bnl) {
#pragma omp simd simdlen(bnl)
      for (uint16_t j = 0; j < bnl; ++j)
        out[i + j] = t[j];
    } else {
      for (uint16_t j = 0; j < nrows - i; ++j)
        out[i + j] = t[j];
    }
  });
}

template <uint16_t bnl, uint16_t bnw, typename loc_t, typename val_t,
    typename idx_t, typename srl_t, typename rhs_t, typename in_t,
    typename out_t>
void bsr_residual_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const rhs_t *rhs, const in_t *in,
    out_t *out) {
  const auto bncols = (ncols + bnw - 1) / bnw;
  kernel<loc_t>::parallel((nrows + bnl - 1) / bnl, [=](idx_t bidx) mutable {
    const auto i = bidx * bnl;
    const auto area = bnl * bnw;
    using tmp_t = decltype(std::declval<val_t>() * std::declval<in_t>());
    tmp_t t[bnl] = {0};
    in_t x[bnw] = {0};
    for (auto j = rptr[bidx]; j < rptr[bidx + 1]; ++j) {
      auto x_ind = col[j] * bnw;
      if (col[j] != bncols - 1) {
        for (int k = 0; k < bnw; k++)
          x[k] = in[x_ind + k];
      } else {
        for (int k = 0; k < (ncols - 1) % bnw + 1; k++)
          x[k] = in[x_ind + k];
        for (int k = (ncols - 1) % bnw + 1; k < bnw; k++)
          x[k] = static_cast<in_t>(0);
      }
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
    if (i < nrows - bnl) {
#pragma omp simd simdlen(bnl)
      for (uint16_t j = 0; j < bnl; ++j)
        out[i + j] = rhs[i + j] - t[j];
    } else {
      for (uint16_t j = 0; j < nrows - i; ++j)
        out[i + j] = rhs[i + j] - t[j];
    }
  });
}

#undef KMM_BCSR_SIMD

} // namespace impl

} // namespace senk

#endif // SENK_MATRIX_BSR_HPP