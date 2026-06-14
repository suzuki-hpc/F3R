#ifndef SENK_MATRIX_SELL_HPP
#define SENK_MATRIX_SELL_HPP

#include "senk/core/tools.hpp"

#include "senk/core/io.hpp"
#include "senk/core/tensor.hpp"
#include "senk/matrix/base.hpp"
#include "senk/models.hpp"

#include "senk/matrix/csr.hpp"
#include "senk/matrix/sigma.hpp"

namespace senk {

namespace impl {

enum class sell_align { left, right };

template <typename T, typename I, typename S, impl::sell_align align>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> csr_to_sell(
    const CSR<T, host, I, S> &in, uint16_t w);
template <typename T, typename I, typename S, impl::sell_align align>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> csr_to_cusell(
    const CSR<T, host, I, S> &in, uint16_t w);

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void cusell_apply_impl(const idx_t _nrows, const idx_t nrows, const uint16_t w,
    const val_t *val, const idx_t *col, const srl_t *sptr, const in_t *in,
    out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void cusell_residual_impl(const idx_t _nrows, const idx_t nrows,
    const uint16_t w, const val_t *val, const idx_t *col, const srl_t *sptr,
    const rhs_t *rhs, const in_t *in, out_t *out);

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename in_t, typename out_t>
void cusell_sigma_apply_impl(const idx_t _nrows, const idx_t nrows,
    const uint16_t w, const uint64_t s, const sig_t *p, const val_t *val,
    const idx_t *col, const srl_t *sptr, const in_t *in, out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename rhs_t, typename in_t, typename out_t>
void cusell_sigma_residual_impl(const idx_t _nrows, const idx_t nrows,
    const uint16_t w, const uint64_t s, const sig_t *p, const val_t *val,
    const idx_t *col, const srl_t *sptr, const rhs_t *rhs, const in_t *in,
    out_t *out);

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void sell_apply_impl(const idx_t nrows, const uint16_t w, const val_t *val,
    const idx_t *col, const srl_t *sptr, const in_t *in, out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void sell_residual_impl(const idx_t nrows, const uint16_t w, const val_t *val,
    const idx_t *col, const srl_t *sptr, const rhs_t *rhs, const in_t *in,
    out_t *out);

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename in_t, typename out_t>
void sell_sigma_apply_impl(const idx_t nrows, const uint16_t w,
    const uint64_t s, const sig_t *p, const val_t *val, const idx_t *col,
    const srl_t *sptr, const in_t *in, out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename rhs_t, typename in_t, typename out_t>
void sell_sigma_residual_impl(const idx_t nrows, const uint16_t w,
    const uint64_t s, const sig_t *p, const val_t *val, const idx_t *col,
    const srl_t *sptr, const rhs_t *rhs, const in_t *in, out_t *out);

} // namespace impl

template <uint16_t w, typename T, class L, typename I = index_t,
    typename S = serial_t, impl::sell_align align = impl::sell_align::left>
struct cuSELL : public has_val_t<T>,
                public has_loc_t<L>,
                public has_idx_t<I>,
                public has_srl_t<S>,
                public model::is_operator<cuSELL<w, T, L, I, S, align>> {
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  using typename has_idx_t<I>::idx_t;
  using typename has_srl_t<S>::srl_t;

  attribute attr;
  std::array<idx_t, 2> shape;
  vector<val_t, loc_t> val;
  vector<idx_t, loc_t> col;
  vector<srl_t, loc_t> sptr;
  idx_t _nrows;

  template <typename T2>
  cuSELL(const CSR<T2, host, I, S> &in)
      : cuSELL(in.attr, in.shape, impl::csr_to_cusell<T2, I, S, align>(in, w)) {
  }
  cuSELL(const cuSELL &) = default;
  template <typename T2, class L2>
  cuSELL(const cuSELL<w, T2, L2, I, S, align> &in)
      : attr(in.attr), shape(in.shape), val(in.val), col(in.col), sptr(in.sptr),
        _nrows(in._nrows) {}

private:
  template <typename T2>
  cuSELL(const attribute &attr, const std::array<idx_t, 2> &shape,
      const std::tuple<vector<T2, host>, vector<idx_t, host>,
          vector<srl_t, host>> &tt)
      : attr(attr), shape(shape), val(std::get<0>(tt)), col(std::get<1>(tt)),
        sptr(std::get<2>(tt)), _nrows((shape[0] + w - 1) / w * w) {}

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::cusell_apply_impl<loc_t>(_nrows, nrows_impl(), w, val.raw(),
        col.raw(), sptr.raw(), in.raw(), out.raw());
  }
  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::cusell_residual_impl<loc_t>(_nrows, nrows_impl(), w, val.raw(),
        col.raw(), sptr.raw(), rhs.raw(), in.raw(), out.raw());
  }
  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }

  friend struct model::is_operator<cuSELL<w, T, L, I, S, align>>;
};

template <uint16_t w, typename T, class L, typename I = index_t,
    typename S = serial_t, impl::sell_align align = impl::sell_align::left>
struct SELL : public has_val_t<T>,
              public has_loc_t<L>,
              public has_idx_t<I>,
              public has_srl_t<S>,
              public model::is_operator<SELL<w, T, L, I, S, align>> {
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  using typename has_idx_t<I>::idx_t;
  using typename has_srl_t<S>::srl_t;

  attribute attr;
  std::array<idx_t, 2> shape;
  vector<val_t, loc_t> val;
  vector<idx_t, loc_t> col;
  vector<srl_t, loc_t> sptr;

  template <typename T2>
  SELL(const CSR<T2, host, I, S> &in)
      : SELL(in.attr, in.shape, impl::csr_to_sell<T2, I, S, align>(in, w)) {}
  SELL(const SELL &) = default;
  template <typename T2, class L2>
  SELL(const SELL<w, T2, L2, I, S, align> &in)
      : attr(in.attr), shape(in.shape), val(in.val), col(in.col),
        sptr(in.sptr) {}

private:
  template <typename T2>
  SELL(const attribute &attr, const std::array<idx_t, 2> &shape,
      const std::tuple<vector<T2, host>, vector<idx_t, host>,
          vector<srl_t, host>> &tt)
      : attr(attr), shape(shape), val(std::get<0>(tt)), col(std::get<1>(tt)),
        sptr(std::get<2>(tt)) {}

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::sell_apply_impl<loc_t>(
        nrows_impl(), w, val.raw(), col.raw(), sptr.raw(), in.raw(), out.raw());
  }
  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::sell_residual_impl<loc_t>(nrows_impl(), w, val.raw(), col.raw(),
        sptr.raw(), rhs.raw(), in.raw(), out.raw());
  }
  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }

  friend struct model::is_operator<SELL<w, T, L, I, S, align>>;
};

template <typename T, class L, typename I = index_t, typename S = serial_t,
    impl::sell_align align = impl::sell_align::left>
using SELL32 = SELL<32, T, L, I, S, align>;

template <typename T, class L, typename I = index_t, typename S = serial_t,
    impl::sell_align align = impl::sell_align::left>
using cuSELL32 = cuSELL<32, T, L, I, S, align>;

template <uint64_t s, uint16_t w, typename T, class L, typename I, typename S,
    impl::sell_align align>
struct SIGMA<s, cuSELL<w, T, L, I, S, align>>
    : cuSELL<w, T, L, I, S, align>,
      public model::is_operator<SIGMA<s, cuSELL<w, T, L, I, S, align>>> {
  using Base = cuSELL<w, T, L, I, S, align>;
  using Base::_nrows;
  using Base::col;
  using Base::ncols;
  using Base::nrows;
  using Base::shape;
  using Base::sptr;
  using Base::val;
  using typename Base::idx_t;
  using typename Base::loc_t;
  using typename Base::val_t;
  using model::is_operator<SIGMA<s, Base>>::apply;
  using model::is_operator<SIGMA<s, Base>>::residual;

  using sig_t = impl::sigma_uint_t<s>;
  vector<sig_t, L> p;
  template <typename T2>
  SIGMA(const CSR<T2, host, I, S> &in) : SIGMA(impl::sigma<s>(in)) {}
  SIGMA(const SIGMA &) = default;
  template <typename T2, class L2>
  SIGMA(const SIGMA<s, cuSELL<w, T2, L2, I, S, align>> &in)
      : cuSELL<w, T, L, I, S, align>(in), p(in.p) {}

private:
  template <typename T2>
  SIGMA(const std::tuple<vector<sig_t, host>, CSR<T2, host, I, S>> &tt)
      : cuSELL<w, T, L, I, S, align>(std::get<1>(tt)), p(std::get<0>(tt)) {}

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::cusell_sigma_apply_impl<loc_t>(_nrows, nrows(), w, s, p.raw(),
        val.raw(), col.raw(), sptr.raw(), in.raw(), out.raw());
  }
  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::cusell_sigma_residual_impl<loc_t>(_nrows, nrows(), w, s, p.raw(),
        val.raw(), col.raw(), sptr.raw(), rhs.raw(), in.raw(), out.raw());
  }

  friend struct model::is_operator<SIGMA<s, cuSELL<w, T, L, I, S, align>>>;
};

template <uint64_t s, uint16_t w, typename T, class L, typename I, typename S,
    impl::sell_align align>
struct SIGMA<s, SELL<w, T, L, I, S, align>>
    : SELL<w, T, L, I, S, align>,
      public model::is_operator<SIGMA<s, SELL<w, T, L, I, S, align>>> {
  using Base = SELL<w, T, L, I, S, align>;
  using Base::col;
  using Base::ncols;
  using Base::nrows;
  using Base::shape;
  using Base::sptr;
  using Base::val;
  using typename Base::idx_t;
  using typename Base::loc_t;
  using typename Base::val_t;
  using model::is_operator<SIGMA<s, Base>>::apply;
  using model::is_operator<SIGMA<s, Base>>::residual;

  using sig_t = impl::sigma_uint_t<s>;
  vector<sig_t, L> p;
  template <typename T2>
  SIGMA(const CSR<T2, host, I, S> &in) : SIGMA(impl::sigma<s>(in)) {}
  SIGMA(const SIGMA &) = default;
  template <typename T2, class L2>
  SIGMA(const SIGMA<s, SELL<w, T2, L2, I, S, align>> &in)
      : SELL<w, T, L, I, S, align>(in), p(in.p) {}

private:
  template <typename T2>
  SIGMA(const std::tuple<vector<sig_t, host>, CSR<T2, host, I, S>> &tt)
      : SELL<w, T, L, I, S, align>(std::get<1>(tt)), p(std::get<0>(tt)) {}

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::sell_sigma_apply_impl<loc_t>(nrows(), w, s, p.raw(), val.raw(),
        col.raw(), sptr.raw(), in.raw(), out.raw());
  }
  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::sell_sigma_residual_impl<loc_t>(nrows(), w, s, p.raw(), val.raw(),
        col.raw(), sptr.raw(), rhs.raw(), in.raw(), out.raw());
  }

  friend struct model::is_operator<SIGMA<s, SELL<w, T, L, I, S, align>>>;
};

namespace impl {

template <typename T, typename I, typename S, impl::sell_align align>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> csr_to_sell(
    const CSR<T, host, I, S> &in, uint16_t w) {
  using val_t = T;
  using idx_t = I;
  using srl_t = S;
  auto nrows = in.nrows();
  auto n_slice = (nrows + w - 1) / w;
  srl_t nnz = 0;
  auto _sptr = vector<srl_t, host>(n_slice + 1);
  _sptr[0] = 0;
  for (idx_t i = 0; i < nrows; i += w) {
    auto t_chunk = (nrows % w != 0 && i / w == n_slice - 1) ? nrows % w : w;
    srl_t row_max = 0;
    for (decltype(t_chunk) j = 0; j < t_chunk; ++j) {
      auto range = in.rptr[i + j + 1] - in.rptr[i + j];
      row_max = std::max(range, row_max);
    }
    nnz += row_max * t_chunk;
    _sptr[i / w + 1] = _sptr[i / w] + row_max;
  }
  auto _val = vector<val_t, host>(nnz);
  auto _col = vector<idx_t, host>(nnz);
  auto assign = [&](auto pos, auto val, auto col) mutable {
    _val[pos] = val;
    _col[pos] = col;
  };
  for (idx_t i = 0; i < nrows; i += w) {
    auto s_id = i / w;
    auto t_chunk = (nrows % w != 0 && s_id == n_slice - 1) ? nrows % w : w;
    auto row_max = _sptr[s_id + 1] - _sptr[s_id];
    if constexpr (align == impl::sell_align::left) {
      idx_t off = _sptr[s_id] * w;
      for (decltype(t_chunk) j = 0; j < t_chunk; ++j) {
        idx_t cnt = 0;
        for (auto k = in.rptr[i + j]; k < in.rptr[i + j + 1]; ++k, ++cnt)
          assign(off + cnt * t_chunk + j, in.val[k], in.col[k]);
        for (auto k = cnt; k < row_max; k++)
          assign(off + k * t_chunk + j, 0,
              (k == 0) ? 0 : _col[off + (k - 1) * t_chunk + j]);
      }
    } else {
      idx_t off = _sptr[s_id] * w;
      for (decltype(t_chunk) j = 0; j < t_chunk; ++j) {
        idx_t cnt = 0;
        for (auto k = in.rptr[i + j + 1] - 1; k >= in.rptr[i + j]; --k, ++cnt)
          assign(off + (row_max - 1 - cnt) * t_chunk + j, in.val[k], in.col[k]);
        for (auto k = cnt; k < row_max; k++)
          assign(off + (row_max - 1 - k) * t_chunk + j, 0.,
              (k == 0) ? 0 : _col[off + (row_max - k) * t_chunk + j]);
      }
    }
  }
  return {_val, _col, _sptr};
}

template <typename T, typename I, typename S, impl::sell_align align>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> csr_to_cusell(
    const CSR<T, host, I, S> &in, uint16_t w) {
  using val_t = T;
  using idx_t = I;
  using srl_t = S;
  auto nrows = in.nrows();
  auto n_slice = (nrows + w - 1) / w;
  srl_t nnz = 0;
  auto sptr = vector<srl_t, host>(n_slice + 1);
  sptr[0] = 0;

#pragma omp parallel for
  for (idx_t i = 0; i < nrows; i += w) {
    auto t_w = (nrows % w != 0 && i / w == n_slice - 1) ? nrows % w : w;
    srl_t row_max = 0;
    for (uint16_t j = 0; j < t_w; ++j) {
      auto range = in.rptr[i + j + 1] - in.rptr[i + j];
      row_max = std::max(range, row_max);
    }
    sptr[i / w + 1] = row_max * w;
  }

#pragma omp parallel for reduction(inscan, + : nnz)
  for (idx_t i = 0; i < n_slice + 1; i++) {
    nnz += sptr[i];
#pragma omp scan inclusive(nnz)
    sptr[i] = nnz;
  }

  auto val = vector<val_t, host>(nnz);
  auto col = vector<idx_t, host>(nnz).fill(-1);
  auto assign = [&](auto pos, auto v, auto c) mutable {
    val[pos] = v;
    col[pos] = c;
  };

  // #pragma omp parallel for
  for (idx_t i = 0; i < nrows; i += w) {
    auto s_id = i / w;
    auto t_w = (nrows % w != 0 && s_id == n_slice - 1) ? nrows % w : w;
    auto row_max = (sptr[s_id + 1] - sptr[s_id]) / w;
    if constexpr (align == impl::sell_align::left) {
      idx_t offset = sptr[s_id];
      for (uint16_t j = 0; j < t_w; ++j) {
        idx_t cnt = 0;
        for (auto k = in.rptr[i + j]; k < in.rptr[i + j + 1]; ++k, ++cnt)
          assign(offset + cnt * w + j, in.val[k], in.col[k]);
      }
    } else {
      idx_t offset = sptr[s_id];
      for (uint16_t j = 0; j < t_w; ++j) {
        idx_t cnt = 0;
        for (auto k = in.rptr[i + j + 1] - 1; k >= in.rptr[i + j]; --k, ++cnt)
          assign(offset + (row_max - 1 - cnt) * w + j, in.val[k], in.col[k]);
      }
    }
  }

  return std::make_tuple(val, col, sptr);
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void cusell_apply_impl(const idx_t _nrows, const idx_t nrows, const uint16_t w,
    const val_t *val, const idx_t *col, const srl_t *sptr, const in_t *in,
    out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(_nrows, [=] SENK_LOC(idx_t i) {
    idx_t sid = i / w;
    auto myid = i % w;
    auto s = sptr[sid];
    auto size = (sptr[sid + 1] - s) / w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = s + myid;
    for (idx_t j = 0; j < size; j++) {
      auto c = col[offset];
      auto v = val[offset];
      if (c != -1)
        tmp += v * in[c];
      offset += w;
    }
    if (i < nrows)
      out[i] = static_cast<out_t>(tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void cusell_residual_impl(const idx_t _nrows, const idx_t nrows,
    const uint16_t w, const val_t *val, const idx_t *col, const srl_t *sptr,
    const rhs_t *rhs, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(_nrows, [=] SENK_LOC(idx_t i) {
    idx_t sid = i / w;
    auto myid = i % w;
    auto s = sptr[sid];
    auto size = (sptr[sid + 1] - s) / w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = s + myid;
    for (idx_t j = 0; j < size; j++) {
      auto c = col[offset];
      auto v = val[offset];
      if (c != -1)
        tmp += v * in[c];
      offset += w;
    }
    if (i < nrows)
      out[i] = static_cast<out_t>(rhs[i] - tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void sell_apply_impl(const idx_t nrows, const uint16_t w, const val_t *val,
    const idx_t *col, const srl_t *sptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) {
    idx_t sid = i / w;
    auto myid = i % w;
    auto s = sptr[sid];
    auto size = sptr[sid + 1] - s;
    auto t_chunk = (sid != nrows / w) ? w : nrows % w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = s * w + myid;
    for (idx_t j = 0; j < size; j++) {
      tmp += val[offset] * in[col[offset]];
      offset += t_chunk;
    }
    out[i] = static_cast<out_t>(tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void sell_residual_impl(const idx_t nrows, const uint16_t w, const val_t *val,
    const idx_t *col, const srl_t *sptr, const rhs_t *rhs, const in_t *in,
    out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) {
    idx_t sid = i / w;
    auto myid = i % w;
    auto s = sptr[sid];
    auto size = sptr[sid + 1] - s;
    auto t_chunk = (sid != nrows / w) ? w : nrows % w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = s * w + myid;
    for (idx_t j = 0; j < size; j++) {
      tmp += val[offset] * in[col[offset]];
      offset += t_chunk;
    }
    out[i] = static_cast<out_t>(rhs[i] - tmp);
  });
}

#if 0
template <typename loc_t, typename val_t, typename idx_t, typename in_t,
    typename out_t>
void fsell_apply_impl([[maybe_unused]] const idx_t nrows, const idx_t nwarps,
    const uint16_t w, const val_t *val, const idx_t *col, const idx_t *sptr,
    const idx_t *sh, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nwarps * w, [=] SENK_LOC(size_t i) {
    auto wid = i / w;
    auto slice = sh[wid + 1] - sh[wid];
    auto myid = i % w;
    auto s = sptr[wid];
    auto width = (sptr[wid + 1] - s) / slice;
    auto tmp = static_cast<tmp_t>(0.);
    auto off = s + myid;

    if (slice == 32) {
      for (idx_t j = 0; j < width; j++, off += slice)
        tmp += val[off] * in[col[off]];
      out[sh[wid] + myid] = static_cast<out_t>(tmp);
    } else if (slice == 16) {
      auto id = myid / 16;
      for (idx_t j = id; j < width; j += 2, off += 2 * slice)
        tmp += val[off] * in[col[off]];
      tmp += __shfl_down_sync(0xffffffff, tmp, 16, 32);
      if (id == 0)
        out[sh[wid] + myid] = static_cast<out_t>(tmp);
    } else if (slice == 8) {
      auto id = myid / 8;
      for (idx_t j = id; j < width; j += 4, off += 4 * slice)
        tmp += val[off] * in[col[off]];
      tmp += __shfl_down_sync(0xffffffff, tmp, 16, 32);
      tmp += __shfl_down_sync(0xffffffff, tmp, 8, 16);
      if (id == 0)
        out[sh[wid] + myid] = static_cast<out_t>(tmp);
    } else if (slice == 4) {
      auto id = myid / 4;
      for (idx_t j = id; j < width; j += 8, off += 8 * slice)
        tmp += val[off] * in[col[off]];
      tmp += __shfl_down_sync(0xffffffff, tmp, 16, 32);
      tmp += __shfl_down_sync(0xffffffff, tmp, 8, 16);
      tmp += __shfl_down_sync(0xffffffff, tmp, 4, 8);
      if (id == 0)
        out[sh[wid] + myid] = static_cast<out_t>(tmp);
    } else if (slice == 2) {
      auto id = myid / 2;
      for (idx_t j = id; j < width; j += 16, off += 16 * slice)
        tmp += val[off] * in[col[off]];
      tmp += __shfl_down_sync(0xffffffff, tmp, 16, 32);
      tmp += __shfl_down_sync(0xffffffff, tmp, 8, 16);
      tmp += __shfl_down_sync(0xffffffff, tmp, 4, 8);
      tmp += __shfl_down_sync(0xffffffff, tmp, 2, 4);
      if (id == 0)
        out[sh[wid] + myid] = static_cast<out_t>(tmp);
    } else if (slice == 1) {
      auto id = myid;
      for (idx_t j = id; j < width; j += 32, off += 32 * slice)
        tmp += val[off] * in[col[off]];
      tmp += __shfl_down_sync(0xffffffff, tmp, 16, 32);
      tmp += __shfl_down_sync(0xffffffff, tmp, 8, 16);
      tmp += __shfl_down_sync(0xffffffff, tmp, 4, 8);
      tmp += __shfl_down_sync(0xffffffff, tmp, 2, 4);
      tmp += __shfl_down_sync(0xffffffff, tmp, 1, 2);
      if (id == 0)
        out[sh[wid] + myid] = static_cast<out_t>(tmp);
    } else {
      for (idx_t j = 0; j < width; j++, off += slice)
        tmp += val[off] * in[col[off]];
      out[sh[wid] + myid] = static_cast<out_t>(tmp);
    }
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename rhs_t,
    typename in_t, typename out_t>
void fsell_residual_impl(const idx_t nrows, const uint16_t w, const val_t *val,
    const idx_t *col, const idx_t *sptr, const idx_t *sh, const rhs_t *rhs,
    const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    idx_t sid = i / w;
    auto myid = i % w;
    auto s = sptr[sid];
    auto size = sptr[sid + 1] - s;
    auto t_chunk = (sid != nrows / w) ? w : nrows % w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = s * w + myid;
    for (idx_t j = 0; j < size; j++) {
      tmp += val[offset] * in[col[offset]];
      offset += t_chunk;
    }
    out[i] = static_cast<out_t>(rhs[i] - tmp);
  });
}
#endif

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename in_t, typename out_t>
void cusell_sigma_apply_impl(const idx_t _nrows, const idx_t nrows,
    const uint16_t w, const uint64_t s, const sig_t *p, const val_t *val,
    const idx_t *col, const srl_t *sptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(_nrows, [=] SENK_LOC(idx_t i) {
    auto sig = i / s * s;
    idx_t row = (i < nrows) ? sig + p[i] : 0;
    idx_t sid = i / w;
    auto myid = i % w;
    auto s = sptr[sid];
    auto size = (sptr[sid + 1] - s) / w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = s + myid;
    for (idx_t j = 0; j < size; j++) {
      auto c = col[offset];
      auto v = val[offset];
      if (c != -1)
        tmp += v * in[c];
      offset += w;
    }
    if (i < nrows)
      out[row] = static_cast<out_t>(tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename rhs_t, typename in_t, typename out_t>
void cusell_sigma_residual_impl(const idx_t _nrows, const idx_t nrows,
    const uint16_t w, const uint64_t s, const sig_t *p, const val_t *val,
    const idx_t *col, const srl_t *sptr, const rhs_t *rhs, const in_t *in,
    out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(_nrows, [=] SENK_LOC(idx_t i) {
    auto sig = i / s * s;
    idx_t row = (i < nrows) ? sig + p[i] : 0;
    idx_t sid = i / w;
    auto myid = i % w;
    auto s = sptr[sid];
    auto size = (sptr[sid + 1] - s) / w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = s + myid;
    for (idx_t j = 0; j < size; j++) {
      auto c = col[offset];
      auto v = val[offset];
      if (c != -1)
        tmp += v * in[c];
      offset += w;
    }
    if (i < nrows)
      out[row] = static_cast<out_t>(rhs[row] - tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename in_t, typename out_t>
void sell_sigma_apply_impl(const idx_t nrows, const uint16_t w,
    const uint64_t s, const sig_t *p, const val_t *val, const idx_t *col,
    const srl_t *sptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    auto sig = i / s * s;
    auto row = sig + p[i];
    idx_t sid = i / w;
    auto myid = i % w;
    auto strt = sptr[sid];
    auto size = sptr[sid + 1] - strt;
    auto t_chunk = (sid != nrows / w) ? w : nrows % w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = strt * w + myid;
    for (idx_t j = 0; j < size; j++) {
      tmp += val[offset] * in[col[offset]];
      offset += t_chunk;
    }
    out[row] = static_cast<out_t>(tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename rhs_t, typename in_t, typename out_t>
void sell_sigma_residual_impl(const idx_t nrows, const uint16_t w,
    const uint64_t s, const sig_t *p, const val_t *val, const idx_t *col,
    const srl_t *sptr, const rhs_t *rhs, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    auto sig = i / s * s;
    auto row = sig + p[i];
    idx_t sid = i / w;
    auto myid = i % w;
    auto strt = sptr[sid];
    auto size = sptr[sid + 1] - strt;
    auto t_chunk = (sid != nrows / w) ? w : nrows % w;
    auto tmp = static_cast<tmp_t>(0.);
    auto offset = strt * w + myid;
    for (idx_t j = 0; j < size; j++) {
      tmp += val[offset] * in[col[offset]];
      offset += t_chunk;
    }
    out[row] = static_cast<out_t>(rhs[row] - tmp);
  });
}

} // namespace impl

} // namespace senk

#endif