#ifndef SENK_MATRIX_CSR_HPP
#define SENK_MATRIX_CSR_HPP

#include "senk/core/tools.hpp"

#include "senk/core/io.hpp"
#include "senk/core/tensor.hpp"
#include "senk/matrix/base.hpp"
#include "senk/matrix/sigma.hpp"
#include "senk/models.hpp"

namespace senk {

template <uint64_t S, class B>
struct SIGMA;
template <uint16_t bnl, uint16_t bnw, typename T, class L, typename I,
    typename S>
struct BSR;

namespace impl {

template <uint16_t bnl, uint16_t bnw, typename T, typename I, typename S>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> bsr_to_csr(
    const BSR<bnl, bnw, T, host, I, S> &B);

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void csr_apply_impl(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const in_t *in, out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void csr_residual_impl(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const rhs_t *rhs, const in_t *in, out_t *out);

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename in_t, typename out_t>
void csr_sigma_apply_impl(const idx_t nrows, const uint64_t s, const sig_t *p,
    const val_t *val, const idx_t *col, const srl_t *rptr, const in_t *in,
    out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename rhs_t, typename in_t, typename out_t>
void csr_sigma_residual_impl(const idx_t nrows, const uint64_t s,
    const sig_t *p, const val_t *val, const idx_t *col, const srl_t *rptr,
    const rhs_t *rhs, const in_t *in, out_t *out);

} // namespace impl

template <typename T, class L, typename I = index_t, typename S = serial_t>
struct CSR : public has_val_t<T>,
             public has_loc_t<L>,
             public has_idx_t<I>,
             public has_srl_t<S>,
             public model::is_operator<CSR<T, L, I, S>> {
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  using typename has_idx_t<I>::idx_t;
  using typename has_srl_t<S>::srl_t;

  attribute attr;
  std::array<idx_t, 2> shape;
  vector<val_t, loc_t> val;
  vector<idx_t, loc_t> col;
  vector<srl_t, loc_t> rptr;

  CSR(const std::string &filename) : CSR(io::readmm_as_csr<double>(filename)) {
    static_assert(std::is_same_v<loc_t, host>);
  }
  CSR(std::array<idx_t, 2> shape, srl_t nnz, attribute attr = attribute{})
      : attr(attr), shape(shape), val(nnz), col(nnz), rptr(shape[0] + 1) {}
  CSR(const CSR &) = default;
  template <typename T2, class L2, typename I2, typename S2>
  CSR(const CSR<T2, L2, I2, S2> &in)
      : attr(in.attr), shape(in.shape), val(in.val), col(in.col),
        rptr(in.rptr) {}
  template <uint16_t bnl, uint16_t bnw>
  CSR(const BSR<bnl, bnw, T, L, I, S> &bsr)
      : CSR(bsr.attr, bsr.shape, impl::bsr_to_csr(bsr)) {}

  template <typename T2, class L2>
  CSR &copy(const CSR<T2, L2, I, S> &in);
  CSR &inverse_last();
  CSR &scaling();
  template <typename T2>
  CSR &scaling(vector<T2, L> &vec);
  CSR &row_scaling();
  CSR duplicate() const;
  CSR duplicate_val() const;
  CSR duplicate_rotate180() const;
  CSR duplicate_block(size_t b_num) const;
  std::tuple<CSR, CSR> split_l1_du() const;
  std::tuple<CSR, CSR> split_l_du() const;
  std::tuple<CSR, CSR> split_ld_u() const;
  std::tuple<vector<T, L>, CSR> split_d_lu() const;

private:
  template <typename T2>
  CSR(const io::CSR<T2> &d)
      : shape({static_cast<idx_t>(d.nrows), static_cast<idx_t>(d.ncols)}),
        val(d.nnzs), col(d.nnzs), rptr(d.nrows + 1) {
    if (d.is_symmetric)
      attr.set_flag(impl::flags::is_symmetric);
    else
      attr.reset_flag(impl::flags::is_symmetric);

#pragma omp parallel for
    for (size_t i = 0; i < val.shape(0); ++i)
      val[i] = static_cast<val_t>(d.val[i]);
#pragma omp parallel for
    for (size_t i = 0; i < col.shape(0); ++i)
      col[i] = static_cast<idx_t>(d.col[i]);
#pragma omp parallel for
    for (size_t i = 0; i < rptr.shape(0); ++i)
      rptr[i] = static_cast<idx_t>(d.rptr[i]);
    delete[] d.val;
    delete[] d.col;
    delete[] d.rptr;
  }
  explicit CSR(const attribute &attr, const std::array<I, 2> &shape,
      std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> &&t)
      : attr(attr), shape(shape), val(std::get<0>(t)), col(std::get<1>(t)),
        rptr(std::get<2>(t)) {}

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::csr_apply_impl<loc_t>(
        nrows_impl(), val.raw(), col.raw(), rptr.raw(), in.raw(), out.raw());
  }
  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::csr_residual_impl<loc_t>(nrows_impl(), val.raw(), col.raw(),
        rptr.raw(), rhs.raw(), in.raw(), out.raw());
  }
  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }

  friend struct model::is_operator<CSR<T, L, I, S>>;
};

template <uint64_t s, typename T, class L, typename I, typename S>
struct SIGMA<s, CSR<T, L, I, S>>
    : CSR<T, L, I, S>, public model::is_operator<SIGMA<s, CSR<T, L, I, S>>> {
  using Base = CSR<T, L, I, S>;
  using Base::col;
  using Base::ncols;
  using Base::nrows;
  using Base::rptr;
  using Base::val;
  using typename Base::idx_t;
  using typename Base::loc_t;
  using typename Base::srl_t;
  using typename Base::val_t;
  using model::is_operator<SIGMA<s, Base>>::apply;
  using model::is_operator<SIGMA<s, Base>>::residual;

  using sig_t = impl::sigma_uint_t<s>;
  vector<sig_t, L> p;
  template <typename T2>
  SIGMA(const CSR<T2, host, I, S> &in) : SIGMA(impl::sigma<s>(in)) {}
  SIGMA(const SIGMA &) = default;
  template <typename T2, class L2>
  SIGMA(const SIGMA<s, CSR<T2, L2, I, S>> &in) : CSR<T, L, I, S>(in), p(in.p) {}

private:
  template <typename T2>
  SIGMA(const std::tuple<vector<sig_t, host>, CSR<T2, host, I, S>> &tt)
      : CSR<T, L, I, S>(std::get<1>(tt)), p(std::get<0>(tt)) {}

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::csr_sigma_apply_impl<loc_t>(nrows(), s, p.raw(), val.raw(), col.raw(),
        rptr.raw(), in.raw(), out.raw());
  }
  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    impl::csr_sigma_residual_impl<loc_t>(nrows(), s, p.raw(), val.raw(),
        col.raw(), rptr.raw(), rhs.raw(), in.raw(), out.raw());
  }

  friend struct model::is_operator<SIGMA<s, CSR<T, L, I, S>>>;
};

namespace impl {

template <uint16_t bnl, uint16_t bnw, typename T, typename I, typename S>
std::tuple<vector<T, host>, vector<I, host>, vector<S, host>> bsr_to_csr(
    const BSR<bnl, bnw, T, host, I, S> &B) {
  using idx_t = I;
  using val_t = T;
  using srl_t = S;
  auto nrows = B.nrows();
  auto bnrows = (nrows + bnl - 1) / bnl;

  auto num_block = B.rptr[bnrows];
  const auto area = bnl * bnw;
  auto _val = vector<val_t, host>(num_block * bnl * bnw);
  auto _idx = vector<idx_t, host>(num_block * bnl * bnw);
  auto rptr = vector<srl_t, host>(B.nrows() + 1);

  rptr[0] = 0;
  for (idx_t bi = 0; bi < bnrows; bi++) {
    auto width = (B.rptr[bi + 1] - B.rptr[bi]) * bnw;
    for (idx_t ci = 0; ci < bnl; ci++) {
      auto i = bi * bnl + ci;
      if (B.nrows() <= i)
        continue;
      rptr[i + 1] = rptr[i] + width;
    }

    idx_t cnt[bnl] = {0};
    for (auto bj = B.rptr[bi]; bj < B.rptr[bi + 1]; bj++) {
      for (idx_t ci = 0; ci < bnl; ci++) {
        auto i = bi * bnl + ci;
        if (B.nrows() <= i)
          continue;
        for (idx_t cj = 0; cj < bnw; cj++) {
          auto j = B.col[bj] * bnw + cj;
          if (B.ncols() <= j)
            continue;
          _idx[rptr[i] + cnt[ci]] = j;
          _val[rptr[i] + cnt[ci]] = B.val[bj * area + cj * bnl + ci];
          cnt[ci]++;
        }
      }
    }
  }

  auto nnz = rptr[nrows];
  auto val = vector<val_t, host>(nnz).copy(_val);
  auto idx = vector<idx_t, host>(nnz).copy(_idx);
  return {val, idx, rptr};
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void csr_apply_impl(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    auto tmp = static_cast<tmp_t>(0.);
    for (auto j = rptr[i]; j < rptr[i + 1]; j++) {
      tmp += val[j] * in[col[j]];
    }
    out[i] = static_cast<out_t>(tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void csr_residual_impl(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const rhs_t *rhs, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    auto tmp = static_cast<tmp_t>(0.);
    for (auto j = rptr[i]; j < rptr[i + 1]; j++) {
      tmp += val[j] * in[col[j]];
    }
    out[i] = static_cast<out_t>(rhs[i] - tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename in_t, typename out_t>
void csr_sigma_apply_impl(const idx_t nrows, const uint64_t s, const sig_t *p,
    const val_t *val, const idx_t *col, const srl_t *rptr, const in_t *in,
    out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    auto sig = i / s * s;
    auto row = sig + p[i];
    auto tmp = static_cast<tmp_t>(0.);
    for (auto j = rptr[i]; j < rptr[i + 1]; j++) {
      tmp += val[j] * in[col[j]];
    }
    out[row] = static_cast<out_t>(tmp);
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename sig_t, typename rhs_t, typename in_t, typename out_t>
void csr_sigma_residual_impl(const idx_t nrows, const uint64_t s,
    const sig_t *p, const val_t *val, const idx_t *col, const srl_t *rptr,
    const rhs_t *rhs, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    auto sig = i / s * s;
    auto row = sig + p[i];
    auto tmp = static_cast<tmp_t>(0.);
    for (auto j = rptr[i]; j < rptr[i + 1]; j++) {
      tmp += val[j] * in[col[j]];
    }
    out[row] = static_cast<out_t>(rhs[row] - tmp);
  });
}

} // namespace impl

template <typename T, class L, typename I, typename S>
template <typename T2, class L2>
CSR<T, L, I, S> &CSR<T, L, I, S>::copy(const CSR<T2, L2, I, S> &in) {
  attr = in.attr;
  shape = in.shape;
  val.copy(in.val);
  col.copy(in.col);
  rptr.copy(in.rptr);
  return *this;
}

template <typename T, class L, typename I, typename S>
CSR<T, L, I, S> &CSR<T, L, I, S>::inverse_last() {
  auto v = val.raw();
  auto r = rptr.raw();
  kernel<L>::parallel(nrows_impl(), [=](I i) mutable {
    v[r[i + 1] - 1] = static_cast<val_t>(1. / v[r[i + 1] - 1]);
  });
  return *this;
}

template <typename T, class L, typename I, typename S>
CSR<T, L, I, S> &CSR<T, L, I, S>::scaling() {
  auto diag = vector<T, L>(nrows_impl()).fill(1);
  auto d = diag.raw();
  auto v = val.raw();
  auto c = col.raw();
  auto r = rptr.raw();
  kernel<L>::parallel(nrows_impl(), [=] SENK_LOC(size_t i) mutable {
    for (auto j = r[i]; j < r[i + 1]; ++j)
      if (c[j] == idx_t(i) && v[j] != 0.)
        d[i] = senk::sqrt(senk::abs(v[j]));
  });
  kernel<L>::parallel(nrows_impl(), [=] SENK_LOC(size_t i) mutable {
    for (auto j = r[i]; j < r[i + 1]; ++j)
      v[j] /= d[i] * d[c[j]];
  });
  return *this;
}

template <typename T, class L, typename I, typename S>
template <typename T2>
CSR<T, L, I, S> &CSR<T, L, I, S>::scaling(vector<T2, L> &vec) {
  auto diag = vector<T, L>(nrows_impl()).fill(1);
  auto d = diag.raw();
  auto v = val.raw();
  auto c = col.raw();
  auto r = rptr.raw();
  auto ve = vec.raw();
  kernel<L>::parallel(nrows_impl(), [=] SENK_LOC(size_t i) mutable {
    for (auto j = r[i]; j < r[i + 1]; ++j)
      if (c[j] == idx_t(i) && v[j] != 0.)
        d[i] = senk::sqrt(senk::abs(v[j]));
  });
  kernel<L>::parallel(nrows_impl(), [=] SENK_LOC(size_t i) mutable {
    for (auto j = r[i]; j < r[i + 1]; ++j)
      v[j] /= d[i] * d[c[j]];
    ve[i] /= d[i];
  });
  return *this;
}

template <typename T, class L, typename I, typename S>
CSR<T, L, I, S> &CSR<T, L, I, S>::row_scaling() {
  auto diag = vector<T, L>(nrows_impl()).fill(1);
  auto d = diag.raw();
  auto v = val.raw();
  auto r = rptr.raw();
  kernel<L>::parallel(nrows_impl(), [=] SENK_LOC(size_t i) mutable {
    for (auto j = r[i]; j < r[i + 1]; ++j)
      d[i] += senk::abs(v[j]);
    for (auto j = r[i]; j < r[i + 1]; ++j)
      v[j] /= d[i];
  });
  return *this;
}

template <typename T, class L, typename I, typename S>
CSR<T, L, I, S> CSR<T, L, I, S>::duplicate() const {
  auto res = CSR(shape, val.shape(0), attr.duplicate());
  return res.copy(*this);
}

template <typename T, class L, typename I, typename S>
CSR<T, L, I, S> CSR<T, L, I, S>::duplicate_val() const {
  auto res = CSR(*this);
  auto vec = vector<T, L>(val.shape(0)).copy(val);
  res.val = vec;
  return res;
}

template <typename T, class L, typename I, typename S>
CSR<T, L, I, S> CSR<T, L, I, S>::duplicate_rotate180() const {
  auto res = CSR(shape, val.shape(0), attr.duplicate_rotate180());
  auto rrp = res.rptr.raw();
  auto rv = res.val.raw();
  auto rc = res.col.raw();

  auto rp = rptr.raw();
  auto v = val.raw();
  auto c = col.raw();
  auto nrows = shape[0];
  auto nnz = val.shape(0);
  kernel<L>::parallel(
      nrows + 1, [=](idx_t i) mutable { rrp[i] = nnz - rp[nrows - i]; });
  kernel<L>::parallel(nnz, [=](idx_t i) mutable {
    rv[i] = v[nnz - 1 - i];
    rc[i] = nrows - 1 - c[nnz - 1 - i];
  });
  // res.spmat::copy(*this);
  // if (is_block_diagonal()) res.block = block.clone().reverse();
  // if (is_colored()) res.color = color.clone().reverse();
  return res;
}

template <typename T, class L, typename I, typename S>
CSR<T, L, I, S> CSR<T, L, I, S>::duplicate_block(size_t b_num) const {
  auto block_size = (this->nrows() + b_num - 1) / b_num;
  auto b_size = vector<int, host>(b_num + 1);
  b_size[0] = 0;
  for (size_t i = 0; i < b_num; ++i) {
    b_size[i + 1] = (i != b_num - 1) ? b_size[i] + block_size : this->nrows();
  }
  int nnz = 0;
  for (size_t bi = 0; bi < b_num; bi++) {
    auto left = b_size[bi];
    auto right = b_size[bi + 1];
    for (auto i = left; i < right; i++) {
      for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
        if (left <= col[j] && col[j] < right)
          nnz++;
      }
    }
  }
  auto res = CSR(shape, nnz, attr.duplicate());
  nnz = 0;
  res.rptr[0] = nnz;
  for (size_t bi = 0; bi < b_num; bi++) {
    auto left = b_size[bi];
    auto right = b_size[bi + 1];
    for (int i = left; i < right; i++) {
      for (int j = rptr[i]; j < rptr[i + 1]; ++j) {
        if (left <= col[j] && col[j] < right) {
          res.val[nnz] = val[j];
          res.col[nnz++] = col[j];
        }
      }
      res.rptr[i + 1] = nnz;
    }
  }

  res.attr.set_flag(impl::flags::is_partitioned);
  res.attr.segm = b_size;
  return res;
}

template <typename T, class L, typename I, typename S>
std::tuple<CSR<T, L, I, S>, CSR<T, L, I, S>>
CSR<T, L, I, S>::split_l1_du() const {
  static_assert(std::is_same_v<host, L>, "split must be on host");
  idx_t l_nnz = 0, u_nnz = 0;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      l_nnz = (col[j] <= i) ? l_nnz + 1 : l_nnz;
      u_nnz = (col[j] >= i) ? u_nnz + 1 : u_nnz;
    }
  }
  auto l = CSR(shape, l_nnz, attr.duplicate());
  l.attr.set_ld(impl::flags::is_lower1);
  auto u = CSR(shape, u_nnz, attr.duplicate());
  u.attr.set_ld(impl::flags::is_upper);

  l_nnz = u_nnz = 0;
  l.rptr[0] = l_nnz;
  u.rptr[0] = u_nnz;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      if (col[j] <= i) {
        l.val[l_nnz] = (col[j] < i) ? val[j] : 1.0;
        l.col[l_nnz++] = col[j];
      }
      if (col[j] >= i) {
        u.val[u_nnz] = val[j];
        u.col[u_nnz++] = col[j];
      }
    }
    l.rptr[i + 1] = l_nnz;
    u.rptr[i + 1] = u_nnz;
  }
  return {l, u};
}

template <typename T, class L, typename I, typename S>
std::tuple<CSR<T, L, I, S>, CSR<T, L, I, S>>
CSR<T, L, I, S>::split_l_du() const {
  static_assert(std::is_same_v<host, L>, "split must be on host");
  idx_t l_nnz = 0, u_nnz = 0;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      l_nnz = (col[j] < i) ? l_nnz + 1 : l_nnz;
      u_nnz = (col[j] >= i) ? u_nnz + 1 : u_nnz;
    }
  }
  auto l = CSR(shape, l_nnz, attr.duplicate());
  l.attr.set_ld(impl::flags::is_lower);
  auto u = CSR(shape, u_nnz, attr.duplicate());
  u.attr.set_ld(impl::flags::is_upper);

  l_nnz = u_nnz = 0;
  l.rptr[0] = l_nnz;
  u.rptr[0] = u_nnz;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      if (col[j] < i) {
        l.val[l_nnz] = val[j];
        l.col[l_nnz++] = col[j];
      }
      if (col[j] >= i) {
        u.val[u_nnz] = val[j];
        u.col[u_nnz++] = col[j];
      }
    }
    l.rptr[i + 1] = l_nnz;
    u.rptr[i + 1] = u_nnz;
  }
  return {l, u};
}

template <typename T, class L, typename I, typename S>
std::tuple<CSR<T, L, I, S>, CSR<T, L, I, S>>
CSR<T, L, I, S>::split_ld_u() const {
  static_assert(std::is_same_v<host, L>, "split must be on host");
  idx_t l_nnz = 0, u_nnz = 0;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      l_nnz = (col[j] <= i) ? l_nnz + 1 : l_nnz;
      u_nnz = (col[j] > i) ? u_nnz + 1 : u_nnz;
    }
  }
  auto l = CSR(shape, l_nnz, attr.duplicate());
  l.attr.set_ld(impl::flags::is_lower);
  auto u = CSR(shape, u_nnz, attr.duplicate());
  u.attr.set_ld(impl::flags::is_upper);

  l_nnz = u_nnz = 0;
  l.rptr[0] = l_nnz;
  u.rptr[0] = u_nnz;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      if (col[j] <= i) {
        l.val[l_nnz] = val[j];
        l.col[l_nnz++] = col[j];
      }
      if (col[j] > i) {
        u.val[u_nnz] = val[j];
        u.col[u_nnz++] = col[j];
      }
    }
    l.rptr[i + 1] = l_nnz;
    u.rptr[i + 1] = u_nnz;
  }
  return {l, u};
}

template <typename T, class L, typename I, typename S>
std::tuple<vector<T, L>, CSR<T, L, I, S>> CSR<T, L, I, S>::split_d_lu() const {
  static_assert(std::is_same_v<host, L>, "split must be on host");
  idx_t d_nnz = 0, lu_nnz = 0;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      d_nnz = (col[j] == i) ? d_nnz + 1 : d_nnz;
      lu_nnz = (col[j] != i) ? lu_nnz + 1 : lu_nnz;
    }
  }
  if (d_nnz != nrows_impl())
    printf("WARNING: There are zero diagonals");
  auto lu = CSR(shape, lu_nnz);
  auto d = vector<T, L>(d_nnz);

  lu_nnz = 0;
  lu.rptr[0] = lu_nnz;
  for (idx_t i = 0; i < nrows_impl(); i++) {
    for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
      if (col[j] != i) {
        lu.val[lu_nnz] = val[j];
        lu.col[lu_nnz++] = col[j];
      } else {
        d[i] = val[j];
      }
    }
    lu.rptr[i + 1] = lu_nnz;
  }
  return {d, lu};
}

} // namespace senk

#endif