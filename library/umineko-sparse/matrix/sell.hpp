#ifndef UMINEKO_SPARSE_MATRIX_SELL_HPP
#define UMINEKO_SPARSE_MATRIX_SELL_HPP

#include "umineko-core/tensor.hpp"
#include "umineko-sparse/matrix/base.hpp"

#include "umineko-sparse/matrix/csr.hpp"
#include "umineko-sparse/models.hpp"

namespace kmm {

namespace impl {

enum class sell_align { left, right };

template <uint16_t w, sell_align align, typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<idx_t, host>, vector<idx_t, host>>
csr_to_sell(const CSR<T2, host> &in);

template <uint16_t w, uint16_t s, sell_align align, typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<idx_t, host>, vector<idx_t, host>,
    vector<uint16_t, host>>
csr_to_sell_sigma(const CSR<T2, host> &in);

#if 0
template <uint16_t w, uint16_t s, uint16_t r, sell_align align, typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<uint16_t, host>,
    vector<uint16_t, host>, vector<uint16_t, host>, vector<idx_t, host>,
    vector<uint16_t, host>>
csr_to_sell_sigma_segment(const CSR<T2, host> &in);
#endif

} // namespace impl

template <uint16_t w, typename T, class L,
    impl::sell_align align = impl::sell_align::left>
struct SELL : spmat {
  using loc_t = L;
  using val_t = T;
  constexpr static uint16_t W = w;
  vector<T, L> val;
  vector<idx_t, L> idx;
  vector<idx_t, L> sptr;

  template <typename T2>
  explicit SELL(const CSR<T2, host> &in) : SELL(impl::csr_to_sell<w, align>(in)) {
    spmat::copy_attrs(in);
  }

  SELL(const SELL &) = default;
  template <typename T2, class L2>
  explicit SELL(const SELL<w, T2, L2> &in)
      : spmat(in), val(in.val), idx(in.idx), sptr(in.sptr) {}

  template <typename in_t, typename out_t>
  void operate(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(out.size(0), [=, *this, n = out.size(0)](idx_t i) mutable {
      auto sid = i / w;
      auto myid = i % w;
      auto s = sptr[sid];
      auto size = sptr[sid + 1] - s;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto t = decltype(std::declval<T &>() * std::declval<in_t &>()){0};
      auto offset = s * w + myid;
      for (idx_t j = 0; j < size; j++) {
        t += val[offset] * in[idx[offset]];
        offset += t_chunk;
      }
      out[i] = t;
    });
  }
  template <typename rhs_t, typename in_t, typename res_t>
  void compute_residual(const vector<rhs_t, L> &rhs, const vector<in_t, L> &in,
      vector<res_t, L> res) const {
    exec<L>::para_for(res.size(0), [=, *this, n = res.size(0)](idx_t i) mutable {
      auto sid = i / w;
      auto myid = i % w;
      auto s = sptr[sid];
      auto size = sptr[sid + 1] - s;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto t = decltype(std::declval<T &>() * std::declval<in_t &>()){0};
      auto offset = s * w + myid;
      for (idx_t j = 0; j < size; j++) {
        t += val[offset] * in[idx[offset]];
        offset += t_chunk;
      }
      res[i] = rhs[i] - t;
    });
  }

private:
  template <typename T2>
  explicit SELL(std::tuple<impl::shape, idx_t, vector<T2, host>, vector<idx_t, host>,
      vector<idx_t, host>> &&t)
      : spmat(std::get<0>(t), std::get<1>(t)), val(std::get<2>(t)),
        idx(std::get<3>(t)), sptr(std::get<4>(t)) {}
};

template <uint16_t w, uint16_t s, typename T, class L,
    impl::sell_align align = impl::sell_align::left>
struct SELLsig : spmat {
  using loc_t = L;
  using val_t = T;
  constexpr static uint16_t W = w;
  constexpr static uint16_t S = s;
  vector<T, L> val;
  vector<idx_t, L> idx;
  vector<idx_t, L> sptr;
  vector<uint16_t, L> p;

  template <typename T2>
  explicit SELLsig(const CSR<T2, host> &in)
      : SELLsig(impl::csr_to_sell_sigma<w, s, align>(in)) {
    spmat::copy_attrs(in);
  }

  SELLsig(const SELLsig &) = default;
  template <typename T2, class L2>
  explicit SELLsig(const SELLsig<w, s, T2, L2> &in)
      : spmat(in), val(in.val), idx(in.idx), sptr(in.sptr), p(in.p) {}

  template <typename in_t, typename out_t>
  void operate(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(out.size(0), [=, *this, n = out.size(0)](idx_t i) mutable {
      auto sig = i / S * S;
      auto sid = i / w;
      auto myid = i % w;
      auto strt = sptr[sid];
      auto size = sptr[sid + 1] - strt;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto t = out_t(0);
      auto offset = strt * w + myid;
      for (idx_t j = 0; j < size; j++) {
        t += val[offset] * in[idx[offset]];
        offset += t_chunk;
      }
      out[sig + p[i]] = t;
    });
  }
  template <typename rhs_t, typename in_t, typename res_t>
  void compute_residual(const vector<rhs_t, L> &rhs, const vector<in_t, L> &in,
      vector<res_t, L> res) const {
    exec<L>::para_for(res.size(0), [=, *this, n = res.size(0)](idx_t i) mutable {
      auto sig = i / S * S;
      auto sid = i / w;
      auto myid = i % w;
      auto strt = sptr[sid];
      auto size = sptr[sid + 1] - strt;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto t = res_t(0);
      auto offset = strt * w + myid;
      for (idx_t j = 0; j < size; j++) {
        t += val[offset] * in[idx[offset]];
        offset += t_chunk;
      }
      res[sig + p[i]] = rhs[sig + p[i]] - t;
    });
  }

private:
  template <typename T2>
  explicit SELLsig(std::tuple<impl::shape, idx_t, vector<T2, host>,
      vector<idx_t, host>, vector<idx_t, host>, vector<uint16_t, host>> &&t)
      : spmat(std::get<0>(t), std::get<1>(t)), val(std::get<2>(t)),
        idx(std::get<3>(t)), sptr(std::get<4>(t)), p(std::get<5>(t)) {}
};

#if 0

template <uint16_t w, uint16_t s, uint16_t r, typename T, class L,
    impl::sell_align align = impl::sell_align::left>
struct SELLsigseg : spmat {
  using loc_t = L;
  using val_t = T;
  constexpr static uint16_t W = w;
  constexpr static uint16_t S = s;
  vector<T, L> val;
  vector<uint16_t, L> idx;
  vector<uint16_t, L> bptr; // nrosw() / 2**15-1;
  vector<uint16_t, L> bidx;
  vector<idx_t, L> sptr; // #slice in a block
  vector<uint16_t, L> p;

  template <typename T2>
  explicit SELLsigseg(const CSR<T2, host> &in)
      : SELLsigseg(impl::csr_to_sell_sigma_segment<w, s, align>(in)) {}

  SELLsigseg(const SELLsigseg &) = default;
  template <typename T2, class L2>
  explicit SELLsigseg(const SELLsigseg<w, s, r, T2, L2> &in)
      : spmat(in), val(in.val), idx(in.idx), snum(in.sptr), p(in.p) {}

  template <typename in_t, typename out_t>
  void operate(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(out.size(0), [=, *this, n = out.size(0)](idx_t i) mutable {
      auto sig = i / S * S;
      auto sid = i / w;
      auto myid = i % w;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto t = out_t(0);
      for (uint16_t bl = bptr[sig]; bl < bptr[sig + 1]; bl++) {
        auto offset = snum[bl] * w + myid;
        for (idx_t j = 0; j < snum[bl + 1] - snum[bl]; j++) {
          t += val[offset] * in[bidx[bl] + idx[offset]];
          offset += t_chunk;
        }
      }
      out[sig + p[i]] = t;
    });
  }
  template <typename rhs_t, typename in_t, typename res_t>
  void compute_residual(const vector<rhs_t, L> &rhs, const vector<in_t, L> &in,
      vector<res_t, L> res) const {
    exec<L>::para_for(res.size(0), [=, *this, n = res.size(0)](idx_t i) mutable {
      auto sig = i / S * S;
      auto sid = i / w;
      auto myid = i % w;
      auto strt = sptr[sid];
      auto size = sptr[sid + 1] - strt;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto t = res_t(0);
      auto offset = strt * w + myid;
      for (idx_t j = 0; j < size; j++) {
        t += val[offset] * in[idx[offset]];
        offset += t_chunk;
      }
      res[sig + p[i]] = rhs[sig + p[i]] - t;
    });
  }

private:
  template <typename T2>
  explicit SELLsigseg(std::tuple<impl::shape, idx_t, vector<T2, host>,
      vector<idx_t, host>, vector<idx_t, host>, vector<uint16_t, host>> &&t)
      : spmat(std::get<0>(t), std::get<1>(t)), val(std::get<2>(t)),
        idx(std::get<3>(t)), sptr(std::get<4>(t)), bidx(std::get<5>(t)),
        p(std::get<6>(t)), first(std::get<7>(t)) {}
};

#endif

template <typename T, class L, impl::sell_align align = impl::sell_align::left>
using SELL32 = SELL<32, T, L, align>;

namespace impl {

template <uint16_t w, sell_align align, typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<idx_t, host>, vector<idx_t, host>>
csr_to_sell(const CSR<T2, host> &in) {
  auto nrows = in.nrows();
  auto n_slice = (nrows + w - 1) / w;
  idx_t nnz = 0;
  auto _sptr = vector<idx_t, host>(n_slice + 1);
  _sptr[0] = 0;
  for (idx_t i = 0; i < nrows; i += w) {
    auto t_chunk = (nrows % w != 0 && i / w == n_slice - 1) ? nrows % w : w;
    idx_t row_max = 0;
    for (decltype(t_chunk) j = 0; j < t_chunk; ++j)
      row_max = (row_max < in.rptr[i + j + 1] - in.rptr[i + j])
                    ? in.rptr[i + j + 1] - in.rptr[i + j]
                    : row_max;
    nnz += row_max * t_chunk;
    _sptr[i / w + 1] = _sptr[i / w] + row_max;
  }
  auto _val = vector<T2, host>(nnz);
  auto _idx = vector<idx_t, host>(nnz);

  for (idx_t i = 0; i < nrows; i += w) {
    auto s_id = i / w;
    auto t_chunk = (nrows % w != 0 && s_id == n_slice - 1) ? nrows % w : w;
    auto row_max = _sptr[s_id + 1] - _sptr[s_id];
    if constexpr (align == impl::sell_align::left) {
      idx_t off = _sptr[s_id] * w;
      for (decltype(t_chunk) j = 0; j < t_chunk; ++j) {
        idx_t row_cnt = 0;
        for (idx_t k = in.rptr[i + j]; k < in.rptr[i + j + 1]; ++k) {
          _val[off + row_cnt * t_chunk + j] = in.val[k];
          _idx[off + row_cnt * t_chunk + j] = in.idx[k];
          row_cnt++;
        }
        while (row_cnt < row_max) {
          _val[off + row_cnt * t_chunk + j] = 0.;
          _idx[off + row_cnt * t_chunk + j] =
              (row_cnt == 0) ? 0 : _idx[off + (row_cnt - 1) * t_chunk + j];
          row_cnt++;
        }
      }
    } else {
      idx_t off = _sptr[s_id] * w;
      for (decltype(t_chunk) j = 0; j < t_chunk; ++j) {
        idx_t row_cnt = 0;
        for (idx_t k = in.rptr[i + j + 1] - 1; k >= in.rptr[i + j]; --k) {
          _val[off + (row_max - 1 - row_cnt) * t_chunk + j] = in.val[k];
          _idx[off + (row_max - 1 - row_cnt) * t_chunk + j] = in.idx[k];
          row_cnt++;
        }
        while (row_cnt < row_max) {
          _val[off + (row_max - 1 - row_cnt) * t_chunk + j] = 0.;
          _idx[off + (row_max - 1 - row_cnt) * t_chunk + j] =
              (row_cnt == 0) ? 0 : _idx[off + (row_max - row_cnt) * t_chunk + j];
          row_cnt++;
        }
      }
    }
  }
  return {in.shp, nnz, _val, _idx, _sptr};
}

template <uint16_t w, uint16_t s, sell_align align, typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<idx_t, host>, vector<idx_t, host>,
    vector<uint16_t, host>>
csr_to_sell_sigma(const CSR<T2, host> &in) {

  auto pi = vector<uint16_t, host>(in.nrows());
  for (int i = 0; i < in.nrows(); i += s) {
    auto len = (i + s > in.nrows()) ? in.nrows() - i : s;
    for (int j = 0; j < len; j++)
      pi[i + j] = uint16_t(j);
  }
  auto key = vector<idx_t, host>(in.nrows());
  for (int i = 0; i < in.nrows(); i++)
    key[i] = in.rptr[i + 1] - in.rptr[i];
  for (int i = 0; i < in.nrows(); i += s) {
    auto sig = i;
    auto e = (i + s > in.nrows()) ? in.nrows() : i + s;
    sort::quick<sort::order::desc>(sig, e, key.raw(), pi.raw());
  }
  auto re_in = CSR<T2, host>(in.get_shape(), in.nnz());
  idx_t nnz = 0;
  re_in.rptr[0] = nnz;
  for (idx_t i = 0; i < in.nrows(); i++) {
    auto sig = i / s * s;
    idx_t row = sig + pi[i];
    for (idx_t j = in.rptr[row]; j < in.rptr[row + 1]; j++, nnz++) {
      re_in.val[nnz] = in.val[j];
      re_in.idx[nnz] = in.idx[j];
    }
    re_in.rptr[i + 1] = nnz;
  }

  auto [shp, _nnz, _val, _idx, _sptr] = csr_to_sell<w, align, T2>(re_in);
  return {shp, _nnz, _val, _idx, _sptr, pi};
}

#if 0

template <uint16_t w, uint16_t s, uint16_t r, sell_align align, typename T2>
std::tuple<shape, idx_t, vector<T2, host>, vector<uint16_t, host>,
    vector<uint16_t, host>, vector<uint16_t, host>, vector<idx_t, host>,
    vector<uint16_t, host>>
csr_to_sell_sigma_segment(const CSR<T2, host> &in) {
  auto pi = vector<uint16_t, host>(in.nrows());
  for (int i = 0; i < in.nrows(); i += s) {
    auto len = (i + s > in.nrows()) ? in.nrows() - i : s;
    for (int j = 0; j < len; j++)
      pi[i + j] = uint16_t(j);
  }
  auto key = vector<idx_t, host>(in.nrows());
  for (int i = 0; i < in.nrows(); i++)
    key[i] = in.rptr[i + 1] - in.rptr[i];
  for (int i = 0; i < in.nrows(); i += s) {
    auto sig = i;
    auto e = (i + s > in.nrows()) ? in.nrows() : i + s;
    sort::quick<sort::order::desc>(sig, e, key.raw(), pi.raw());
  }
  auto re_in = CSR<T2, host>(in.get_shape(), in.nnz());
  idx_t nnz = 0;
  re_in.rptr[0] = nnz;
  for (idx_t i = 0; i < in.nrows(); i++) {
    auto sig = i / s * s;
    idx_t row = sig + pi[i];
    for (idx_t j = in.rptr[row]; j < in.rptr[row + 1]; j++, nnz++) {
      re_in.val[nnz] = in.val[j];
      re_in.idx[nnz] = in.idx[j];
    }
    re_in.rptr[i + 1] = nnz;
  }

  auto nrows = in.nrows();
  auto n_slice = (nrows + w - 1) / w;
  idx_t _nnz = 0;
  auto _bptr = vector<idx_t, host>(n_slice + 1);
  _bptr[0] = 0;
  for (idx_t i = 0; i < nrows; i += w) {
    auto t_chunk = (nrows % w != 0 && i / w == n_slice - 1) ? nrows % w : w;
    idx_t row_max = 0;
    for (decltype(t_chunk) j = 0; j < t_chunk; ++j)
      row_max = (row_max < in.rptr[i + j + 1] - in.rptr[i + j])
                    ? in.rptr[i + j + 1] - in.rptr[i + j]
                    : row_max;
    _nnz += row_max * t_chunk;
    _sptr[i / w + 1] = _sptr[i / w] + row_max;
  }
  auto _val = vector<T2, host>(_nnz);
  auto _idx = vector<idx_t, host>(_nnz);

  for (idx_t i = 0; i < nrows; i += w) {
    auto s_id = i / w;
    auto t_chunk = (nrows % w != 0 && s_id == n_slice - 1) ? nrows % w : w;
    auto row_max = _sptr[s_id + 1] - _sptr[s_id];
    if constexpr (align == impl::sell_align::left) {
      idx_t off = _sptr[s_id] * w;
      for (decltype(t_chunk) j = 0; j < t_chunk; ++j) {
        idx_t row_cnt = 0;
        for (idx_t k = in.rptr[i + j]; k < in.rptr[i + j + 1]; ++k) {
          _val[off + row_cnt * t_chunk + j] = in.val[k];
          _idx[off + row_cnt * t_chunk + j] = in.idx[k];
          row_cnt++;
        }
        while (row_cnt < row_max) {
          _val[off + row_cnt * t_chunk + j] = 0.;
          _idx[off + row_cnt * t_chunk + j] =
              (row_cnt == 0) ? 0 : _idx[off + (row_cnt - 1) * t_chunk + j];
          row_cnt++;
        }
      }
    } else {
      idx_t off = _sptr[s_id] * w;
      for (decltype(t_chunk) j = 0; j < t_chunk; ++j) {
        idx_t row_cnt = 0;
        for (idx_t k = in.rptr[i + j + 1] - 1; k >= in.rptr[i + j]; --k) {
          _val[off + (row_max - 1 - row_cnt) * t_chunk + j] = in.val[k];
          _idx[off + (row_max - 1 - row_cnt) * t_chunk + j] = in.idx[k];
          row_cnt++;
        }
        while (row_cnt < row_max) {
          _val[off + (row_max - 1 - row_cnt) * t_chunk + j] = 0.;
          _idx[off + (row_max - 1 - row_cnt) * t_chunk + j] =
              (row_cnt == 0) ? 0 : _idx[off + (row_max - row_cnt) * t_chunk + j];
          row_cnt++;
        }
      }
    }
  }

  return {shp, _nnz, _val, _idx, _sptr, pi};
}

#endif

#if defined(KMM_WITH_CUDA)
#define KMM_LOC __host__ __device__
#else
#define KMM_LOC
#endif
#define KMM_ENABULER(cond) std::enable_if_t<cond, std::nullptr_t> = nullptr
#define KMM_IS_SAME_V(cond1, cond2) std::is_same_v<cond1, cond2>

template <typename> struct is_sell {
  static constexpr bool value = false;
};
template <class T> inline constexpr bool is_sell_v = is_sell<T>::value;
template <uint16_t w, typename T, class L, sell_align align>
struct is_sell<SELL<w, T, L, align>> {
  static constexpr bool value = true;
};

template <typename L, typename R, KMM_ENABULER(is_sell_v<L>),
    KMM_ENABULER(is_vector_expr_v<R>)>
struct SELLMV {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<typename L::val_t &>() *
                         std::declval<typename R::val_t &>());
  L l;
  R r;
  SELLMV(const L &l, const R &r) : l(l), r(r) {}
  KMM_LOC auto eval(idx_t i = 0, [[maybe_unused]] idx_t j = 0) const {
    auto sid = i / l.W;
    auto myid = i % l.W;
    auto s = l.sptr[sid];
    auto size = l.sptr[sid + 1] - s;
    uint16_t t_chunk = (sid != l.nrows() / l.W) ? l.W : l.nrows() % l.W;
    auto t = val_t{0};
    auto offset = s * l.W + myid;
    for (idx_t k = 0; k < size; k++) {
      t += l.val[offset] * r.eval(l.idx[offset]);
      offset += t_chunk;
    }
    return t;
  }
  [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return r.size(i); }
};
template <typename T1, typename T2> struct is_vector_expr<SELLMV<T1, T2>> {
  static constexpr bool value = true;
};

#undef KMM_ENABULER
#undef KMM_IS_SAME_V
#undef KMM_LOC

} // namespace impl

} // namespace kmm

template <typename L, typename R> kmm::impl::SELLMV<L, R> operator*(L v1, R v2) {
  return kmm::impl::SELLMV<L, R>(v1, v2);
}

#endif // UMINEKO_SPARSE_MATRIX_SELL_HPP
