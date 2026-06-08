#ifndef SENK_SPMV_CSR_HPP
#define SENK_SPMV_CSR_HPP

#include "senk/matrix/base.hpp"
#include "senk/matrix/csr.hpp"

#include "senk/spmv/base.hpp"

namespace senk {

namespace impl {

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void spmv_csr_apply_impl(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const in_t *in, out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void spmv_csr_residual_impl(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const rhs_t *rhs, const in_t *in,
    out_t *out);

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void spmv_csr_apply_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out);
template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void spmv_csr_residual_impl(const idx_t nrows, const idx_t ncols,
    const val_t *val, const idx_t *col, const srl_t *rptr, const rhs_t *rhs,
    const in_t *in, out_t *out);

template <uint8_t p, typename loc_t, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void spmv_csr_col_apply_impl(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out);
template <uint8_t p, typename loc_t, typename val_t, typename idx_t,
    typename srl_t, typename rhs_t, typename in_t, typename out_t>
void spmv_csr_col_residual_impl(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const rhs_t *rhs, const in_t *in,
    out_t *out);

} // namespace impl

namespace spmv {

template <class Algo, typename T, class L, typename I, typename S>
struct CSR : private senk::CSR<T, L, I, S>,
             public has_params<Algo>,
             public model::is_operator<CSR<Algo, T, L, I, S>> {
  using Base = senk::CSR<T, L, I, S>;
  using Base::attr;
  using Base::col;
  using Base::rptr;
  using Base::shape;
  using Base::val;
  using typename Base::idx_t;
  using typename Base::loc_t;
  using typename Base::srl_t;
  using typename Base::val_t;
  using model::is_operator<CSR>::apply;
  using model::is_operator<CSR>::residual;
  using model::is_operator<Base>::nrows;
  using model::is_operator<Base>::ncols;
  using typename has_params<Algo>::Params;

  const Params prm;

  explicit CSR(const Base &in, Params prm = Params{}) : Base(in), prm(prm) {}

// private:
#define PASTA_SPMV_APPLY_ARGS                                                  \
  nrows_impl(), val.raw(), col.raw(), rptr.raw(), in.raw(), out.raw()
#define PASTA_SPMV_APPLY_ARGS_M                                                \
  static_cast<idx_t>(in.shape(0)), static_cast<idx_t>(in.shape(1)), val.raw(), \
      col.raw(), rptr.raw(), in.raw(), out.raw()
#define PASTA_SPMV_RESIDUAL_ARGS                                               \
  nrows_impl(), val.raw(), col.raw(), rptr.raw(), rhs.raw(), in.raw(), out.raw()

  template <typename in_t, typename out_t>
  void apply_impl(const vector<in_t, L> &in, vector<out_t, L> out) const {
    if constexpr (spmv::is_algo_default_v<Algo>) {
      impl::spmv_csr_apply_impl<loc_t>(PASTA_SPMV_APPLY_ARGS);
    } else if constexpr (spmv::is_algo_reduce_v<Algo>) {
      impl::spmv_csr_col_apply_impl<Algo::p, loc_t>(PASTA_SPMV_APPLY_ARGS);
    } else if constexpr (spmv::is_algo_reduce_opt_v<Algo>) {
      if (prm.score <= 1.) {
        impl::spmv_csr_col_apply_impl<1, loc_t>(PASTA_SPMV_APPLY_ARGS);
      } else if (prm.score <= 2) {
        impl::spmv_csr_col_apply_impl<2, loc_t>(PASTA_SPMV_APPLY_ARGS);
      } else if (prm.score <= 4) {
        impl::spmv_csr_col_apply_impl<4, loc_t>(PASTA_SPMV_APPLY_ARGS);
      } else if (prm.score <= 8) {
        impl::spmv_csr_col_apply_impl<8, loc_t>(PASTA_SPMV_APPLY_ARGS);
      } else if (prm.score <= 16) {
        impl::spmv_csr_col_apply_impl<16, loc_t>(PASTA_SPMV_APPLY_ARGS);
      } else {
        impl::spmv_csr_col_apply_impl<32, loc_t>(PASTA_SPMV_APPLY_ARGS);
      }
    }
  }

  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    if constexpr (spmv::is_algo_default_v<Algo>) {
      impl::spmv_csr_residual_impl<loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
    } else if constexpr (spmv::is_algo_reduce_v<Algo>) {
      impl::spmv_csr_col_residual_impl<Algo::p, loc_t>(
          PASTA_SPMV_RESIDUAL_ARGS);
    } else if constexpr (spmv::is_algo_reduce_opt_v<Algo>) {
      if (prm.score <= 1.)
        impl::spmv_csr_col_residual_impl<1, loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
      else if (prm.score <= 2)
        impl::spmv_csr_col_residual_impl<2, loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
      else if (prm.score <= 4)
        impl::spmv_csr_col_residual_impl<4, loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
      else if (prm.score <= 8)
        impl::spmv_csr_col_residual_impl<8, loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
      else if (prm.score <= 16)
        impl::spmv_csr_col_residual_impl<16, loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
      else
        impl::spmv_csr_col_residual_impl<32, loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
    }
  }

  template <typename in_t, typename out_t>
  void apply_impl(const matrix<in_t, L> &in, matrix<out_t, L> out) const {
    impl::spmv_csr_apply_impl<loc_t>(PASTA_SPMV_APPLY_ARGS_M);
  }

  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }

  friend struct model::is_operator<CSR>;

#undef PASTA_SPMV_APPLY_ARGS
#undef PASTA_SPMV_RESIDUAL_ARGS
};

} // namespace spmv

template <class Algo, typename T, class L, typename I, typename S>
spmv::CSR<Algo, T, L, I, S> SpMV(const senk::CSR<T, L, I, S> &in) {
  if constexpr (spmv::is_algo_reduce_opt_v<Algo>) {
    auto nrows = in.nrows();
    auto sco = vector<double, L>(nrows);
    auto reduce = reducer<L>(nrows);
    scalar<double, L> score;
    scalar<double, host> hscore;
    auto scop = sco.raw();
    auto rptr = in.rptr.raw();
    kernel<L>::parallel(nrows, [=] SENK_LOC(size_t i) {
      auto bw = rptr[i + 1] - rptr[i];
      auto e = (std::is_same_v<T, double>) ? (bw + 7) / 8 : (bw + 15) / 16;
      scop[i] = e * bw;
    });
    score = reduce.add(sco);
    score /= in.rptr(in.nrows());
    hscore.copy(score);
    // printf("score: %e\n", hscore[0]);
    return spmv::CSR<Algo, T, L, I, S>(in, {hscore[0]});
  } else
    return spmv::CSR<Algo, T, L, I, S>(in, {});
}

template <typename T, class L, typename I, typename S>
auto SpMV(const senk::CSR<T, L, I, S> &in) {
  return spmv::CSR<spmv::algo_default, T, L, I, S>(in);
}

namespace impl {

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename in_t, typename out_t>
void spmv_csr_apply_impl(const idx_t nrows, const val_t *val, const idx_t *col,
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
void spmv_csr_residual_impl(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const rhs_t *rhs, const in_t *in,
    out_t *out) {
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
    typename in_t, typename out_t>
void spmv_csr_apply_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    const int chunk = 5;
    tmp_t tmp[chunk] = {0.};
    for (auto j = rptr[i]; j < rptr[i + 1]; j++) {
      auto _val = val[j];
      auto _col = col[j];
      for (idx_t k = 0; k < ncols; k += chunk) {
        auto end = std::min(chunk, ncols - k);
        for (idx_t l = 0; l < end; l++) {
          tmp[l] += _val * in[(k + l) * nrows + _col];
        }
      }
    }
    for (idx_t k = 0; k < ncols; k += chunk) {
      auto end = std::min(chunk, ncols - k);
      for (idx_t l = 0; l < end; l++)
        out[(k + l) * nrows + i] = static_cast<out_t>(tmp[l]);
    }
  });
}

template <typename loc_t, typename val_t, typename idx_t, typename srl_t,
    typename rhs_t, typename in_t, typename out_t>
void spmv_csr_residual_impl(const idx_t nrows, const idx_t ncols,
    const val_t *val, const idx_t *col, const srl_t *rptr, const rhs_t *rhs,
    const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
    auto tmp = static_cast<tmp_t>(0.);
    for (auto j = rptr[i]; j < rptr[i + 1]; j++) {
      tmp += val[j] * in[col[j]];
    }
    out[i] = static_cast<out_t>(rhs[i] - tmp);
  });
}

template <uint8_t p, typename loc_t, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void spmv_csr_col_apply_impl(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  if constexpr (std::is_same_v<loc_t, host>) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
      auto tmp = static_cast<tmp_t>(0.);
      for (auto j = rptr[i]; j < rptr[i + 1]; j++)
        tmp += val[j] * in[col[j]];
      out[i] = static_cast<out_t>(tmp);
    });
  } else {
#if defined(SENK_WITH_CUDA)
    kernel<loc_t>::template parallel_p<p>(
        nrows, [=] SENK_LOC(size_t i, size_t j) {
          auto tmp = static_cast<tmp_t>(0.);
          for (auto k = rptr[i] + j; k < rptr[i + 1]; k += p)
            tmp += val[k] * in[col[k]];
#pragma unroll
          for (int k = p; k >= 2; k >>= 1)
            tmp += __shfl_down_sync(0xffffffff, tmp, k >> 1, k);
          if (j == 0)
            out[i] = static_cast<out_t>(tmp);
        });
#endif
#if defined(SENK_WITH_HIP)
    kernel<loc_t>::template parallel_p<p>(
        nrows, [=] SENK_LOC(size_t i, size_t j) {
          auto tmp = static_cast<tmp_t>(0.);
          for (auto k = rptr[i] + j; k < rptr[i + 1]; k += p)
            tmp += val[k] * in[col[k]];
#pragma unroll
          for (int k = p; k >= 2; k >>= 1)
            tmp += __shfl_down_sync(0xffffffffffffffff, tmp, k >> 1, k);
          if (j == 0)
            out[i] = static_cast<out_t>(tmp);
        });
#endif
  }
}

template <uint8_t p, typename loc_t, typename val_t, typename idx_t,
    typename srl_t, typename rhs_t, typename in_t, typename out_t>
void spmv_csr_col_residual_impl(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const rhs_t *rhs, const in_t *in,
    out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  if constexpr (std::is_same_v<loc_t, host>) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(size_t i) {
      auto tmp = static_cast<tmp_t>(0.);
      for (auto j = rptr[i]; j < rptr[i + 1]; j++)
        tmp += val[j] * in[col[j]];
      out[i] = static_cast<out_t>(rhs[i] - tmp);
    });
  } else {
#if defined(SENK_WITH_CUDA)
    kernel<loc_t>::template parallel_p<p>(
        nrows, [=] SENK_LOC(size_t i, size_t j) {
          auto tmp = static_cast<tmp_t>(0.);
          for (auto k = rptr[i] + j; k < rptr[i + 1]; k += p)
            tmp += val[k] * in[col[k]];
#pragma unroll
          for (int k = p; k >= 2; k >>= 1)
            tmp += __shfl_down_sync(0xffffffff, tmp, k >> 1, k);
          if (j == 0)
            out[i] = static_cast<out_t>(rhs[i] - tmp);
        });
#endif
#if defined(SENK_WITH_HIP)
    kernel<loc_t>::template parallel_p<p>(
        nrows, [=] SENK_LOC(size_t i, size_t j) {
          auto tmp = static_cast<tmp_t>(0.);
          for (auto k = rptr[i] + j; k < rptr[i + 1]; k += p)
            tmp += val[k] * in[col[k]];
#pragma unroll
          for (int k = p; k >= 2; k >>= 1)
            tmp += __shfl_down_sync(0xffffffffffffffff, tmp, k >> 1, k);
          if (j == 0)
            out[i] = static_cast<out_t>(rhs[i] - tmp);
        });
#endif
  }
}

} // namespace impl

} // namespace senk

#endif