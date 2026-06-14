#ifndef SENK_SPMV_BSR_HPP
#define SENK_SPMV_BSR_HPP

#include "senk/matrix/base.hpp"
#include "senk/matrix/bsr.hpp"

#include "senk/spmv/base.hpp"

namespace senk {

namespace impl {

template <uint16_t bnl, uint16_t bnw, typename loc_t, typename val_t,
    typename idx_t, typename srl_t, typename in_t, typename out_t>
void spmv_bsr_apply_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out);
template <uint16_t bnl, uint16_t bnw, typename loc_t, typename val_t,
    typename idx_t, typename srl_t, typename rhs_t, typename in_t,
    typename out_t>
void spmv_bsr_residual_impl(const idx_t nrows, const idx_t ncols,
    const val_t *val, const idx_t *col, const srl_t *rptr, const rhs_t *rhs,
    const in_t *in, out_t *out);

} // namespace impl

namespace spmv {

template <class Algo, uint16_t bnl, uint16_t bnw, typename T, class L,
    typename I, typename S>
struct BSR : private senk::BSR<bnl, bnw, T, L, I, S>,
             public has_params<Algo>,
             public model::is_operator<BSR<Algo, bnl, bnw, T, L, I, S>> {
  using Base = senk::BSR<bnl, bnw, T, L, I, S>;
  using Base::attr;
  using Base::col;
  using Base::rptr;
  using Base::shape;
  using Base::val;
  using typename Base::idx_t;
  using typename Base::loc_t;
  using typename Base::val_t;
  using model::is_operator<BSR>::apply;
  using model::is_operator<BSR>::residual;
  using model::is_operator<Base>::nrows;
  using model::is_operator<Base>::ncols;
  using typename has_params<Algo>::Params;

  const Params prm;

  explicit BSR(const Base &in, Params prm = Params{}) : Base(in), prm(prm) {}

private:
#define PASTA_SPMV_APPLY_ARGS                                                  \
  nrows_impl(), ncols_impl(), val.raw(), col.raw(), rptr.raw(), in.raw(),      \
      out.raw()
#define PASTA_SPMV_RESIDUAL_ARGS                                               \
  nrows_impl(), ncols_impl(), val.raw(), col.raw(), rptr.raw(), rhs.raw(),     \
      in.raw(), out.raw()

  template <typename in_t, typename out_t>
  void apply_impl(const vector<in_t, L> &in, vector<out_t, L> out) const {
    if constexpr (spmv::is_algo_default_v<Algo>) {
      impl::spmv_bsr_apply_impl<bnl, bnw, loc_t>(PASTA_SPMV_APPLY_ARGS);
    }
  }

  template <typename rhs_t, typename in_t, typename out_t>
  void residual_impl(const vector<rhs_t, loc_t> &rhs,
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    if constexpr (spmv::is_algo_default_v<Algo>) {
      impl::spmv_bsr_residual_impl<bnl, bnw, loc_t>(PASTA_SPMV_RESIDUAL_ARGS);
    }
  }
  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }

  friend struct model::is_operator<BSR>;

#undef PASTA_SPMV_APPLY_ARGS
#undef PASTA_SPMV_RESIDUAL_ARGS
};

} // namespace spmv

template <class Algo, uint16_t bnl, uint16_t bnw, typename T, class L,
    typename I, typename S>
auto SpMV(const senk::BSR<bnl, bnw, T, L, I, S> &in) {
  return spmv::BSR<Algo, bnl, bnw, T, L, I, S>(in, {});
}

template <uint16_t bnl, uint16_t bnw, typename T, class L, typename I,
    typename S>
auto SpMV(const senk::BSR<bnl, bnw, T, L, I, S> &in) {
  return spmv::BSR<spmv::algo_default, bnl, bnw, T, L, I, S>(in);
}

namespace impl {

#define PASTA_BCSR_SIMD(col)                                                   \
  _Pragma("omp simd simdlen(bnl)") for (uint16_t k = 0; k < bnl; ++k) {        \
    t[k] += val[j * area + bnl * (col) + k] * x[(col)];                        \
  }

template <uint16_t bnl, uint16_t bnw, typename loc_t, typename val_t,
    typename idx_t, typename srl_t, typename in_t, typename out_t>
void spmv_bsr_apply_impl(const idx_t nrows, const idx_t ncols, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out) {
  const auto bncols = (ncols + bnw - 1) / bnw;
  kernel<loc_t>::parallel(
      (nrows + bnl - 1) / bnl, [=] SENK_LOC(idx_t bidx) mutable {
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
          PASTA_BCSR_SIMD(0)
          if constexpr (bnw == 2 || bnw == 4 || bnw == 8) {
            PASTA_BCSR_SIMD(1)
          }
          if constexpr (bnw == 4 || bnw == 8) {
            PASTA_BCSR_SIMD(2)
            PASTA_BCSR_SIMD(3)
          }
          if constexpr (bnw == 8) {
            PASTA_BCSR_SIMD(4)
            PASTA_BCSR_SIMD(5)
            PASTA_BCSR_SIMD(6)
            PASTA_BCSR_SIMD(7)
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
void spmv_bsr_residual_impl(const idx_t nrows, const idx_t ncols,
    const val_t *val, const idx_t *col, const srl_t *rptr, const rhs_t *rhs,
    const in_t *in, out_t *out) {
  const auto bncols = (ncols + bnw - 1) / bnw;
  kernel<loc_t>::parallel(
      (nrows + bnl - 1) / bnl, [=] SENK_LOC(idx_t bidx) mutable {
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
          PASTA_BCSR_SIMD(0)
          if constexpr (bnw == 2 || bnw == 4 || bnw == 8) {
            PASTA_BCSR_SIMD(1)
          }
          if constexpr (bnw == 4 || bnw == 8) {
            PASTA_BCSR_SIMD(2)
            PASTA_BCSR_SIMD(3)
          }
          if constexpr (bnw == 8) {
            PASTA_BCSR_SIMD(4)
            PASTA_BCSR_SIMD(5)
            PASTA_BCSR_SIMD(6)
            PASTA_BCSR_SIMD(7)
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

#undef PASTA_BCSR_SIMD

} // namespace impl

} // namespace senk

#endif
