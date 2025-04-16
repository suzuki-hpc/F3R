#ifndef UMINEKO_SPARSE_TRSV_BJ_HPP
#define UMINEKO_SPARSE_TRSV_BJ_HPP

#include "umineko-sparse/matrix/base.hpp"
#include "umineko-sparse/matrix/bcsr.hpp"
#include "umineko-sparse/matrix/csr.hpp"

#include "umineko-sparse/models.hpp"

namespace kmm {

namespace trsv {

template <class, spmat::Form> struct BJ;

namespace l {
template <class C> using BJ = BJ<C, spmat::Form::lower>;
}
namespace ld {
template <class C> using BJ = BJ<C, spmat::Form::lower_d>;
}
namespace du {
template <class C> using BJ = BJ<C, spmat::Form::d_upper>;
}

template <typename T, class L, spmat::Form form>
struct BJ<CSR<T, L>, form> : private CSR<T, L> {
  static_assert(std::is_same_v<L, host>, "BJ<CSR> does not work on device");
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  using CSR<T, L>::pattern_set;

  struct params {
    idx_t block_num;
  };

  template <typename _>
  explicit BJ(const CSR<_, host> &in, params param)
      : CSR<T, L>([&] {
          if (in.pattern == spmat::Pattern::block_diagonal) {
            if constexpr (form == spmat::Form::d_upper)
              return in.duplicate().rotate180().inverse_last();
            else if constexpr (form == spmat::Form::lower_d)
              return in.duplicate_val().inverse_last();
            else
              return in.duplicate();
          } else {
            if constexpr (form == spmat::Form::d_upper)
              return in.duplicate_block_diagonal(param.block_num, 1)
                  .rotate180()
                  .inverse_last();
            else if constexpr (form == spmat::Form::lower_d)
              return in.duplicate_block_diagonal(param.block_num, 1).inverse_last();
            else
              return in.duplicate_block_diagonal(param.block_num, 1);
          }
        }()) {}

  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
    if constexpr (form == spmat::Form::lower) {
      exec<L>::para_for(pattern_set.idx_size, [=, *this](idx_t id) mutable {
        auto s = pattern_set.idx_ptr[id];
        auto e = pattern_set.idx_ptr[id + 1];
        for (idx_t i = s; i < e; ++i) {
          auto t = in[i];
          auto j = rptr[i];
          for (; j < rptr[i + 1] - 1; ++j)
            t -= val[j] * out[idx[j]];
          out[i] = t;
        }
      });
    } else if constexpr (form == spmat::Form::lower_d) {
      exec<L>::para_for(pattern_set.idx_size, [=, *this](idx_t id) mutable {
        auto s = pattern_set.idx_ptr[id];
        auto e = pattern_set.idx_ptr[id + 1];
        for (idx_t i = s; i < e; ++i) {
          auto t = in[i];
          auto j = rptr[i];
          for (; j < rptr[i + 1] - 1; ++j)
            t -= val[j] * out[idx[j]];
          out[i] = t * val[j];
        }
      });
    } else if constexpr (form == spmat::Form::d_upper) {
      auto n = in.size(0);
      exec<L>::para_for(pattern_set.idx_size, [=, *this](idx_t id) mutable {
        auto s = pattern_set.idx_ptr[id];
        auto e = pattern_set.idx_ptr[id + 1];
        for (idx_t i = s; i < e; ++i) {
          auto t = in[n - 1 - i];
          auto j = rptr[i];
          for (; j < rptr[i + 1] - 1; ++j)
            t -= val[j] * out[n - 1 - idx[j]];
          out[n - 1 - i] = t * val[j];
        }
      });
    }
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

template <uint16_t bnl, uint16_t bnw, typename T, class L, spmat::Form form>
struct BJ<BCSR<bnl, bnw, T, L>, form> : private BCSR<bnl, bnw, T, L> {
  static_assert(std::is_same_v<L, host>, "BJ<CSR> does not work on device");
  using BCSR<bnl, bnw, T, L>::val, BCSR<bnl, bnw, T, L>::idx,
      BCSR<bnl, bnw, T, L>::rptr;
  using BCSR<bnl, bnw, T, L>::pattern_set;

  struct params {
    idx_t block_num;
  };

  template <typename _>
  explicit BJ(const CSR<_, host> &in, params param)
      : BCSR<bnl, bnw, T, L>([&] {
          if (in.pattern == spmat::Pattern::block_diagonal) {
            if constexpr (form == spmat::Form::d_upper)
              return in.duplicate().rotate180().inverse_last();
            else if constexpr (form == spmat::Form::lower_d)
              return in.duplicate_val().inverse_last();
            else
              return in.duplicate();
          } else {
            if constexpr (form == spmat::Form::d_upper)
              return in.duplicate_block_diagonal(param.block_num, bnl)
                  .rotate180()
                  .inverse_last();
            else if constexpr (form == spmat::Form::lower_d)
              return in.duplicate_block_diagonal(param.block_num, bnl)
                  .inverse_last();
            else {
              return in.duplicate_block_diagonal(param.block_num, bnl);
            }
          }
        }()) {}

  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define BCSR_TRSV_L_INNER(col, l, r)                                                \
  _Pragma("omp simd simdlen(bnl)") for (uint16_t k = 0; k < bnl; k++) {             \
    (l)[i + k] -= val[j * area + bnl * (col) + k] * (r)[x_ind + (col)];             \
  }

#define BCSR_TRSV_U_INNER(col, l, r)                                                \
  _Pragma("omp simd simdlen(bnl)") for (uint16_t k = 0; k < bnl; k++) {             \
    (l)[n - 1 - i - k] -=                                                           \
        val[j * area + bnl * (col) + k] * (r)[n - 1 - x_ind - (col)];               \
  }
    const auto area = bnl * bnw;
    const auto n_diag_blocks = bnl / bnw;
    if constexpr (form == spmat::Form::lower) {
      exec<L>::para_for(pattern_set.idx_size, [=, *this](idx_t id) mutable {
        auto s = pattern_set.idx_ptr[id];
        auto e = pattern_set.idx_ptr[id + 1];
        for (idx_t i = s; i < e; i += bnl) {
          auto bidx = i / bnl;
#pragma omp simd simdlen(bnl)
          for (idx_t k = 0; k < bnl; k++)
            out[i + k] = in[i + k];
          auto j = rptr[bidx];
          for (; j < rptr[bidx + 1] - n_diag_blocks; ++j) {
            auto x_ind = idx[j] * bnw;
            BCSR_TRSV_L_INNER(0, out, out)
            if constexpr (bnw == 2 || bnw == 4 || bnw == 8) {
              BCSR_TRSV_L_INNER(1, out, out)
            }
            if constexpr (bnw == 4 || bnw == 8) {
              BCSR_TRSV_L_INNER(2, out, out)
              BCSR_TRSV_L_INNER(3, out, out)
            }
            if constexpr (bnw == 8) {
              BCSR_TRSV_L_INNER(4, out, out)
              BCSR_TRSV_L_INNER(5, out, out)
              BCSR_TRSV_L_INNER(6, out, out)
              BCSR_TRSV_L_INNER(7, out, out)
            }
          }
          auto cnt = 0;
          for (idx_t _s = 0; _s < bnl; _s++) {
            auto m = _s / bnw;
            auto k = _s % bnw;
            for (idx_t l = cnt + 1; l < bnl; l++) {
              out[i + l] -= val[(j + m) * area + k * bnl + l] * out[i + cnt];
            }
            ++cnt;
          }
        }
      });
    } else if constexpr (form == spmat::Form::lower_d) {
      exec<L>::para_for(pattern_set.idx_size, [=, *this](idx_t id) mutable {
        auto s = pattern_set.idx_ptr[id];
        auto e = pattern_set.idx_ptr[id + 1];
        for (idx_t i = s; i < e; i += bnl) {
          auto bidx = i / bnl;
#pragma omp simd simdlen(bnl)
          for (idx_t k = 0; k < bnl; k++)
            out[i + k] = in[i + k];
          auto j = rptr[bidx];
          for (; j < rptr[bidx + 1] - n_diag_blocks; ++j) {
            auto x_ind = idx[j] * bnw;
            BCSR_TRSV_L_INNER(0, out, out)
            if constexpr (bnw == 2 || bnw == 4 || bnw == 8) {
              BCSR_TRSV_L_INNER(1, out, out)
            }
            if constexpr (bnw == 4 || bnw == 8) {
              BCSR_TRSV_L_INNER(2, out, out)
              BCSR_TRSV_L_INNER(3, out, out)
            }
            if constexpr (bnw == 8) {
              BCSR_TRSV_L_INNER(4, out, out)
              BCSR_TRSV_L_INNER(5, out, out)
              BCSR_TRSV_L_INNER(6, out, out)
              BCSR_TRSV_L_INNER(7, out, out)
            }
          }
          auto cnt = 0;
          for (idx_t _s = 0; _s < bnl; _s++) {
            auto m = _s / bnw;
            auto k = _s % bnw;
            out[i + cnt] *= val[(j + m) * area + k * bnl + cnt];
            for (idx_t l = cnt + 1; l < bnl; l++) {
              out[i + l] -= val[(j + m) * area + k * bnl + l] * out[i + cnt];
            }
            ++cnt;
          }
        }
      });
    } else if constexpr (form == spmat::Form::d_upper) {
      auto n = in.size(0);
      exec<L>::para_for(pattern_set.idx_size, [=, *this](idx_t id) mutable {
        auto s = pattern_set.idx_ptr[id];
        auto e = pattern_set.idx_ptr[id + 1];
        for (idx_t i = s; i < e; i += bnl) {
          auto bidx = i / bnl;
#pragma omp simd simdlen(bnl)
          for (idx_t k = 0; k < bnl; k++)
            out[n - 1 - i - k] = in[n - 1 - i - k];
          auto j = rptr[bidx];
          for (; j < rptr[bidx + 1] - n_diag_blocks; ++j) {
            auto x_ind = idx[j] * bnw;
            BCSR_TRSV_U_INNER(0, out, out)
            if constexpr (bnw == 2 || bnw == 4 || bnw == 8) {
              BCSR_TRSV_U_INNER(1, out, out)
            }
            if constexpr (bnw == 4 || bnw == 8) {
              BCSR_TRSV_U_INNER(2, out, out)
              BCSR_TRSV_U_INNER(3, out, out)
            }
            if constexpr (bnw == 8) {
              BCSR_TRSV_U_INNER(4, out, out)
              BCSR_TRSV_U_INNER(5, out, out)
              BCSR_TRSV_U_INNER(6, out, out)
              BCSR_TRSV_U_INNER(7, out, out)
            }
          }
          auto cnt = 0;
          for (idx_t _s = 0; _s < bnl; _s++) {
            auto m = _s / bnw;
            auto k = _s % bnw;
            out[n - 1 - i - cnt] *= val[(j + m) * area + k * bnl + cnt];
            for (idx_t l = cnt + 1; l < bnl; l++) {
              out[n - 1 - i - l] -=
                  val[(j + m) * area + k * bnl + l] * out[n - 1 - i - cnt];
            }
            ++cnt;
          }
        }
      });
    }
#undef BCSR_TRSV_L_INNER
#undef BCSR_TRSV_U_INNER
  }
  [[nodiscard]] idx_t nrows() const { return BCSR<bnl, bnw, T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return BCSR<bnl, bnw, T, L>::ncols(); }
};

} // namespace trsv

template <class C, spmat::Form S> struct is_trsv<trsv::BJ<C, S>> {
  static constexpr bool value = true;
};

} // namespace kmm

#endif // UMINEKO_SPARSE_TRSV_BJ_HPP
