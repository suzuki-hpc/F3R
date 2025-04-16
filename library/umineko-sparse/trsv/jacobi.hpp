#ifndef UMINEKO_SPARSE_TRSV_JACOBI_HPP
#define UMINEKO_SPARSE_TRSV_JACOBI_HPP

#include "umineko-sparse/matrix/base.hpp"
#include "umineko-sparse/matrix/csr.hpp"
#include "umineko-sparse/matrix/sell.hpp"

#include "umineko-sparse/models.hpp"

namespace kmm {

namespace trsv {

template <class, spmat::Form> struct Jacobi;

namespace l {
template <class C> using Jacobi = Jacobi<C, spmat::Form::lower>;
}
namespace ld {
template <class C> using Jacobi = Jacobi<C, spmat::Form::lower_d>;
}
namespace du {
template <class C> using Jacobi = Jacobi<C, spmat::Form::d_upper>;
}

template <typename T, class L, spmat::Form form>
struct Jacobi<CSR<T, L>, form> : private CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  mutable vector<T, L> tmp;
  struct params {
    int32_t max_ier;
    double omega;
  } param;

  template <typename _>
  explicit Jacobi(const CSR<_, host> &in, params param)
      : CSR<T, L>([&] {
          if constexpr (form == spmat::Form::d_upper)
            return in.duplicate_rotate180().inverse_last();
          else if constexpr (form == spmat::Form::lower_d)
            return in.duplicate_val().inverse_last();
          else
            return in;
        }()),
        tmp(in.nrows()), param(param) {}

  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_TRSV_L_JACOBI_KERNEL(arg1, arg2)                                        \
  exec<L>::para_for(out.size(0), [=, *this, om = param.omega](idx_t i) mutable {    \
    auto t = in[i];                                                                 \
    auto j = rptr[i];                                                               \
    for (; j < rptr[i + 1] - 1; ++j)                                                \
      t -= val[j] * arg1[idx[j]];                                                   \
    arg2[i] = om * t + (1 - om) * arg1[i];                                          \
  })
#define KMM_TRSV_LD_JACOBI_KERNEL(arg1, arg2)                                       \
  exec<L>::para_for(out.size(0), [=, *this, om = param.omega](idx_t i) mutable {    \
    auto t = in[i];                                                                 \
    auto j = rptr[i];                                                               \
    for (; j < rptr[i + 1] - 1; ++j)                                                \
      t -= val[j] * arg1[idx[j]];                                                   \
    arg2[i] = om * t * val[j] + (1 - om) * arg1[i];                                 \
  })
#define KMM_TRSV_DU_JACOBI_KERNEL(arg1, arg2)                                       \
  exec<L>::para_for(                                                                \
      out.size(0), [=, *this, n = out.size(0), om = param.omega](idx_t i) mutable { \
        auto t = in[n - 1 - i];                                                     \
        auto j = rptr[i];                                                           \
        for (; j < rptr[i + 1] - 1; ++j)                                            \
          t -= val[j] * arg1[n - 1 - idx[j]];                                       \
        arg2[n - 1 - i] = om * t * val[j] + (1 - om) * arg1[n - 1 - i];             \
      })

    if constexpr (form == spmat::Form::lower) {
      if (param.max_ier % 2 == 0) {
        exec<L>::para_for(out.size(0),
            [=, om = param.omega](idx_t i) mutable { out[i] = om * in[i]; });
        for (int32_t k = 0; k < param.max_ier; k += 2) {
          KMM_TRSV_L_JACOBI_KERNEL(out, tmp);
          KMM_TRSV_L_JACOBI_KERNEL(tmp, out);
        }
      } else {
        exec<L>::para_for(out.size(0),
            [=, *this, om = param.omega](idx_t i) mutable { tmp[i] = om * in[i]; });
        KMM_TRSV_L_JACOBI_KERNEL(tmp, out);
        for (int32_t k = 1; k < param.max_ier; k += 2) {
          KMM_TRSV_L_JACOBI_KERNEL(out, tmp);
          KMM_TRSV_L_JACOBI_KERNEL(tmp, out);
        }
      }
    } else if constexpr (form == spmat::Form::lower_d) {
      if (param.max_ier % 2 == 0) {
        exec<L>::para_for(
            out.size(0), [=, *this, om = param.omega](idx_t i) mutable {
              out[i] = om * in[i] * val[rptr[i + 1] - 1];
            });
        for (int32_t k = 0; k < param.max_ier; k += 2) {
          KMM_TRSV_LD_JACOBI_KERNEL(out, tmp);
          KMM_TRSV_LD_JACOBI_KERNEL(tmp, out);
        }
      } else {
        exec<L>::para_for(
            out.size(0), [=, *this, om = param.omega](idx_t i) mutable {
              tmp[i] = om * in[i] * val[rptr[i + 1] - 1];
            });
        KMM_TRSV_LD_JACOBI_KERNEL(tmp, out);
        for (int32_t k = 1; k < param.max_ier; k += 2) {
          KMM_TRSV_LD_JACOBI_KERNEL(out, tmp);
          KMM_TRSV_LD_JACOBI_KERNEL(tmp, out);
        }
      }
    } else if constexpr (form == spmat::Form::d_upper) {
      if (param.max_ier % 2 == 0) {
        exec<L>::para_for(
            out.size(0), [=, n = nrows(), *this, om = param.omega](idx_t i) mutable {
              out[n - 1 - i] = om * in[n - 1 - i] * val[rptr[i + 1] - 1];
            });
        for (int32_t k = 0; k < param.max_ier; k += 2) {
          KMM_TRSV_DU_JACOBI_KERNEL(out, tmp);
          KMM_TRSV_DU_JACOBI_KERNEL(tmp, out);
        }
      } else {
        exec<L>::para_for(
            out.size(0), [=, n = nrows(), *this, om = param.omega](idx_t i) mutable {
              tmp[n - 1 - i] = om * in[n - 1 - i] * val[rptr[i + 1] - 1];
            });
        KMM_TRSV_DU_JACOBI_KERNEL(tmp, out);
        for (int32_t k = 1; k < param.max_ier; k += 2) {
          KMM_TRSV_DU_JACOBI_KERNEL(out, tmp);
          KMM_TRSV_DU_JACOBI_KERNEL(tmp, out);
        }
      }
    }

#undef KMM_TRSV_L_JACOBI_KERNEL
#undef KMM_TRSV_LD_JACOBI_KERNEL
#undef KMM_TRSV_DU_JACOBI_KERNEL
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

#if 0
template <typename T, class L>
struct Jacobi<CSR<T, L>, spmat::Form::lower> : CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  mutable vector<T, L> tmp;
  struct params {
    int32_t max_ier;
    double omega;
  } params;

  explicit Jacobi(const CSR<double, host> &in, struct params params)
      : CSR<T, L>(in), tmp(in.nrows()), params(params) {}
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_TRSV_JACOBI_KERNEL(arg1, arg2)                                          \
  exec<L>::para_for(out.size(0), [=, *this, om = params.omega](idx_t i) mutable {   \
    auto t = in[i];                                                                 \
    auto j = rptr[i];                                                               \
    for (; j < rptr[i + 1] - 1; ++j)                                                \
      t -= val[j] * arg1[idx[j]];                                                   \
    arg2[i] = om * t + (1 - om) * arg1[i];                                          \
  })
    if (params.max_ier % 2 == 0) {
      exec<L>::para_for(out.size(0),
          [=, om = params.omega](idx_t i) mutable { out[i] = om * in[i]; });
      for (int32_t k = 0; k < params.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    } else {
      exec<L>::para_for(out.size(0),
          [=, om = params.omega](idx_t i) mutable { tmp[i] = om * in[i]; });
      KMM_TRSV_JACOBI_KERNEL(tmp, out);
      for (int32_t k = 1; k < params.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    }
#undef KMM_TRSV_JACOBI_KERNEL
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

template <typename T, class L>
struct Jacobi<CSR<T, L>, spmat::Form::lower_d> : CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  mutable vector<T, L> tmp;
  struct params {
    int32_t max_ier;
    double omega;
  } params;

  explicit Jacobi(const CSR<double, host> &in, struct params params)
      : CSR<T, L>(in.duplicate()), tmp(in.nrows()), params(params) {
    exec<L>::para_for(nrows(), [=, *this](idx_t i) mutable {
      val[rptr[i + 1] - 1] = 1. / val[rptr[i + 1] - 1];
    });
  }
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_TRSV_JACOBI_KERNEL(arg1, arg2)                                          \
  exec<L>::para_for(out.size(0), [=, *this, om = params.omega](idx_t i) mutable {   \
    auto t = in[i];                                                                 \
    auto j = rptr[i];                                                               \
    for (; j < rptr[i + 1] - 1; ++j)                                                \
      t -= val[j] * arg1[idx[j]];                                                   \
    arg2[i] = om * t * val[j] + (1 - om) * arg1[i];                                 \
  })
    if (params.max_ier % 2 == 0) {
      exec<L>::para_for(out.size(0), [=, *this, om = params.omega](idx_t i) mutable {
        out[i] = om * in[i] * val[rptr[i + 1] - 1];
      });
      for (int32_t k = 0; k < params.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    } else {
      exec<L>::para_for(out.size(0), [=, *this, om = params.omega](idx_t i) mutable {
        tmp[i] = om * in[i] * val[rptr[i + 1] - 1];
      });
      KMM_TRSV_JACOBI_KERNEL(tmp, out);
      for (int32_t k = 1; k < params.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    }
#undef KMM_TRSV_JACOBI_KERNEL
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

template <typename T, class L>
struct Jacobi<CSR<T, L>, spmat::Form::d_upper> : CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  mutable vector<T, L> tmp;
  struct params {
    int32_t max_ier;
    double omega;
  } params;

  explicit Jacobi(const CSR<double, host> &in, struct params params)
      : CSR<T, L>(in.duplicate_rotate180()), tmp(in.nrows()), params(params) {
    exec<L>::para_for(nrows(), [=, *this](idx_t i) mutable {
      val[rptr[i + 1] - 1] = 1. / val[rptr[i + 1] - 1];
    });
  }
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_TRSV_JACOBI_KERNEL(arg1, arg2)                                          \
  exec<L>::para_for(out.size(0),                                                    \
      [=, *this, n = out.size(0), om = params.omega](idx_t i) mutable {             \
        auto t = in[n - 1 - i];                                                     \
        auto j = rptr[i];                                                           \
        for (; j < rptr[i + 1] - 1; ++j)                                            \
          t -= val[j] * arg1[n - 1 - idx[j]];                                       \
        arg2[n - 1 - i] = om * t * val[j] + (1 - om) * arg1[n - 1 - i];             \
      });
    if (params.max_ier % 2 == 0) {
      exec<L>::para_for(
          out.size(0), [=, n = nrows(), *this, om = params.omega](idx_t i) mutable {
            out[n - 1 - i] = om * in[n - 1 - i] * val[rptr[i + 1] - 1];
          });
      for (int32_t k = 0; k < params.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    } else {
      exec<L>::para_for(
          out.size(0), [=, n = nrows(), *this, om = params.omega](idx_t i) mutable {
            tmp[n - 1 - i] = om * in[n - 1 - i] * val[rptr[i + 1] - 1];
          });
      KMM_TRSV_JACOBI_KERNEL(tmp, out);
      for (int32_t k = 1; k < params.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    }
#undef KMM_TRSV_JACOBI_KERNEL
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};
#endif

template <uint16_t w, typename T, class L>
struct Jacobi<SELL<w, T, L>, spmat::Form::lower>
    : SELL<w, T, L, impl::sell_align::right> {
  using sell_t = SELL<w, T, L, impl::sell_align::right>;
  using sell_t::val, sell_t::idx, sell_t::sptr;
  mutable vector<T, L> tmp;
  struct params {
    int32_t max_ier;
    double omega;
  } param;

  explicit Jacobi(const CSR<double, host> &in, params param)
      : sell_t(in), tmp(in.nrows()), param(param) {}
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_TRSV_JACOBI_KERNEL(arg1, arg2)                                          \
  exec<L>::para_for(                                                                \
      out.size(0), [=, *this, n = out.size(0), om = param.omega](idx_t i) mutable { \
        auto sid = i / w;                                                           \
        auto myid = i % w;                                                          \
        auto s = sptr[sid];                                                         \
        auto size = sptr[sid + 1] - s;                                              \
        uint16_t t_chunk = (sid != n / w) ? w : n % w;                              \
        auto t = in[i];                                                             \
        auto offset = s * w;                                                        \
        for (idx_t j = 0; j < size - 1; j++, offset += t_chunk)                     \
          t -= val[offset + myid] * arg1[idx[offset + myid]];                       \
        arg2[i] = om * t + (1 - om) * arg1[i];                                      \
      });
    if (param.max_ier % 2 == 0) {
      exec<L>::para_for(out.size(0),
          [=, om = param.omega](idx_t i) mutable { out[i] = om * in[i]; });
      for (int32_t k = 0; k < param.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    } else {
      exec<L>::para_for(out.size(0),
          [=, *this, om = param.omega](idx_t i) mutable { tmp[i] = om * in[i]; });
      KMM_TRSV_JACOBI_KERNEL(tmp, out);
      for (int32_t k = 1; k < param.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    }
#undef KMM_TRSV_JACOBI_KERNEL
  }
  [[nodiscard]] idx_t nrows() const { return sell_t::nrows(); }
  [[nodiscard]] idx_t ncols() const { return sell_t::ncols(); }
};

template <uint16_t w, typename T, class L>
struct Jacobi<SELL<w, T, L>, spmat::Form::lower_d>
    : SELL<w, T, L, impl::sell_align::right> {
  using sell_t = SELL<w, T, L, impl::sell_align::right>;
  using sell_t::val, sell_t::idx, sell_t::sptr;
  mutable vector<T, L> tmp;
  struct params {
    int32_t max_ier;
    double omega;
  } param;

  explicit Jacobi(const CSR<double, host> &in, params param)
      : sell_t(in.duplicate()), tmp(in.nrows()), param(param) {
    exec<L>::para_for(nrows(), [=, *this, n = nrows()](idx_t i) mutable {
      auto sid = i / w;
      auto myid = i % w;
      auto s = sptr[sid];
      auto size = sptr[sid + 1] - s;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto offset = s * w + (size - 1) * t_chunk + myid;
      val[offset] = 1. / val[offset];
    });
  }

  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_TRSV_JACOBI_KERNEL(arg1, arg2)                                          \
  exec<L>::para_for(                                                                \
      out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {     \
        auto sid = i / w;                                                           \
        auto myid = i % w;                                                          \
        auto s = sptr[sid];                                                         \
        auto size = sptr[sid + 1] - s;                                              \
        uint16_t t_chunk = (sid != n / w) ? w : n % w;                              \
        auto t = in[i];                                                             \
        auto offset = s * w + myid;                                                 \
        for (idx_t j = 0; j < size - 1; ++j, offset += t_chunk)                     \
          t -= val[offset] * arg1[idx[offset]];                                     \
        arg2[i] = om * t * val[offset] + (1 - om) * arg1[i];                        \
      });
    if (param.max_ier % 2 == 0) {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto offset = s * w + (size - 1) * t_chunk + myid;
            out[i] = om * in[i] * val[offset];
          });
      for (int32_t k = 0; k < param.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    } else {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto offset = s * w + (size - 1) * t_chunk + myid;
            tmp[i] = om * in[i] * val[offset];
          });
      KMM_TRSV_JACOBI_KERNEL(tmp, out);
      for (int32_t k = 1; k < param.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    }
#undef KMM_TRSV_JACOBI_KERNEL
  }
  [[nodiscard]] idx_t nrows() const { return sell_t::nrows(); }
  [[nodiscard]] idx_t ncols() const { return sell_t::ncols(); }
};

template <uint16_t w, typename T, class L>
struct Jacobi<SELL<w, T, L>, spmat::Form::d_upper>
    : SELL<w, T, L, impl::sell_align::right> {
  using sell_t = SELL<w, T, L, impl::sell_align::right>;
  using sell_t::val, sell_t::idx, sell_t::sptr;
  mutable vector<T, L> tmp;
  struct params {
    int32_t max_ier;
    double omega;
  } param;

  explicit Jacobi(const CSR<double, host> &in, params param)
      : sell_t(in.duplicate_rotate180()), tmp(in.nrows()), param(param) {
    exec<L>::para_for(nrows(), [=, *this, n = nrows()](idx_t i) mutable {
      auto sid = i / w;
      auto myid = i % w;
      auto s = sptr[sid];
      auto size = sptr[sid + 1] - s;
      uint16_t t_chunk = (sid != n / w) ? w : n % w;
      auto offset = s * w + (size - 1) * t_chunk + myid;
      val[offset] = 1. / val[offset];
    });
  }

  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
#define KMM_TRSV_JACOBI_KERNEL(arg1, arg2)                                          \
  exec<L>::para_for(                                                                \
      out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {     \
        auto sid = i / w;                                                           \
        auto myid = i % w;                                                          \
        auto s = sptr[sid];                                                         \
        auto size = sptr[sid + 1] - s;                                              \
        uint16_t t_chunk = (sid != n / w) ? w : n % w;                              \
        auto t = in[n - 1 - i];                                                     \
        auto offset = s * w + myid;                                                 \
        for (idx_t j = 0; j < size - 1; ++j, offset += t_chunk)                     \
          t -= val[offset] * arg1[n - 1 - idx[offset]];                             \
        arg2[n - 1 - i] = om * t * val[offset] + (1 - om) * arg1[n - 1 - i];        \
      });
    if (param.max_ier % 2 == 0) {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto offset = s * w + (size - 1) * t_chunk + myid;
            out[n - 1 - i] = om * in[n - 1 - i] * val[offset];
          });
      for (int32_t k = 0; k < param.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    } else {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto offset = s * w + (size - 1) * t_chunk + myid;
            tmp[n - 1 - i] = om * in[n - 1 - i] * val[offset];
          });
      KMM_TRSV_JACOBI_KERNEL(tmp, out);
      for (int32_t k = 1; k < param.max_ier; k += 2) {
        KMM_TRSV_JACOBI_KERNEL(out, tmp);
        KMM_TRSV_JACOBI_KERNEL(tmp, out);
      }
    }
#undef KMM_TRSV_JACOBI_KERNEL
  }
  [[nodiscard]] idx_t nrows() const { return sell_t::nrows(); }
  [[nodiscard]] idx_t ncols() const { return sell_t::ncols(); }
};

} // namespace trsv

template <class C, spmat::Form S> struct is_trsv<trsv::Jacobi<C, S>> {
  static constexpr bool value = true;
};

} // namespace kmm

#endif // UMINEKO_SPARSE_TRSV_JACOBI_HPP
