#ifndef UMINEKO_SPARSE_TRSV_BAJ_HPP
#define UMINEKO_SPARSE_TRSV_BAJ_HPP

#include "umineko-sparse/matrix/base.hpp"
#include "umineko-sparse/matrix/csr.hpp"
#include "umineko-sparse/matrix/sell.hpp"

#include "umineko-sparse/models.hpp"

namespace kmm {

namespace trsv {

template <class, spmat::Form> struct BAJ;

namespace l {
template <class C> using BAJ = BAJ<C, spmat::Form::lower>;
}
namespace ld {
template <class C> using BAJ = BAJ<C, spmat::Form::lower_d>;
}
namespace du {
template <class C> using BAJ = BAJ<C, spmat::Form::d_upper>;
}

template <typename T, class L, spmat::Form form>
struct BAJ<CSR<T, L>, form> : private CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  struct params {
    int32_t max_iter;
    double omega;
  } param;

  explicit BAJ(const CSR<double, host> &in, params param)
      : CSR<T, L>([&] {
          if constexpr (form == spmat::Form::d_upper)
            return in.duplicate_rotate180().inverse_last();
          else if constexpr (form == spmat::Form::lower_d)
            return in.duplicate_val().inverse_last();
          else
            return in;
        }()),
        param(param) {}

  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
    if constexpr (form == spmat::Form::lower) {
      exec<L>::para_for(
          out.size(0), [=, om = param.omega](idx_t i) mutable { out[i] = in[i]; });
      for (int32_t k = 1; k < param.max_iter; ++k) {
        exec<L>::para_for(
            out.size(0), [=, *this, om = param.omega](idx_t i) mutable {
              auto t = in[i];
              auto j = rptr[i];
              for (; j < rptr[i + 1] - 1; ++j)
                t -= val[j] * out[idx[j]];
              out[i] = t;
            });
      }
    } else if constexpr (form == spmat::Form::lower_d) {
      exec<L>::para_for(out.size(0), [=, *this, om = param.omega](idx_t i) mutable {
        out[i] = in[i] * val[rptr[i + 1] - 1];
      });
      for (int32_t k = 1; k < param.max_iter; ++k) {
        exec<L>::para_for(
            out.size(0), [=, *this, om = param.omega](idx_t i) mutable {
              auto t = in[i];
              auto j = rptr[i];
              for (; j < rptr[i + 1] - 1; ++j)
                t -= val[j] * out[idx[j]];
              out[i] = t * val[j];
            });
      }
    } else if constexpr (form == spmat::Form::d_upper) {
      exec<L>::para_for(
          out.size(0), [=, n = nrows(), *this, om = param.omega](idx_t i) mutable {
            out[n - 1 - i] = in[n - 1 - i] * val[rptr[i + 1] - 1];
          });
      for (int32_t k = 1; k < param.max_iter; ++k) {
        exec<L>::para_for(out.size(0),
            [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
              auto t = in[n - 1 - i];
              auto j = rptr[i];
              for (; j < rptr[i + 1] - 1; ++j)
                t -= val[j] * out[n - 1 - idx[j]];
              out[n - 1 - i] = t * val[j];
            });
      }
    }
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

template <uint16_t w, typename T, class L, spmat::Form form>
struct BAJ<SELL<w, T, L>, form> : SELL<w, T, L, impl::sell_align::right> {
  using sell_t = SELL<w, T, L, impl::sell_align::right>;
  using sell_t::val, sell_t::idx, sell_t::sptr;
  struct params {
    int32_t max_iter;
    double omega;
  } param;

  explicit BAJ(const CSR<double, host> &in, params param)
      : sell_t([&] {
          if constexpr (form == spmat::Form::d_upper)
            return in.duplicate_rotate180().inverse_last();
          else if constexpr (form == spmat::Form::lower_d)
            return in.duplicate_val().inverse_last();
          else
            return in;
        }()),
        param(param) {}

  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
    if constexpr (form == spmat::Form::lower) {
      exec<L>::para_for(
          out.size(0), [=, om = param.omega](idx_t i) mutable { out[i] = in[i]; });
      for (int32_t k = 1; k < param.max_iter; ++k) {
        exec<L>::para_for(out.size(0),
            [=, *this, n = out.size(0), om = param.omega](idx_t i) mutable {
              auto sid = i / w;
              auto myid = i % w;
              auto s = sptr[sid];
              auto size = sptr[sid + 1] - s;
              uint16_t t_chunk = (sid != n / w) ? w : n % w;
              auto t = in[i];
              auto offset = s * w;
              for (idx_t j = 0; j < size - 1; j++, offset += t_chunk)
                t -= val[offset + myid] * out[idx[offset + myid]];
              out[i] = t;
            });
      }
    } else if constexpr (form == spmat::Form::lower_d) {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto offset = s * w + (size - 1) * t_chunk + myid;
            out[i] = in[i] * val[offset];
          });
      for (int32_t k = 1; k < param.max_iter; ++k) {
        exec<L>::para_for(out.size(0),
            [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
              auto sid = i / w;
              auto myid = i % w;
              auto s = sptr[sid];
              auto size = sptr[sid + 1] - s;
              uint16_t t_chunk = (sid != n / w) ? w : n % w;
              auto t = in[i];
              auto offset = s * w + myid;
              for (idx_t j = 0; j < size - 1; ++j, offset += t_chunk)
                t -= val[offset] * out[idx[offset]];
              out[i] = t * val[offset];
            });
      }
    } else if constexpr (form == spmat::Form::d_upper) {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto offset = s * w + (size - 1) * t_chunk + myid;
            out[i] = in[i] * val[offset];
          });
      for (int32_t k = 1; k < param.max_iter; ++k) {
        exec<L>::para_for(out.size(0),
            [=, *this, n = nrows(), om = param.omega](idx_t i) mutable {
              auto sid = i / w;
              auto myid = i % w;
              auto s = sptr[sid];
              auto size = sptr[sid + 1] - s;
              uint16_t t_chunk = (sid != n / w) ? w : n % w;
              auto t = in[n - 1 - i];
              auto offset = s * w + myid;
              for (idx_t j = 0; j < size - 1; ++j, offset += t_chunk)
                t -= val[offset] * out[n - 1 - idx[offset]];
              out[n - 1 - i] = t * val[offset];
            });
      }
    }
  }
  [[nodiscard]] idx_t nrows() const { return sell_t::nrows(); }
  [[nodiscard]] idx_t ncols() const { return sell_t::ncols(); }
};

#if 0

template <typename T, class L>
struct BAJ<CSR<T, L>, spmat::Form::lower> : CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  struct params {
    int32_t max_iter;
    double omega;
  } params;

  explicit BAJ(const CSR<double, host> &in, struct params params)
      : CSR<T, L>(in), params(params) {}
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(
        out.size(0), [=, om = params.omega](idx_t i) mutable { out[i] = in[i]; });
    for (int32_t k = 1; k < params.max_iter; ++k) {
      exec<L>::para_for(out.size(0), [=, *this, om = params.omega](idx_t i) mutable {
        auto t = in[i];
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[idx[j]];
        out[i] = t;
      });
    }
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

template <typename T, class L>
struct BAJ<CSR<T, L>, spmat::Form::lower_d> : CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  struct params {
    int32_t max_iter;
    double omega;
  } params;

  explicit BAJ(const CSR<double, host> &in, struct params params)
      : CSR<T, L>(in.duplicate()), params(params) {
    exec<L>::para_for(nrows(), [=, *this](idx_t i) mutable {
      val[rptr[i + 1] - 1] = 1. / val[rptr[i + 1] - 1];
    });
  }
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(out.size(0), [=, *this, om = params.omega](idx_t i) mutable {
      out[i] = in[i] * val[rptr[i + 1] - 1];
    });
    for (int32_t k = 1; k < params.max_iter; ++k) {
      exec<L>::para_for(out.size(0), [=, *this, om = params.omega](idx_t i) mutable {
        auto t = in[i];
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[idx[j]];
        out[i] = t * val[j];
      });
    }
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

template <typename T, class L>
struct BAJ<CSR<T, L>, spmat::Form::d_upper> : CSR<T, L> {
  using CSR<T, L>::val, CSR<T, L>::idx, CSR<T, L>::rptr;
  struct params {
    int32_t max_iter;
    double omega;
  } params;

  explicit BAJ(const CSR<double, host> &in, struct params params)
      : CSR<T, L>(in.duplicate_rotate180()), params(params) {
    exec<L>::para_for(nrows(), [=, *this](idx_t i) mutable {
      val[rptr[i + 1] - 1] = 1. / val[rptr[i + 1] - 1];
    });
  }
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(
        out.size(0), [=, n = nrows(), *this, om = params.omega](idx_t i) mutable {
          out[n - 1 - i] = in[n - 1 - i] * val[rptr[i + 1] - 1];
        });
    for (int32_t k = 1; k < params.max_iter; ++k) {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = params.omega](idx_t i) mutable {
            auto t = in[n - 1 - i];
            auto j = rptr[i];
            for (; j < rptr[i + 1] - 1; ++j)
              t -= val[j] * out[n - 1 - idx[j]];
            out[n - 1 - i] = t * val[j];
          });
    }
  }
  [[nodiscard]] idx_t nrows() const { return CSR<T, L>::nrows(); }
  [[nodiscard]] idx_t ncols() const { return CSR<T, L>::ncols(); }
};

#endif

#if 0

template <uint16_t w, typename T, class L>
struct BAJ<SELL<w, T, L>, spmat::Form::lower>
    : SELL<w, T, L, impl::sell_align::right> {
  using sell_t = SELL<w, T, L, impl::sell_align::right>;
  using sell_t::val, sell_t::idx, sell_t::sptr;
  struct params {
    int32_t max_iter;
    double omega;
  } params;

  explicit BAJ(const CSR<double, host> &in, struct params params)
      : sell_t(in), params(params) {}
  template <typename in_t, typename out_t>
  void solve(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(
        out.size(0), [=, om = params.omega](idx_t i) mutable { out[i] = in[i]; });
    for (int32_t k = 1; k < params.max_iter; ++k) {
      exec<L>::para_for(out.size(0),
          [=, *this, n = out.size(0), om = params.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto t = in[i];
            auto offset = s * w;
            for (idx_t j = 0; j < size - 1; j++, offset += t_chunk)
              t -= val[offset + myid] * out[idx[offset + myid]];
            out[i] = t;
          });
    }
  }
  [[nodiscard]] idx_t nrows() const { return sell_t::nrows(); }
  [[nodiscard]] idx_t ncols() const { return sell_t::ncols(); }
};

template <uint16_t w, typename T, class L>
struct BAJ<SELL<w, T, L>, spmat::Form::lower_d>
    : SELL<w, T, L, impl::sell_align::right> {
  using sell_t = SELL<w, T, L, impl::sell_align::right>;
  using sell_t::val, sell_t::idx, sell_t::sptr;
  struct params {
    int32_t max_iter;
    double omega;
  } params;

  explicit BAJ(const CSR<double, host> &in, struct params params)
      : sell_t(in.duplicate()), params(params) {
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
    exec<L>::para_for(
        out.size(0), [=, *this, n = nrows(), om = params.omega](idx_t i) mutable {
          auto sid = i / w;
          auto myid = i % w;
          auto s = sptr[sid];
          auto size = sptr[sid + 1] - s;
          uint16_t t_chunk = (sid != n / w) ? w : n % w;
          auto offset = s * w + (size - 1) * t_chunk + myid;
          out[i] = in[i] * val[offset];
        });
    for (int32_t k = 1; k < params.max_iter; ++k) {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = params.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto t = in[i];
            auto offset = s * w + myid;
            for (idx_t j = 0; j < size - 1; ++j, offset += t_chunk)
              t -= val[offset] * out[idx[offset]];
            out[i] = t * val[offset];
          });
    }
  }
  [[nodiscard]] idx_t nrows() const { return sell_t::nrows(); }
  [[nodiscard]] idx_t ncols() const { return sell_t::ncols(); }
};

template <uint16_t w, typename T, class L>
struct BAJ<SELL<w, T, L>, spmat::Form::d_upper>
    : SELL<w, T, L, impl::sell_align::right> {
  using sell_t = SELL<w, T, L, impl::sell_align::right>;
  using sell_t::val, sell_t::idx, sell_t::sptr;
  struct params {
    int32_t max_iter;
    double omega;
  } params;

  explicit BAJ(const CSR<double, host> &in, struct params params)
      : sell_t(in.duplicate_rotate180()), params(params) {
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
    exec<L>::para_for(
        out.size(0), [=, *this, n = nrows(), om = params.omega](idx_t i) mutable {
          auto sid = i / w;
          auto myid = i % w;
          auto s = sptr[sid];
          auto size = sptr[sid + 1] - s;
          uint16_t t_chunk = (sid != n / w) ? w : n % w;
          auto offset = s * w + (size - 1) * t_chunk + myid;
          out[i] = in[i] * val[offset];
        });
    for (int32_t k = 1; k < params.max_iter; ++k) {
      exec<L>::para_for(
          out.size(0), [=, *this, n = nrows(), om = params.omega](idx_t i) mutable {
            auto sid = i / w;
            auto myid = i % w;
            auto s = sptr[sid];
            auto size = sptr[sid + 1] - s;
            uint16_t t_chunk = (sid != n / w) ? w : n % w;
            auto t = in[n - 1 - i];
            auto offset = s * w + myid;
            for (idx_t j = 0; j < size - 1; ++j, offset += t_chunk)
              t -= val[offset] * out[n - 1 - idx[offset]];
            out[n - 1 - i] = t * val[offset];
          });
    }
  }
  [[nodiscard]] idx_t nrows() const { return sell_t::nrows(); }
  [[nodiscard]] idx_t ncols() const { return sell_t::ncols(); }
};

#endif

} // namespace trsv

template <class C, spmat::Form S> struct is_trsv<trsv::BAJ<C, S>> {
  static constexpr bool value = true;
};

} // namespace kmm

#endif // UMINEKO_SPARSE_TRSV_BAJ_HPP
