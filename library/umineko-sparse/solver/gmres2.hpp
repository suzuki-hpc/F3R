#ifndef UMINEKO_SPARSE_SOLVER_GMRES_HPP
#define UMINEKO_SPARSE_SOLVER_GMRES_HPP

#include "umineko-sparse/models.hpp"

#if 0 // __has_include("umineko-type/fixed.hpp")
#include "umineko-type/fixed.hpp"
#endif

namespace kmm {

template <typename T, class L> struct Blas1 {
  //! Rotates input values assuming that the final value is 0.
  static void final_grot(
      const vector<T, L> &c, const vector<T, L> &s, vector<T, L> e, int k) {
    auto kernel = [c = c.raw(), s = s.raw(), e = e.raw(), k = k] {
      e[k + 1] = -s[k] * e[k];
      e[k] = c[k] * e[k];
    };
    exec<L>::seq(kernel);
  }

  //! Rotates input values and generates Givens pair for the last values.
  static void full_grot(vector<T, L> &c, vector<T, L> &s, vector<T, L> H, int k) {
    auto kernel = [=, c = c.raw(), s = s.raw(), H = H.raw()] {
      for (int i = 0; i < k; i++) {
        auto t = H[i];
        H[i] = c[i] * t + s[i] * H[i + 1];
        H[i + 1] = -s[i] * t + c[i] * H[i + 1];
      }
      auto t = sqrt(H[k] * H[k] + H[k + 1] * H[k + 1]);
      c[k] = H[k] / t;
      s[k] = H[k + 1] / t;
      H[k] = t;
      H[k + 1] = 0;
    };
    exec<L>::seq(kernel);
  }

  //! Solves a triangular linear system.
  static void trsv(const matrix<T, L> &U, const vector<T, L> &b, vector<T, L> x,
      const int h, const int w) {
    auto kernel = [=, U = U.raw(), b = b.raw(), x = x.raw()] {
      for (int i = w - 1; i >= 0; i--) {
        T t = b[i];
        for (int j = w - 1; j > i; j--) {
          t -= U[j * h + i] * x[j];
        }
        x[i] = t / U[i * h + i];
      }
    };
    exec<L>::seq(kernel);
  }
};

template <typename T, class L> struct FGMRES {
  const CoefficientMatrix<L> A;
  const Operator<L> M;
  int32_t max_iter;
  bool normalize;
  mutable matrix<T, L> H, V, Z;
  mutable vector<T, L> c, s, y, e;
  mutable scalar<T, L> h;
  mutable scalar<T, host> h_nrm_r;

  mutable vector<T, L> act;

  struct params {
    int32_t max_iter;
    bool normalize;
  };

  FGMRES(const CoefficientMatrix<L> A, const Operator<L> M, params param)
      : A(A), M(M), max_iter(param.max_iter), normalize(param.normalize),
        H(max_iter + 1, max_iter), V(A.nrows(), max_iter + 1),
        Z(A.nrows(), max_iter), c(max_iter), s(max_iter), y(max_iter),
        e(max_iter + 1), act(A.nrows()) {
    impl::pool<T, L>::init(A.nrows());
  }

  [[nodiscard]] idx_t nrows() const { return A.nrows(); }
  [[nodiscard]] idx_t ncols() const { return A.ncols(); }

  template <typename val1_t, typename val2_t>
  solve_res_t solve(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    A.compute_residual(rhs, x, act);
    return kernel<false>(act, x, cond);
  }

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    kernel<true>(in, out, nullptr);
  }

  template <bool with_init, typename val1_t, typename val2_t>
  solve_res_t kernel(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    bool flag = false;
    (normalize) ? h = nrm(rhs) : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (normalize) ? V(0) = h.inv() * rhs : V(0).copy(rhs);
    if (cond && cond(0, abs(h_nrm_r[0]))) {
      if constexpr (with_init)
        x.fill(val2_t(0.0));
      return solve_res_t{true, 0, abs(h_nrm_r[0])};
    }
    int j = 0;
    for (; j < max_iter; j++) {
      M.operate(V(j), Z(j));
      A.operate(Z(j), V(j + 1));
      for (int k = 0; k <= j; k++) {
        H(j, k).copy(h = dot(V(j + 1), V(k)));
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j, j + 1).copy(h = nrm(V(j + 1)));
      V(j + 1) *= h.inv();
      Blas1<T, L>::full_grot(c, s, H(j), j);
      Blas1<T, L>::final_grot(c, s, e, j);
      h_nrm_r.copy(e(j + 1));
      if (cond && cond(j + 1, abs(h_nrm_r[0]))) {
        j++;
        flag = true;
        break;
      }
    }
    Blas1<T, L>::trsv(H, e, y, max_iter + 1, j);
    if constexpr (with_init)
      x = Z * y.slice(j);
    else
      x += Z * y.slice(j);
    return flag ? solve_res_t{true, j, abs(h_nrm_r[0])}
                : solve_res_t{false, max_iter, abs(h_nrm_r[0])};
  }
};

template <typename T, class L> struct sFGMRES {
  const CoefficientMatrix<L> A;
  const Operator<L> M;
  int32_t max_iter;
  bool normalize;
  mutable matrix<T, L> H, V, Z;
  mutable vector<T, L> c, s, y, e;
  mutable scalar<T, L> h;
  mutable scalar<T, host> h_nrm_r;

  mutable vector<T, L> act;

  struct params {
    int32_t max_iter;
    bool normalize;
  };

  sFGMRES(const CoefficientMatrix<L> A, const Operator<L> M, params param)
      : A(A), M(M), max_iter(param.max_iter), normalize(param.normalize),
        H(max_iter + 1, max_iter), V(A.nrows(), max_iter + 1),
        Z(A.nrows(), max_iter), c(max_iter), s(max_iter), y(max_iter),
        e(max_iter + 1), act(A.nrows()) {
    impl::pool<T, L>::init(A.nrows());
  }

  [[nodiscard]] idx_t nrows() const { return A.nrows(); }
  [[nodiscard]] idx_t ncols() const { return A.ncols(); }

  template <typename val1_t, typename val2_t>
  solve_res_t solve(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    A.compute_residual(rhs, x, act);
    return kernel<false>(act, x, cond);
  }

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    kernel<true>(in, out, nullptr);
  }

  template <bool with_init, typename val1_t, typename val2_t>
  solve_res_t kernel(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    bool flag = false;
    (normalize) ? h = nrm(rhs) : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (normalize) ? V(0) = h.inv() * rhs : V(0).copy(rhs);
    if (cond && cond(0, std::abs(h_nrm_r[0]))) {
      if constexpr (with_init)
        return x.fill(val2_t(0.0)), solve_res_t{true, 0, std::abs(h_nrm_r[0])};
      else
        return solve_res_t{true, 0, std::abs(h_nrm_r[0])};
    }
    for (int j = 0; j < max_iter; j++) {
      M.operate(V(j), Z(j));
      A.operate(Z(j), V(j + 1));
    }
    for (int j = 0; j < max_iter; j++) {
      for (int k = 0; k <= j; k++) {
        H(j, k).copy(h = dot(V(j + 1), V(k)));
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j, j + 1).copy(h = nrm(V(j + 1)));
      V(j + 1) *= h.inv();
    }
    int j = 0;
    for (; j < max_iter; j++) {
      Blas1<T, L>::full_grot(c, s, H(j), j);
      Blas1<T, L>::final_grot(c, s, e, j);
      h_nrm_r.copy(e(j + 1));
      if (cond && cond(j + 1, std::abs(h_nrm_r[0]))) {
        // j++;
        // flag = true;
        // break;
      }
    }
    Blas1<T, L>::trsv(H, e, y, max_iter + 1, j);
    if constexpr (with_init)
      x = Z * y.slice(j);
    else
      x += Z * y.slice(j);
    return flag ? solve_res_t{true, j, std::abs(h_nrm_r[0])}
                : solve_res_t{false, max_iter, std::abs(h_nrm_r[0])};
  }
};

#if 0 // __has_include("umineko-type/fixed.hpp")

template <class L, uint32_t label> struct Blas1<fixed32_t<label>, L> {
  using upper_t = typename fixed32_t<label>::template _mid_fixed_t<label>;
  static void final_grot(const vector<upper_t, L> &c, const vector<upper_t, L> &s,
      vector<upper_t, L> e, int k) {
    exec<L>::seq([=, c = c.raw(), s = s.raw(), e = e.raw(), bit = upper_t::bit] {
      e[k + 1].val = -s[k].val * e[k].val >> bit;
      e[k].val = c[k].val * e[k].val >> bit;
    });
  }

  //! Rotates input values and generates Givens pair for the last values.
  static void full_grot(
      vector<upper_t, L> &c, vector<upper_t, L> &s, vector<upper_t, L> H, int k) {
    uint8_t bit = upper_t::bit;
    uint8_t bit62 = 62 - bit - bit;
    exec<L>::seq(
        [=, c = c.raw(), s = s.raw(), H = H.raw(), bit = bit, bit62 = bit62] {
          for (int i = 0; i < k; i++) {
            int64_t t1 = H[i].val;
            int64_t t2 = H[i + 1].val;
            H[i].val = int64_t(c[i].val * t1 + s[i].val * t2) >> bit;
            H[i + 1].val = int64_t(-s[i].val * t1 + c[i].val * t2) >> bit;
          }
          int64_t t = sqrt(H[k].val * H[k].val + H[k + 1].val * H[k + 1].val);
          int64_t inv_t = (int64_t(1) << 62) / t >> bit62;
          c[k].val = H[k].val * inv_t >> bit;
          s[k].val = H[k + 1].val * inv_t >> bit;
          H[k].val = t;
          H[k + 1].val = 0;
        });
  }

  //! Solves a triangular linear system.
  static void trsv(const matrix<upper_t, L> &U, const double b_coef,
      const vector<upper_t, L> &b, vector<double, L> x, const int h, const int w) {
    auto kernel = [=, U = U.raw(), b = b.raw(), x = x.raw(), bit = upper_t::bit] {
      const double fact = double(int64_t(1) << bit);
      for (int i = w - 1; i >= 0; i--) {
        double t = b_coef * double(b[i].val) / fact;
        for (int j = w - 1; j > i; j--) {
          t -= double(U[j * h + i].val) / fact * x[j];
        }
        x[i] = t / (double(U[i * h + i].val) / fact);
      }
    };
    exec<L>::seq(kernel);
  }
};

template <typename T> T d2h(const scalar<T, device> &in) {
  auto res = scalar<T, host>(in);
  return res[0];
}

template <uint32_t label, class L> struct FGMRES<fixed32_t<label>, L> {
  using upper_t = typename fixed32_t<label>::template _mid_fixed_t<label>;
  const IntOperator<L, label> A;
  const IntOperator<L, label> M;
  int32_t max_iter;
  bool normalize;
  mutable matrix<upper_t, L> H;
  mutable matrix<fixed32_t<label>, L> V, Z;
  mutable vector<upper_t, L> c, s, e;
  mutable vector<double, L> y;
  mutable scalar<upper_t, L> h;
  mutable scalar<double, L> dh;
  mutable scalar<double, host> h_nrm_r;
  mutable scalar<upper_t, host> h_e;

  struct params {
    int32_t max_iter;
    bool normalize;
  };

  FGMRES(const IntOperator<L, label> A, const IntOperator<L, label> M, params param)
      : A(A), M(M), max_iter(param.max_iter), normalize(param.normalize),
        H(max_iter + 1, max_iter), V(A.nrows(), max_iter + 1),
        Z(A.nrows(), max_iter), c(max_iter), s(max_iter), e(max_iter + 1),
        y(max_iter) {
    impl::pool<upper_t, L>::init(A.nrows());
  }

  [[nodiscard]] idx_t nrows() const { return A.nrows(); }
  [[nodiscard]] idx_t ncols() const { return A.ncols(); }

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    kernel<true>(in, out, nullptr);
  }

  template <bool with_init, typename val1_t, typename val2_t>
  solve_res_t kernel(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    bool flag = false;
    (normalize) ? dh = nrm(rhs) : dh.fill(1.0);
    h_nrm_r.copy(dh);
    double e_coef = h_nrm_r[0];
    e(0) = 1.0;
    (normalize) ? V(0) = dh.inv() * rhs : V(0).copy(rhs);
    if (cond && cond(0, h_nrm_r[0]))
      return x.fill(val2_t(0.0)), solve_res_t{true, 0, h_nrm_r[0]};
    int j = 0;
    for (; j < max_iter; j++) {
      M.operate(V(j), Z(j));
      A.operate(Z(j), V(j + 1));
      for (int k = 0; k <= j; k++) {
        H(j, k) = dot(V(k), V(j + 1));
        exec<L>::seq([ptr = &H(j, k)[0].val]() { *ptr >>= fixed32_t<label>::bit; });
      }

      {
        // V(j + 1) -= V * H(j).slice(j + 1);
        exec<L>::para_for(x.size(0), [=, n = A.nrows(), vj1 = V(j + 1).raw(),
                                         V = V.raw(), h = H(j).raw()](idx_t i) {
          int64_t t = V[i].val * h[0].val;
          for (idx_t k = 1; k < j + 1; ++k) {
            t += V[k * n + i].val * h[k].val;
          }
          vj1[i].val -= t >> fixed32_t<label>::bit;
        });
      }

      H(j, j + 1) = nrm(V(j + 1));
      {
        // V(j + 1) *= h.inv();
        scalar<double, L> one;
        exec<L>::seq(
            [=, h = h.raw(), H = H(j, j + 1).raw()]() { h[0] = 1. / H[0]; });
        exec<L>::para_for(A.nrows(), [=, vj1 = V(j + 1).raw(), h = h.raw()](auto i) {
          vj1[i].val = h[0].val * vj1[i].val >> fixed32_t<label>::bit;
        });
      }
      Blas1<fixed32_t<label>, L>::full_grot(c, s, H(j), j);
      Blas1<fixed32_t<label>, L>::final_grot(c, s, e, j);

      // if (cond && cond(j + 1, h_nrm_r[0])) {
      if (cond && cond(j + 1, e_coef * std::abs(double(h_e.copy(e(j + 1))[0])))) {
        j++;
        flag = true;
        break;
      }
    }
    Blas1<fixed32_t<label>, L>::trsv(H, e_coef, e, y, max_iter + 1, j);

    // if constexpr (with_init)
    //   x = Z * y.slice(j);
    // else
    //   x += Z * y.slice(j);

    if constexpr (with_init) {
      double fact = int64_t(1) << fixed32_t<label>::bit;
      exec<L>::para_for(x.size(0),
          [=, n = A.nrows(), x = x.raw(), Z = Z.raw(), y = y.raw()](idx_t i) {
            auto t = double(Z[i].val) / fact * y[0];
            for (idx_t k = 1; k < j; ++k) {
              t += double(Z[k * n + i].val) / fact * y[k];
            }
            x[i] = t;
          });
    } else {
      double fact = int64_t(1) << fixed32_t<label>::bit;
      exec<L>::para_for(x.size(0),
          [=, n = A.nrows(), x = x.raw(), Z = Z.raw(), y = y.raw()](idx_t i) {
            auto t = double(Z[i].val) / fact * y[0];
            for (idx_t k = 1; k < j; ++k) {
              t += double(Z[k * n + i].val) / fact * y[k];
            }
            x[i] += t;
          });
    }

    return flag ? solve_res_t{true, j, e_coef * std::abs(double(h_e[0]))}
                : solve_res_t{false, max_iter, e_coef * std::abs(double(h_e[0]))};
  }
};

#endif

} // namespace kmm

#endif // UMINEKO_SPARSE_SOLVER_GMRES_HPP
