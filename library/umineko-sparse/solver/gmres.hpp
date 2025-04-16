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
      auto t = std::sqrt(H[k] * H[k] + H[k + 1] * H[k + 1]);
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
    if (cond && cond(0, std::abs(h_nrm_r[0]))) {
      if constexpr (with_init)
        x.fill(val2_t(0.0));
      return solve_res_t{true, 0, std::abs(h_nrm_r[0])};
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
      if (cond && cond(j + 1, std::abs(h_nrm_r[0]))) {
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
    return flag ? solve_res_t{true, j, std::abs(h_nrm_r[0])}
                : solve_res_t{false, max_iter, std::abs(h_nrm_r[0])};
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_SOLVER_GMRES_HPP
