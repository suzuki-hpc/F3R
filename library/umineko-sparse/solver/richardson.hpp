#ifndef UMINEKO_SPARSE_SOLVER_RICHARDSON_HPP
#define UMINEKO_SPARSE_SOLVER_RICHARDSON_HPP

#include "umineko-sparse/models.hpp"

#if __has_include("umineko-type/fixed.hpp")
#include "umineko-type/fixed.hpp"
#endif

namespace kmm {

template <typename T, class L> struct Richardson {
  const CoefficientMatrix<L> A;
  const Operator<L> M;
  mutable vector<T, L> r, temp, ans;
  mutable scalar<T, host> h_nrm_r;
  scalar<T, L> alpha;

  struct params {
    int max_iter;
    double alpha;
  } param;

  Richardson(const CoefficientMatrix<L> A, const Operator<L> M, params param)
      : A(A), M(M), r(A.nrows()), temp(A.nrows()), ans(A.nrows()), param(param) {
    alpha = param.alpha;
    impl::pool<T, L>::init(A.nrows());
  }

  [[nodiscard]] idx_t nrows() const { return A.nrows(); }
  [[nodiscard]] idx_t ncols() const { return A.ncols(); }

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    M.operate(in, out);
    out *= alpha;
    for (int k = 1; k < param.max_iter; k++) {
      A.compute_residual(in, out, r);
      M.operate(r, temp);
      out += alpha * temp;
    }
#if 0
    scalar<T, L> dot_ar, dot_r;
    auto tmp = vector<T, L>(A.ncols());
    M.operate(in, temp);
    {
      A.operate(temp, tmp);
      dot_ar = dot(tmp, tmp);
      dot_r = dot(tmp, in);
      // if (alpha[0] == 1.8)
      // printf(" %e\n", dot_r[0] / dot_ar[0]);
    }
    out = dot_r / dot_ar * temp;
    for (int k = 1; k < param.max_iter; k++) {
      A.compute_residual(in, out, r);
      M.operate(r, temp);
      {
        A.operate(temp, tmp);
        dot_ar = dot(tmp, tmp);
        dot_r = dot(tmp, r);
        // if (alpha[0] == 1.8)
        //   printf("%e\n", dot_r[0] / dot_ar[0]);
      }
      out += dot_r / dot_ar * temp;
    }
#endif
  }

  template <typename val1_t, typename val2_t>
  solve_res_t solve(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    for (int k = 0; k < param.max_iter; k++) {
      A.compute_residual(rhs, x, r);
      if (cond) {
        h_nrm_r = nrm(r);
        if (cond(k, abs(h_nrm_r[0])))
          return {true, k, abs(h_nrm_r[0])};
      }
      M.operate(r, temp);
      x += alpha * temp;
    }
    if (cond)
      return {false, param.max_iter, abs(h_nrm_r[0])};
    else
      return {false, param.max_iter, 1.};
  }

  template <typename val1_t, typename val2_t>
  void smooth(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    for (int k = 0; k < param.max_iter; k++) {
      A.compute_residual(in, out, r);
      M.operate(r, temp);
      out += alpha * temp;
    }
  }
};

template <typename T, class L> struct AutoRichardson {
  const CoefficientMatrix<L> A;
  const Operator<L> M;
  mutable vector<T, L> r, temp1, temp2, ans, tmp;
  mutable scalar<T, host> h_nrm_r;
  mutable scalar<T, L> dot_ar, dot_r;
  mutable scalar<T, L> alpha1, alpha2, sum1, sum2, den;
  mutable int cnt;

  struct params {
    int max_iter;
    int period;
  } param;

  AutoRichardson(const CoefficientMatrix<L> A, const Operator<L> M, params param)
      : A(A), M(M), r(A.nrows()), temp1(A.nrows()), temp2(A.nrows()), ans(A.nrows()),
        tmp(A.nrows()), param(param) {
    alpha1 = 1.0;
    alpha2 = 1.0;
    sum1 = 1.0;
    sum2 = 1.0;
    cnt = param.period;
    impl::pool<T, L>::init(A.nrows());
  }

  [[nodiscard]] idx_t nrows() const { return A.nrows(); }
  [[nodiscard]] idx_t ncols() const { return A.ncols(); }

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    ++cnt;
    M.operate(in, temp1);
    if (cnt % param.period == 0) {
      A.operate(temp1, tmp);
      dot_ar = dot(tmp, tmp);
      dot_r = dot(tmp, in);
      out = dot_r / dot_ar * temp1;
      // alpha1 = dot_r / dot_ar;
      sum1 += dot_r / dot_ar;
      alpha1 = sum1 / (den = cnt / param.period);
    } else {
      out = alpha1 * temp1;
    }
    for (int k = 1; k < param.max_iter; k++) {
      A.compute_residual(in, out, r);
      M.operate(r, temp2);
      if (cnt % param.period == 0) {
        A.operate(temp2, tmp);
        dot_ar = dot(tmp, tmp);
        dot_r = dot(tmp, r);
        out += dot_r / dot_ar * temp2;
        // alpha2 = dot_r / dot_ar;
        sum2 += dot_r / dot_ar;
        alpha2 = sum2 / den;
      } else {
        out += alpha2 * temp2;
      }
    }
  }

  template <typename val1_t, typename val2_t>
  solve_res_t solve(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    printf("ERROR: AutoRichardson solve is not implemented\n");
    return {false, param.max_iter, abs(h_nrm_r[0])};
  }
};

#if 0 // __has_include("umineko-type/fixed.hpp")

template <uint32_t label, class L> struct Richardson<fixed32_t<label>, L> {
  const IntOperator<L, label> A;
  const IntOperator<L, label> M;
  mutable vector<fixed32_t<label>, L> r, temp, ans;

  struct params {
    int max_iter;
    double alpha;
  } param;

  Richardson(
      const IntOperator<L, label> A, const IntOperator<L, label> M, params param)
      : A(A), M(M), r(A.nrows()), temp(A.nrows()), ans(A.nrows()), param(param) {
    impl::pool<typename fixed32_t<label>::template _mid_fixed_t<label>, L>::init(
        A.nrows());
  }

  [[nodiscard]] idx_t nrows() const { return A.nrows(); }
  [[nodiscard]] idx_t ncols() const { return A.ncols(); }

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    M.operate(in, out);
    for (int k = 1; k < param.max_iter; k++) {
      A.operate(out, r);
      r = in - r;
      M.operate(r, temp);
      out += temp;
    }
  }

  template <typename val1_t, typename val2_t>
  solve_res_t solve(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    for (int k = 0; k < param.max_iter; k++) {
      A.operate(x, r);
      r = rhs - r;
      M.operate(r, temp);
      x += temp;
    }
    if (cond)
      return {false, param.max_iter, 1};
    else
      return {false, param.max_iter, 1.};
  }

  template <typename val1_t, typename val2_t>
  void smooth(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    for (int k = 0; k < param.max_iter; k++) {
      A.operate(out, r);
      r = in - r;
      M.operate(r, temp);
      out += temp;
    }
  }
};

#endif

} // namespace kmm

#endif // UMINEKO_SPARSE_SOLVER_RICHARDSON_HPP
