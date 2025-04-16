#ifndef UMINEKO_SPARSE_SOLVER_GCR_HPP
#define UMINEKO_SPARSE_SOLVER_GCR_HPP

#include "umineko-sparse/models.hpp"

namespace kmm {

template <typename T, class L> struct GCR {
  const CoefficientMatrix<L> A;
  const Operator<L> M;
  int32_t max_iter;
  mutable matrix<T, L> U, C;
  mutable vector<T, L> r, dot_c, alpha;
  mutable scalar<T, L> beta;
  mutable scalar<T, host> h_nrm_r;

  mutable vector<T, L> act;

  struct params {
    int32_t max_iter;
  };

  GCR(const CoefficientMatrix<L> A, const Operator<L> M, params param)
      : A(A), M(M), max_iter(param.max_iter), U(A.nrows(), max_iter),
        C(A.nrows(), max_iter), r(A.ncols()), dot_c(max_iter), alpha(max_iter),
        act(A.nrows()) {
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
    r.copy(rhs);
    h_nrm_r = nrm(r);
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (with_init)
        x.fill(val2_t(0.0));
      return (x.fill(0.0), solve_res_t{true, 0, h_nrm_r[0]});
    }
    M.operate(r, U(0));
    A.operate(U(0), C(0));
    for (int j = 0; j < max_iter; j++) {
      dot_c(j) = dot(C(j), C(j));
      beta = dot(C(j), r) / dot_c(j);
      if constexpr (with_init)
        (j == 0) ? (x = beta * U(j)) : (x += beta * U(j));
      else
        x += beta * U(j);
      r -= beta * C(j);
      h_nrm_r = nrm(r);
      if (cond && cond(j + 1, h_nrm_r[0]))
        return {true, j + 1, h_nrm_r[0]};
      if (j == max_iter - 1)
        break;
      M.operate(r, U(j + 1));
      A.operate(U(j + 1), C(j + 1));
      for (int k = 0; k <= j; k++)
        alpha(k) = -dot(C(k), C(j + 1)) / dot_c(k);
      U(j + 1) += U * alpha.slice(j + 1);
      C(j + 1) += C * alpha.slice(j + 1);
    }
    return {false, max_iter, h_nrm_r[0]};
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_SOLVER_CG_HPP
