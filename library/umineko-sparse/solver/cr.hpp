#ifndef UMINEKO_SPARSE_SOLVER_CR_HPP
#define UMINEKO_SPARSE_SOLVER_CR_HPP

#include "umineko-sparse/models.hpp"

namespace kmm {

template <typename T, class L> struct CR {
  const CoefficientMatrix<L> A;
  const Operator<L> M;
  int32_t max_iter;
  mutable vector<T, L> p, Ap, MAp, r, Mr, AMr;
  mutable scalar<T, L> alpha, beta, AMr_Mr;
  mutable scalar<T, host> h_nrm_r;

  mutable vector<T, L> act;

  struct params {
    int32_t max_iter;
  };

  CR(const CoefficientMatrix<L> A, const Operator<L> M, params param)
      : A(A), M(M), max_iter(param.max_iter), p(A.nrows()), Ap(A.nrows()), MAp(A.nrows()),
        r(A.nrows()), Mr(A.nrows()), AMr(A.nrows()), act(A.nrows()) {
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
    if (cond && cond(0, h_nrm_r[0]))
      return (x.fill(0.0), solve_res_t{true, 0, h_nrm_r[0]});
    M.operate(r, Mr);
    A.operate(Mr, AMr);
    p.copy(Mr);
    Ap.copy(AMr);
    AMr_Mr = dot(AMr, Mr);
    for (int i = 0; i < max_iter; i++) {
      M.operate(Ap, MAp);
      alpha = AMr_Mr / dot(MAp, Ap);
      if constexpr (with_init)
        (i == 0) ? (x = alpha * p) : (x += alpha * p);
      else
        x += alpha * p;
      r -= alpha * Ap;
      h_nrm_r = nrm(r);
      if (cond && cond(i + 1, h_nrm_r[0]))
        return solve_res_t{true, i + 1, h_nrm_r[0]};
      Mr -= alpha * MAp;
      A.operate(Mr, AMr);
      beta.copy(AMr_Mr);
      beta = (AMr_Mr = dot(AMr, Mr)) / beta;
      p = Mr + beta * p;
      Ap = AMr + beta * Ap;
    }
    return solve_res_t{false, max_iter, h_nrm_r[0]};
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_SOLVER_CG_HPP
