#ifndef UMINEKO_SPARSE_SOLVER_BICGSTAB_HPP
#define UMINEKO_SPARSE_SOLVER_BICGSTAB_HPP

#include "umineko-sparse/models.hpp"

namespace kmm {

template <typename T, class L> struct BiCGSTAB {
  const CoefficientMatrix<L> A;
  const Operator<L> M;
  int32_t max_iter;
  mutable vector<T, L> p, Mp, AMp, s, Ms, AMs, r, rs;
  mutable scalar<T, L> alpha, beta, omega, r_rs;
  mutable scalar<T, host> h_nrm_r;

  mutable vector<T, L> act;

  struct params {
    int32_t max_iter;
  };

  BiCGSTAB(const CoefficientMatrix<L> A, const Operator<L> M, params param)
      : A(A), M(M), max_iter(param.max_iter), p(A.nrows()), Mp(A.nrows()),
        AMp(A.nrows()), s(A.nrows()), Ms(A.nrows()), AMs(A.nrows()), r(A.nrows()),
        rs(A.nrows()), act(A.nrows()) {
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
    p.copy(rs.copy(r.copy(rhs)));
    r_rs = dot(r, rs);
    h_nrm_r.copy(r_rs).sqrt();
    if (cond && cond(0, h_nrm_r[0]))
      return (x.fill(0.0), solve_res_t{true, 0, h_nrm_r[0]});
    for (int i = 0; i < max_iter; i++) {
      M.operate(p, Mp);
      A.operate(Mp, AMp);
      alpha = r_rs / dot(rs, AMp);
      s = r - alpha * AMp;
      M.operate(s, Ms);
      A.operate(Ms, AMs);
      omega = dot(s, AMs);
      omega /= dot(AMs, AMs);
      if constexpr (with_init)
        (i == 0) ? (x = alpha * Mp + omega * Ms) : (x += alpha * Mp + omega * Ms);
      else
        x += alpha * Mp + omega * Ms;
      r = s - omega * AMs;
      h_nrm_r = nrm(r);
      if (cond && cond(i + 1, h_nrm_r[0]))
        return solve_res_t{true, i + 1, h_nrm_r[0]};
      beta = omega * r_rs;
      beta = alpha * (r_rs = dot(r, rs)) / beta;
      p = r + beta * (p - omega * AMp);
    }
    return solve_res_t{false, max_iter, h_nrm_r[0]};
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_SOLVER_BICGSTAB_HPP
