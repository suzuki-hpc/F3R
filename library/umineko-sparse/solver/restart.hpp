#ifndef UMINEKO_SPARSE_SOLVER_RESTART_HPP
#define UMINEKO_SPARSE_SOLVER_RESTART_HPP

#include "umineko-sparse/models.hpp"

namespace kmm {

template <typename T, class L> struct Restarted {
  const CoefficientMatrix<L> A;
  const Solver<L> M;
  mutable vector<T, L> r, temp, ans;
  mutable scalar<T, host> h_nrm_r;

  struct params {
    int32_t max_iter;
  } param;

  Restarted(const CoefficientMatrix<L> &A, const Solver<L> &M, params param)
      : A(A), M(M), r(A.nrows()), temp(A.nrows()), ans(A.nrows()), param(param) {
    impl::pool<T, L>::init(A.nrows());
  }

  [[nodiscard]] idx_t nrows() const { return A.nrows(); }
  [[nodiscard]] idx_t ncols() const { return A.ncols(); }

  template <typename val1_t, typename val2_t>
  solve_res_t solve(const vector<val1_t, L> &rhs, vector<val2_t, L> x,
      const std::function<bool(int, double)> &cond) const {
    int32_t cnt = 1;
    auto flag = M.solve(rhs, x, cond);
    while (!flag.is_solved && cnt < param.max_iter) {
      flag += M.solve(rhs, x, cond);
      cnt++;
    }
    return flag;
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_SOLVER_RESTART_HPP
