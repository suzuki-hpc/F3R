#ifndef SENK_SOLVER_CR_HPP
#define SENK_SOLVER_CR_HPP

#include "senk/core/tensor.hpp"
#include "senk/models.hpp"
#include "senk/solver/base.hpp"

namespace senk {

template <typename T, typename T2, typename Op, typename Pre>
struct CR : public has_val_t<T>,
            public has_idx_t<typename Op::idx_t>,
            public has_loc_t<typename Op::loc_t>,
            public has_params<param::kls>,
            public model::is_operator<CR<T, T2, Op, Pre>>,
            public model::is_solver<CR<T, T2, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<CR<T, T2, Op, Pre>>::nrows;
  using model::is_operator<CR<T, T2, Op, Pre>>::ncols;
  using mid_t = T2;

  Params prm;

  Op A;
  Pre M;

  CR(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), p(A.nrows()), Ap(A.nrows()), MAp(A.nrows()),
        r(A.nrows()), Mr(A.nrows()), AMr(A.nrows()), reduce(A.nrows()) {}

private:
  mutable vector<val_t, loc_t> p, Ap, MAp, r, Mr, AMr;
  mutable scalar<mid_t, loc_t> alpha, beta, AMr_Mr;
  mutable scalar<double, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    if (!init)
      A.residual(in, out, r);
    else
      r.copy(in);
    h_nrm_r = reduce.norm(r.template as<double>());
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }

    for (int i = 0; i < prm.max_iter; i++) {
      if (i == 0) {
        M.apply(r, Mr);
        A.apply(Mr, AMr);
        p.copy(Mr);
        Ap.copy(AMr);
        AMr_Mr = reduce.dot(AMr.template as<mid_t>(), Mr.template as<mid_t>());
      } else {
        Mr -= alpha * MAp;
        A.apply(Mr, AMr);
        beta.copy(AMr_Mr);
        AMr_Mr = reduce.dot(AMr.template as<mid_t>(), Mr.template as<mid_t>());
        beta = AMr_Mr / beta;
        p = Mr + beta * p;
        Ap = AMr + beta * Ap;
      }
      M.apply(Ap, MAp);
      alpha = AMr_Mr /
              reduce.dot(MAp.template as<mid_t>(), Ap.template as<mid_t>());
      if constexpr (init)
        (i == 0) ? (out = alpha * p) : (out += alpha * p);
      else
        out += alpha * p;
      r -= alpha * Ap;
      h_nrm_r = reduce.norm(r.template as<double>());
      if (cond && cond(i + 1, h_nrm_r[0]))
        return solve_res_t{true, i + 1, h_nrm_r[0]};
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<CR<T, T2, Op, Pre>>;
  friend struct model::is_solver<CR<T, T2, Op, Pre>>;
};

} // namespace senk

#endif
