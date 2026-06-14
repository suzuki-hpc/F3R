#ifndef SENK_SOLVER_MINRES_HPP
#define SENK_SOLVER_MINRES_HPP

#include "senk/core/tensor.hpp"
#include "senk/models.hpp"
#include "senk/solver/base.hpp"

namespace senk {

template <typename T, typename Op, typename Pre>
struct MINRES : public has_val_t<T>,
                public has_idx_t<typename Op::idx_t>,
                public has_loc_t<typename Op::loc_t>,
                public has_params<param::kls_normalize>,
                public model::is_operator<MINRES<T, Op, Pre>>,
                public model::is_solver<MINRES<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<MINRES<T, Op, Pre>>::nrows;
  using model::is_operator<MINRES<T, Op, Pre>>::ncols;

  Params prm;

  Op A;
  Pre M;

  MINRES(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), V(A.nrows(), 3), W(A.nrows(), 3), U(A.nrows(), 3),
        r(A.nrows()), alp(3), bet(3), gam(3), sig(3), reduce(A.nrows()) {}

private:
  mutable matrix<val_t, loc_t> V, W, U;
  mutable vector<val_t, loc_t> r, alp, bet, gam, sig;
  mutable scalar<val_t, loc_t> eta, delta, rho1, rho2, rho3, tmp, nrm_r;
  mutable scalar<val_t, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    if (!init)
      A.residual(in, out, r);
    else
      r.copy(in);
    if (prm.normalized)
      h_nrm_r.copy(nrm_r = 1.);
    else
      h_nrm_r.copy(nrm_r = reduce.norm(r));
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    gam.fill(1);
    sig.fill(0);
    W(0).fill(0.0);
    W(1).fill(0.0);
    V(0).fill(0.0);
    V(1).copy(r);
    M.apply(V(1), U(0));
    bet(0) = reduce.dot(V(1), U(0));
    bet(0).sqrt();
    eta.copy(bet(0));
    for (int j = 0; j < prm.max_iter; j++) {
      auto k = j % 3;
      auto k1 = (j + 1) % 3;
      auto k2 = (j + 2) % 3;
      tmp.copy(bet(k)).inv();
      V(k1) *= tmp;
      U(k) *= tmp;
      A.apply(U(k), V(k2));
      alp(k) = reduce.dot(U(k), V(k2));
      V(k2) -= alp(k) * V(k1) + bet(k) * V(k);
      M.apply(V(k2), U(k1));
      bet(k1) = reduce.dot(V(k2), U(k1));
      bet(k1).sqrt();
      delta = gam(k1) * alp(k) - gam(k) * sig(k1) * bet(k);
      rho1 = delta * delta + bet(k1) * bet(k1);
      rho1.sqrt().inv();
      rho2 = sig(k1) * alp(k) + gam(k) * gam(k1) * bet(k);
      rho3 = sig(k) * bet(k);
      gam(k2) = delta * rho1;
      sig(k2) = bet(k1) * rho1;
      W(k2) = rho1 * (U(k) - rho3 * W(k) - rho2 * W(k1));
      if constexpr (init)
        (j == 0) ? out = gam(k2) * eta * W(k2) : out += gam(k2) * eta * W(k2);
      else
        out += gam(k2) * eta * W(k2);
      h_nrm_r.copy(nrm_r *= sig(k2));
      if (cond && cond(j + 1, h_nrm_r[0]))
        return solve_res_t{true, j + 1, h_nrm_r[0]};
      eta *= -sig(k2);
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<MINRES<T, Op, Pre>>;
  friend struct model::is_solver<MINRES<T, Op, Pre>>;
};

} // namespace senk

#endif
