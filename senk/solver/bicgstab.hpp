#ifndef SENK_SOLVER_BICG_HPP
#define SENK_SOLVER_BICG_HPP

#include "senk/core/tensor.hpp"
#include "senk/models.hpp"
#include "senk/solver/base.hpp"

namespace senk {

template <typename T, typename T2, typename Op, typename Pre>
struct BiCGStab : public has_val_t<T>,
                  public has_loc_t<typename Op::loc_t>,
                  public has_idx_t<typename Op::idx_t>,
                  public has_params<param::kls>,
                  public model::is_operator<BiCGStab<T, T2, Op, Pre>>,
                  public model::is_solver<BiCGStab<T, T2, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>);
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<BiCGStab>::nrows;
  using model::is_operator<BiCGStab>::ncols;
  using mid_t = T2;

  Params prm;

  Op A;
  Pre M;

  BiCGStab(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), p(A.nrows()), Mp(A.nrows()), AMp(A.nrows()),
        s(A.nrows()), Ms(A.nrows()), AMs(A.nrows()), r(A.nrows()),
        rs(A.nrows()), reduce(A.nrows()) {}

private:
  mutable vector<val_t, loc_t> p, Mp, AMp, s, Ms, AMs, r, rs;
  mutable scalar<double, loc_t> alpha, beta, omega, r_rs, prev_r_rs;
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
    p.copy(rs.copy(r));
    r_rs = reduce.dot(r, rs);
    h_nrm_r.copy(r_rs).sqrt();
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    for (int i = 0; i < prm.max_iter; i++) {
      M.apply(p, Mp);
      A.apply(Mp, AMp);
      alpha =
          r_rs / reduce.dot(rs.template as<mid_t>(), AMp.template as<mid_t>());
      s = r - alpha * AMp;
      M.apply(s, Ms);
      A.apply(Ms, AMs);
      omega = reduce.dot(s.template as<mid_t>(), AMs.template as<mid_t>());
      omega /= reduce.dot(AMs.template as<mid_t>(), AMs.template as<mid_t>());
      if constexpr (init)
        (i == 0) ? (out = alpha * Mp + omega * Ms)
                 : (out += alpha * Mp + omega * Ms);
      else
        out += alpha * Mp + omega * Ms;
      r = s - omega * AMs;
      h_nrm_r = reduce.norm(r.template as<double>());
      if (cond && cond(i + 1, h_nrm_r[0]))
        return solve_res_t{true, i + 1, h_nrm_r[0]};
      prev_r_rs.copy(r_rs);
      r_rs = reduce.dot(r.template as<mid_t>(), rs.template as<mid_t>());
      beta = (alpha / omega) * (r_rs / prev_r_rs);
      p = r + beta * (p - omega * AMp);
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<BiCGStab>;
  friend struct model::is_solver<BiCGStab>;
};

} // namespace senk

#endif