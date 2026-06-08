#ifndef SENK_SOLVER_CG_HPP
#define SENK_SOLVER_CG_HPP

#include "senk/core/tensor.hpp"
#include "senk/models.hpp"
#include "senk/solver/base.hpp"

namespace senk {

template <typename T, typename T2, typename Op, typename Pre>
struct CG : public has_val_t<T>,
            public has_idx_t<typename Op::idx_t>,
            public has_loc_t<typename Op::loc_t>,
            public has_params<param::kls>,
            public model::is_operator<CG<T, T2, Op, Pre>>,
            public model::is_solver<CG<T, T2, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<CG<T, T2, Op, Pre>>::nrows;
  using model::is_operator<CG<T, T2, Op, Pre>>::ncols;
  using mid_t = T2;

  Params prm;

  Op A;
  Pre M;

  CG(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), p(A.nrows()), Ap(A.nrows()), r(A.nrows()),
        z(A.nrows()), reduce(A.nrows()) {}

private:
  mutable vector<val_t, loc_t> p, Ap, r, z;
  mutable scalar<mid_t, loc_t> alpha, beta, r_z;
  mutable scalar<double, loc_t> nrm_r;
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
    if (cond)
      h_nrm_r.copy(nrm_r = reduce.norm(r.template as<double>()));
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    for (int i = 0; i < prm.max_iter; i++) {
      M.apply(r, z);
      beta.copy(r_z);
      r_z = reduce.dot(r.template as<mid_t>(), z.template as<mid_t>());

      if (i == 0) {
        p.copy(z);
      } else {
        beta = r_z / beta;
        p = z + beta * p;
      }

      A.apply(p, Ap);
      alpha = r_z / reduce.dot(p.template as<mid_t>(), Ap.template as<mid_t>());
      if constexpr (init)
        (i == 0) ? (out = alpha * p) : (out += alpha * p);
      else
        out += alpha * p;
      r -= alpha * Ap;
      if (cond)
        h_nrm_r.copy(nrm_r = reduce.norm(r.template as<double>()));
      if (cond && cond(i + 1, h_nrm_r[0]))
        return solve_res_t{true, i + 1, h_nrm_r[0]};
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<CG<T, T2, Op, Pre>>;
  friend struct model::is_solver<CG<T, T2, Op, Pre>>;
};

template <typename T, typename T2, typename Op, typename Pre>
struct FCG : public has_val_t<T>,
             public has_idx_t<typename Op::idx_t>,
             public has_loc_t<typename Op::loc_t>,
             public has_params<param::kls>,
             public model::is_operator<FCG<T, T2, Op, Pre>>,
             public model::is_solver<FCG<T, T2, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<FCG<T, T2, Op, Pre>>::nrows;
  using model::is_operator<FCG<T, T2, Op, Pre>>::ncols;
  using mid_t = T2;

  Params prm;

  Op A;
  Pre M;

  FCG(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), P(A.nrows(), prm.max_iter + 1),
        AP(A.nrows(), prm.max_iter + 1), r(A.nrows()), z(A.nrows()),
        p_Ap(prm.max_iter + 1), reduce(A.nrows()) {}

private:
  mutable matrix<val_t, loc_t> P, AP;
  mutable vector<val_t, loc_t> r, z, p_Ap;
  mutable scalar<mid_t, loc_t> alpha, beta, r_z;
  mutable scalar<double, loc_t> nrm_r;
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
    if (cond)
      h_nrm_r.copy(nrm_r = reduce.norm(r));
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    for (int i = 0; i < prm.max_iter; i++) {
      M.apply(r, z);
      P(i).copy(z);
      for (int k = 0; k < i; k++) {
        beta = reduce.dot(z.template as<mid_t>(), AP(k).template as<mid_t>()) /
               p_Ap(k);
        P(i) -= beta * P(k);
      }

      A.apply(P(i), AP(i));
      p_Ap(i) =
          reduce.dot(P(i).template as<mid_t>(), AP(i).template as<mid_t>());
      r_z = reduce.dot(r.template as<mid_t>(), P(i).template as<mid_t>());
      alpha = r_z / p_Ap(i);
      if constexpr (init)
        (i == 0) ? (out = alpha * P(i)) : (out += alpha * P(i));
      else
        out += alpha * P(i);
      r -= alpha * AP(i);
      if (cond)
        h_nrm_r.copy(nrm_r = reduce.norm(r.template as<double>()));
      if (cond && cond(i + 1, h_nrm_r[0]))
        return solve_res_t{true, i + 1, h_nrm_r[0]};
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<FCG<T, T2, Op, Pre>>;
  friend struct model::is_solver<FCG<T, T2, Op, Pre>>;
};

} // namespace senk

#endif