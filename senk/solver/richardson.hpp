#ifndef SENK_SOLVER_RICHARDSON_HPP
#define SENK_SOLVER_RICHARDSON_HPP

#include "senk/core/tensor.hpp"
#include "senk/models.hpp"
#include "senk/solver/base.hpp"

namespace senk {

template <typename T, typename Op, typename Pre>
struct Richardson : public has_val_t<T>,
                    public has_idx_t<typename Op::idx_t>,
                    public has_loc_t<typename Op::loc_t>,
                    public has_params<param::ss>,
                    public model::is_operator<Richardson<T, Op, Pre>>,
                    public model::is_solver<Richardson<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<Richardson<T, Op, Pre>>::nrows;
  using model::is_operator<Richardson<T, Op, Pre>>::ncols;

  mutable Params prm;

  Op A;
  Pre M;

  Richardson(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), r(A.nrows()), temp(A.nrows()), reduce(A.nrows()) {
    a = prm.weight;
  }

private:
  mutable vector<val_t, loc_t> r, temp;
  mutable scalar<val_t, loc_t> a;
  mutable scalar<val_t, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    M.apply(in, out);
    out *= a;
    for (int k = 1; k < prm.max_iter; k++) {
      A.residual(in, out, r);
      M.apply(r, temp);
      out += a * temp;
    }
  }

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    for (int k = 0; k < prm.max_iter; k++) {
      A.residual(in, out, r);
      if (cond) {
        h_nrm_r = reduce.norm(r);
        if (cond(k, h_nrm_r[0]))
          return {true, k, h_nrm_r[0]};
      }
      M.apply(r, temp);
      out += a * temp;
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<Richardson<T, Op, Pre>>;
  friend struct model::is_solver<Richardson<T, Op, Pre>>;
};

template <typename T, typename Op, typename Pre>
struct RichardsonAdapt : public has_val_t<T>,
                         public has_idx_t<typename Op::idx_t>,
                         public has_loc_t<typename Op::loc_t>,
                         public has_params<param::ss_cycle>,
                         public model::is_operator<RichardsonAdapt<T, Op, Pre>>,
                         public model::is_solver<RichardsonAdapt<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<RichardsonAdapt<T, Op, Pre>>::nrows;
  using model::is_operator<RichardsonAdapt<T, Op, Pre>>::ncols;

  using mid_t = std::conditional_t<std::is_same_v<val_t, half>, float, T>;

  Params prm;

  Op A;
  Pre M;

  RichardsonAdapt(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), r(A.nrows()), temp1(A.nrows()), temp2(A.nrows()),
        a(prm.max_iter), reduce(A.nrows()) {
    a.fill(prm.weight);
    cnt = 0;
    num.fill(1.);
  }

  void reset_param() {
    a.fill(prm.weight);
    cnt = 0;
    num.fill(1.);
  }

private:
  mutable vector<val_t, loc_t> r, temp1, temp2;
  mutable vector<mid_t, loc_t> a;
  mutable scalar<mid_t, loc_t> dot1, dot2;
  mutable scalar<val_t, host> h_nrm_r;
  mutable scalar<double, loc_t> num;
  mutable int cnt;
  reducer<loc_t> reduce;

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    if (cnt == prm.cycle - 1) {
      M.apply(in, temp1);
      A.apply(temp1, temp2);
      dot1 = reduce.dot(temp2.template as<mid_t>(), temp2.template as<mid_t>());
      dot2 = reduce.dot(temp2.template as<mid_t>(), in.template as<mid_t>());
      out = dot2 / dot1 * temp1;
      a(0) = a(0) * num + dot2 / dot1;
      for (int k = 1; k < prm.max_iter; k++) {
        A.residual(in, out, r);
        M.apply(r, temp1);
        A.apply(temp1, temp2);
        dot1 =
            reduce.dot(temp2.template as<mid_t>(), temp2.template as<mid_t>());
        dot2 = reduce.dot(temp2.template as<mid_t>(), r.template as<mid_t>());
        out += dot2 / dot1 * temp1;
        a(k) = a(k) * num + dot2 / dot1;
      }
      num += 1.;
      a /= num;
      cnt = -1;
    } else {
      M.apply(in, out);
      out *= a(0);
      for (int k = 1; k < prm.max_iter; k++) {
        A.residual(in, out, r);
        M.apply(r, temp1);
        out += a(k) * temp1;
      }
    }
    cnt++;
  }

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    for (int k = 0; k < prm.max_iter; k++) {
      A.residual(in, out, r);
      if (cond) {
        h_nrm_r = reduce.norm(r);
        if (cond(k, h_nrm_r[0]))
          return {true, k, h_nrm_r[0]};
      }
      M.apply(r, temp1);
      if (cnt == prm.cycle - 1) {
        A.apply(temp1, temp2);
        dot1 =
            reduce.dot(temp2.template as<mid_t>(), temp2.template as<mid_t>());
        dot2 = reduce.dot(temp2.template as<mid_t>(), r);
        out += dot2 / dot1 * temp1;
        a(k) = a(k) * num + dot2 / dot1;
      } else {
        out += a(k) * temp1;
      }
    }
    if (cnt == prm.cycle - 1) {
      num += 1.;
      a /= num;
      cnt = -1;
    }
    cnt++;
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<RichardsonAdapt<T, Op, Pre>>;
  friend struct model::is_solver<RichardsonAdapt<T, Op, Pre>>;
};

template <typename T, typename Op, typename Pre>
struct GMRES1 : public has_val_t<T>,
                public has_idx_t<typename Op::idx_t>,
                public has_loc_t<typename Op::loc_t>,
                public has_params<param::ss>,
                public model::is_operator<GMRES1<T, Op, Pre>>,
                public model::is_solver<GMRES1<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<GMRES1<T, Op, Pre>>::nrows;
  using model::is_operator<GMRES1<T, Op, Pre>>::ncols;

  mutable Params prm;

  Op A;
  Pre M;

  GMRES1(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), r(A.nrows()), temp(A.nrows()), AMr(A.nrows()),
        reduce(A.nrows()) {
    a = prm.weight;
  }

private:
  mutable vector<val_t, loc_t> r, temp, AMr;
  mutable scalar<val_t, loc_t> a, b, c;
  mutable scalar<val_t, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    M.apply(in, out);
    out *= a;
    for (int k = 1; k < prm.max_iter; k++) {
      A.residual(in, out, r);
      M.apply(r, temp);
      out += a * temp;
    }
  }

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    for (int k = 0; k < prm.max_iter; k++) {
      A.residual(in, out, r);
      if (cond) {
        h_nrm_r = reduce.norm(r);
        if (cond(k, h_nrm_r[0]))
          return {true, k, h_nrm_r[0]};
      }
      M.apply(r, temp);

      // if (k == 0) {
      {
        A.apply(temp, AMr);
        b = reduce.dot(AMr, AMr);
        c = reduce.dot(r, AMr);
        a = c / b;
        printf("%e\n", a[0]);
      }

      out += a * temp;
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<GMRES1<T, Op, Pre>>;
  friend struct model::is_solver<GMRES1<T, Op, Pre>>;
};

} // namespace senk

#endif