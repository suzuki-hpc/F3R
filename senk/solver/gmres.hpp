#ifndef SENK_SOLVER_GMRES_HPP
#define SENK_SOLVER_GMRES_HPP

#include "senk/core/tensor.hpp"
#include "senk/models.hpp"
#include "senk/solver/base.hpp"

namespace senk {

namespace impl {

template <typename val_t, class loc_t>
void final_grot(const vector<val_t, loc_t> &c, const vector<val_t, loc_t> &s,
    vector<val_t, loc_t> e, int k) {
  auto cp = c.raw();
  auto sp = s.raw();
  auto ep = e.raw();
  kernel<loc_t>::single([=] SENK_LOC() mutable {
    ep[k + 1] = -sp[k] * ep[k];
    ep[k] = cp[k] * ep[k];
  });
}

template <typename val_t, class loc_t>
void full_grot(vector<val_t, loc_t> &c, vector<val_t, loc_t> &s,
    vector<val_t, loc_t> H, int k) {
  auto cp = c.raw();
  auto sp = s.raw();
  auto Hp = H.raw();
  kernel<loc_t>::single([=] SENK_LOC() mutable {
    for (int i = 0; i < k; i++) {
      auto t = Hp[i];
      Hp[i] = cp[i] * t + sp[i] * Hp[i + 1];
      Hp[i + 1] = -sp[i] * t + cp[i] * Hp[i + 1];
    }
    auto t = sqrt(Hp[k] * Hp[k] + Hp[k + 1] * Hp[k + 1]);
    cp[k] = Hp[k] / t;
    sp[k] = Hp[k + 1] / t;
    Hp[k] = t;
    Hp[k + 1] = 0;
  });
}

template <typename mat_t, typename rhs_t, typename ans_t, class loc_t>
void trsv(const matrix<mat_t, loc_t> &U, const vector<rhs_t, loc_t> &b,
    vector<ans_t, loc_t> x, const int h, const int w) {
  using tmp_t = decltype(std::declval<mat_t>() * std::declval<ans_t>());
  auto Up = U.raw();
  auto bp = b.raw();
  auto xp = x.raw();
  kernel<loc_t>::single([=] SENK_LOC() mutable {
    for (int i = w - 1; i >= 0; i--) {
      tmp_t t = static_cast<tmp_t>(bp[i]);
      for (int j = w - 1; j > i; j--) {
        t -= Up[j * h + i] * xp[j];
      }
      xp[i] = t / Up[i * h + i];
    }
  });
}

} // namespace impl

template <typename T, typename T2, typename Op, typename Pre>
struct GMRES : public has_val_t<T>,
               public has_idx_t<typename Op::idx_t>,
               public has_loc_t<typename Op::loc_t>,
               public has_params<param::kls_normalize>,
               public model::is_operator<GMRES<T, T2, Op, Pre>>,
               public model::is_solver<GMRES<T, T2, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<GMRES>::nrows;
  using model::is_operator<GMRES>::ncols;
  using mit_t = T2;

  Params prm;

  Op A;
  Pre M;

  GMRES(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), V(A.nrows(), prm.max_iter + 1),
        H(prm.max_iter + 1, prm.max_iter), w(A.nrows()), w2(A.nrows()),
        s(prm.max_iter), c(prm.max_iter), e(prm.max_iter + 1), y(prm.max_iter),
        reduce(A.nrows()) {}

private:
  mutable matrix<val_t, loc_t> V;
  mutable matrix<double, loc_t> H;
  mutable vector<double, loc_t> w, w2, s, c, e, y;
  mutable scalar<double, loc_t> h;
  mutable scalar<double, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    bool flag = false;
    if (!init)
      A.residual(in, out, V(0));
    else
      V(0).copy(in);
    (!prm.normalized) ? h = reduce.norm(V(0).template as<double>())
                      : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (!prm.normalized) ? V(0) = h.inv() * V(0) : V(0);
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    int j = 0;
    for (; j < prm.max_iter; j++) {
      M.apply(V(j), w);
      A.apply(w, V(j + 1));
      for (int k = 0; k <= j; k++) {
        H(k, j) = reduce.dot(
            V(j + 1).template as<mit_t>(), V(k).template as<mit_t>());
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j + 1, j).copy(h = reduce.norm(V(j + 1).template as<mit_t>()));
      V(j + 1) = V(j + 1) * h.inv();
      impl::full_grot(c, s, H(j), j);
      impl::final_grot(c, s, e, j);
      h_nrm_r.copy(e(j + 1)).abs();
      if (cond && cond(j + 1, h_nrm_r[0])) {
        j++;
        flag = true;
        break;
      }
    }
    impl::trsv(H, e, y, prm.max_iter + 1, j);
    w = V * y.slice(j);
    if constexpr (init)
      M.apply(w, out);
    else {
      M.apply(w, w2);
      out += w2;
    }
    return flag ? solve_res_t{true, j, h_nrm_r[0]}
                : solve_res_t{false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<GMRES>;
  friend struct model::is_solver<GMRES>;
};

template <typename T, typename T2, typename Op, typename Pre>
struct FGMRES : public has_val_t<T>,
                public has_idx_t<typename Op::idx_t>,
                public has_loc_t<typename Op::loc_t>,
                public has_params<param::kls_normalize>,
                public model::is_operator<FGMRES<T, T2, Op, Pre>>,
                public model::is_solver<FGMRES<T, T2, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<FGMRES>::nrows;
  using model::is_operator<FGMRES>::ncols;
  using mid_t = T2;

  Params prm;

  Op A;
  Pre M;

  FGMRES(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), V(A.nrows(), prm.max_iter + 1),
        Z(A.nrows(), prm.max_iter), H(prm.max_iter + 1, prm.max_iter),
        s(prm.max_iter), c(prm.max_iter), e(prm.max_iter + 1), y(prm.max_iter),
        reduce(A.nrows()) {}

private:
  mutable matrix<val_t, loc_t> V, Z;
  mutable matrix<double, loc_t> H;
  mutable vector<double, loc_t> s, c, e, y;
  mutable scalar<double, loc_t> h;
  mutable scalar<double, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    bool flag = false;
    if (!init)
      A.residual(in, out, V(0));
    else
      V(0).copy(in);
    (!prm.normalized) ? h = reduce.norm(V(0).template as<double>())
                      : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (!prm.normalized) ? V(0) = h.inv() * V(0) : V(0);
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    int j = 0;
    for (; j < prm.max_iter; j++) {
      M.apply(V(j), Z(j));
      A.apply(Z(j), V(j + 1));
      for (int k = 0; k <= j; k++) {
        H(k, j) = reduce.dot(
            V(j + 1).template as<mid_t>(), V(k).template as<mid_t>());
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j + 1, j).copy(h = reduce.norm(V(j + 1).template as<mid_t>()));
      V(j + 1) = V(j + 1) * h.inv();
      impl::full_grot(c, s, H(j), j);
      impl::final_grot(c, s, e, j);
      h_nrm_r.copy(e(j + 1)).abs();
      if (cond && cond(j + 1, h_nrm_r[0])) {
        j++;
        flag = true;
        break;
      }
    }
    impl::trsv(H, e, y, prm.max_iter + 1, j);
    if constexpr (init)
      out = Z * y.slice(j);
    else
      out += Z * y.slice(j);
    return flag ? solve_res_t{true, j, h_nrm_r[0]}
                : solve_res_t{false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<FGMRES>;
  friend struct model::is_solver<FGMRES>;
};

template <typename T, typename T2, typename Op, typename Pre>
struct FLGMRES : public has_val_t<T>,
                 public has_idx_t<typename Op::idx_t>,
                 public has_loc_t<typename Op::loc_t>,
                 public has_params<param::kls_aug_normalize>,
                 public model::is_operator<FLGMRES<T, T2, Op, Pre>>,
                 public model::is_solver<FLGMRES<T, T2, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<FLGMRES>::nrows;
  using model::is_operator<FLGMRES>::ncols;
  using mid_t = T2;

  Params prm;

  Op A;
  Pre M;

  FLGMRES(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), V(A.nrows(), prm.max_iter + 1),
        Z(A.nrows(), prm.max_iter), AW(A.nrows(), prm.aug),
        H(prm.max_iter + 1, prm.max_iter), G(prm.max_iter + 1, prm.max_iter),
        s(prm.max_iter), c(prm.max_iter), e(prm.max_iter + 1), y(prm.max_iter),
        hy(prm.max_iter + 1), t(A.nrows()), reduce(A.nrows()), cnt(0) {}

private:
  mutable matrix<val_t, loc_t> V, Z, AW;
  mutable matrix<double, loc_t> H, G;
  mutable vector<double, loc_t> s, c, e, y, hy, t;
  mutable scalar<double, loc_t> h;
  mutable scalar<double, host> h_nrm_r;
  reducer<loc_t> reduce;
  mutable int cnt;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    bool flag = false;
    if (!init)
      A.residual(in, out, V(0));
    else
      V(0).copy(in);
    (!prm.normalized) ? h = reduce.norm(V(0).template as<double>())
                      : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (!prm.normalized) ? V(0) = h.inv() * V(0) : V(0);
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init)
        out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    int j = 0;
    for (; j < prm.max_iter; j++) {
      auto k_min = std::min(prm.aug, cnt);
      if (j < prm.max_iter - k_min) {
        M.apply(V(j), Z(j));
        A.apply(Z(j), V(j + 1));
      } else
        V(j + 1).copy(AW(prm.max_iter - 1 - j));
      for (int k = 0; k <= j; k++) {
        H(k, j) = reduce.dot(
            V(j + 1).template as<mid_t>(), V(k).template as<mid_t>());
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j + 1, j).copy(h = reduce.norm(V(j + 1).template as<mid_t>()));
      V(j + 1) = V(j + 1) * h.inv();
      G(j).copy(H(j));
      impl::full_grot(c, s, H(j), j);
      impl::final_grot(c, s, e, j);
      h_nrm_r.copy(e(j + 1)).abs();
      if (cond && cond(j + 1, h_nrm_r[0])) {
        j++;
        flag = true;
        break;
      }
    }
    impl::trsv(H, e, y, prm.max_iter + 1, j);
    auto ktt = cnt % prm.aug;
    Z(prm.max_iter - 1 - ktt) = Z * y.slice(j);
    if constexpr (init)
      out.copy(Z(prm.max_iter - 1 - ktt));
    else
      out += Z(prm.max_iter - 1 - ktt);
    AW(ktt) = V * (hy = G * y.slice(j));
    cnt++;
    return flag ? solve_res_t{true, j, h_nrm_r[0]}
                : solve_res_t{false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<FLGMRES>;
  friend struct model::is_solver<FLGMRES>;
};

#if 0
template <typename T, typename Op, typename Pre>
struct GMRES : public has_val_t<T>,
               public has_idx_t<typename Op::idx_t>,
               public has_loc_t<typename Op::loc_t>,
               public has_params<param::kls_normalize>,
               public model::is_operator<GMRES<T, Op, Pre>>,
               public model::is_solver<GMRES<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<GMRES<T, Op, Pre>>::nrows;
  using model::is_operator<GMRES<T, Op, Pre>>::ncols;

  Params prm;

  Op A;
  Pre M;

  GMRES(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), V(A.nrows(), prm.max_iter + 1),
        H(prm.max_iter + 1, prm.max_iter), w(A.nrows()), w2(A.nrows()),
        s(prm.max_iter), c(prm.max_iter), e(prm.max_iter + 1), y(prm.max_iter),
        reduce(A.nrows()) {}

private:
  mutable matrix<val_t, loc_t> V, H;
  mutable vector<val_t, loc_t> w, w2, s, c, e, y;
  mutable scalar<val_t, loc_t> h;
  mutable scalar<val_t, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    bool flag = false;
    if (!init)
      A.residual(in, out, V(0));
    else
      V(0).copy(in);
    (!prm.normalized) ? h = reduce.norm(V(0)) : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (!prm.normalized) ? V(0) = h.inv() * V(0) : V(0);
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init) out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    int j = 0;
    for (; j < prm.max_iter; j++) {
      M.apply(V(j), w);
      A.apply(w, V(j + 1));
      for (int k = 0; k <= j; k++) {
        H(k, j) = reduce.dot(V(j + 1), V(k));
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j + 1, j).copy(h = reduce.norm(V(j + 1)));
      V(j + 1) = V(j + 1) * h.inv();
      impl::full_grot(c, s, H(j), j);
      impl::final_grot(c, s, e, j);
      h_nrm_r.copy(e(j + 1));
      if (cond && cond(j + 1, std::abs(double(h_nrm_r[0])))) {
        j++;
        flag = true;
        break;
      }
    }
    impl::trsv(H, e, y, prm.max_iter + 1, j);
    w = V * y.slice(j);
    if constexpr (init)
      M.apply(w, out);
    else {
      M.apply(w, w2);
      out += w2;
    }
    return flag
               ? solve_res_t{true, j, std::abs(double(h_nrm_r[0]))}
               : solve_res_t{false, prm.max_iter, std::abs(double(h_nrm_r[0]))};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<GMRES<T, Op, Pre>>;
  friend struct model::is_solver<GMRES<T, Op, Pre>>;
};

template <typename T, typename Op, typename Pre>
struct FGMRES : public has_val_t<T>,
                public has_idx_t<typename Op::idx_t>,
                public has_loc_t<typename Op::loc_t>,
                public has_params<param::kls_normalize>,
                public model::is_operator<FGMRES<T, Op, Pre>>,
                public model::is_solver<FGMRES<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<FGMRES<T, Op, Pre>>::nrows;
  using model::is_operator<FGMRES<T, Op, Pre>>::ncols;

  Params prm;

  Op A;
  Pre M;

  FGMRES(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), V(A.nrows(), prm.max_iter + 1),
        Z(A.nrows(), prm.max_iter), H(prm.max_iter + 1, prm.max_iter),
        s(prm.max_iter), c(prm.max_iter), e(prm.max_iter + 1), y(prm.max_iter),
        reduce(A.nrows()) {}

  // private:
  mutable matrix<val_t, loc_t> V, Z, H;
  mutable vector<val_t, loc_t> s, c, e, y;
  mutable scalar<val_t, loc_t> h;
  mutable scalar<val_t, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    bool flag = false;
    if (!init)
      A.residual(in, out, V(0));
    else
      V(0).copy(in);
    (!prm.normalized) ? h = reduce.norm(V(0)) : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (!prm.normalized) ? V(0) = h.inv() * V(0) : V(0);
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init) out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    int j = 0;
    for (; j < prm.max_iter; j++) {
      M.apply(V(j), Z(j));
      A.apply(Z(j), V(j + 1));
      for (int k = 0; k <= j; k++) {
        H(k, j) = reduce.dot(V(j + 1), V(k));
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j + 1, j).copy(h = reduce.norm(V(j + 1)));
      V(j + 1) = V(j + 1) * h.inv();
      impl::full_grot(c, s, H(j), j);
      impl::final_grot(c, s, e, j);
      h_nrm_r.copy(e(j + 1));
      if (cond && cond(j + 1, std::abs(double(h_nrm_r[0])))) {
        j++;
        flag = true;
        break;
      }
    }
    impl::trsv(H, e, y, prm.max_iter + 1, j);
    if constexpr (init)
      out = Z * y.slice(j);
    else
      out += Z * y.slice(j);
    return flag ? solve_res_t{true, j, h_nrm_r[0]}
                : solve_res_t{false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<FGMRES<T, Op, Pre>>;
  friend struct model::is_solver<FGMRES<T, Op, Pre>>;
};
#endif

#if 0
template <typename T, typename Op, typename Pre>
struct dFGMRES : public has_val_t<T>,
                 public has_idx_t<typename Op::idx_t>,
                 public has_loc_t<typename Op::loc_t>,
                 public has_params<param::kls_normalize>,
                 public model::is_operator<dFGMRES<T, Op, Pre>>,
                 public model::is_solver<dFGMRES<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<dFGMRES<T, Op, Pre>>::nrows;
  using model::is_operator<dFGMRES<T, Op, Pre>>::ncols;

  Params prm;

  Op A;
  Pre M;

  dFGMRES(const Op &A, const Pre &M, Params prm)
      : prm(prm), A(A), M(M), V(A.nrows(), prm.max_iter + 1),
        Z(A.nrows(), prm.max_iter), H(prm.max_iter + 1, prm.max_iter),
        s(prm.max_iter), c(prm.max_iter), e(prm.max_iter + 1), y(prm.max_iter),
        reduce(A.nrows()) {}

private:
  mutable matrix<val_t, loc_t> V, Z;
  mutable matrix<double, loc_t> H;
  mutable vector<double, loc_t> s, c, e, y;
  mutable scalar<double, loc_t> h;
  mutable scalar<double, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    auto dummy = vector<double, loc_t>(A.nrows());
    auto Ac = CSR<T, host>(A);
    auto resi = vector<double, loc_t>(A.nrows());
    dummy.copy(out);
    auto to64 = [](auto item) { return static_cast<double>(item); };
    bool flag = false;
    if (!init)
      A.residual(in, out, V(0));
    else
      V(0).copy(in);
    (!prm.normalized) ? h = reduce.norm(V(0).apply(to64)) : h.fill(1.0);
    h_nrm_r.copy(e(0).copy(h));
    (!prm.normalized) ? V(0) = h.inv() * V(0) : V(0);
    if (cond && cond(0, h_nrm_r[0])) {
      if constexpr (init) out.fill(0.0);
      return solve_res_t{true, 0, h_nrm_r[0]};
    }
    int j = 0;
    for (; j < prm.max_iter; j++) {
      M.apply(V(j), Z(j));
      Ac.apply(Z(j), V(j + 1));
      for (int k = 0; k <= j; k++) {
        H(k, j) = reduce.dot(V(j + 1).apply(to64), V(k).apply(to64));
      }
      V(j + 1) -= V * H(j).slice(j + 1);
      H(j + 1, j).copy(h = reduce.norm(V(j + 1).apply(to64)));
      V(j + 1) = V(j + 1) * h.inv();
      impl::full_grot(c, s, H(j), j);
      impl::final_grot(c, s, e, j);

      impl::trsv(H, e, y, prm.max_iter + 1, j + 1);
      if constexpr (init)
        dummy = Z * y.slice(j + 1);
      else
        dummy = out + Z * y.slice(j + 1);
      A.residual(V(0), dummy, resi);
      h_nrm_r = reduce.norm(resi);

      // h_nrm_r.copy(e(j + 1));
      if (cond && cond(j + 1, std::abs(h_nrm_r[0]))) {
        j++;
        flag = true;
        break;
      }
    }
    impl::trsv(H, e, y, prm.max_iter + 1, j);
    if constexpr (init)
      out = Z * y.slice(j);
    else
      out += Z * y.slice(j);
    return flag ? solve_res_t{true, j, h_nrm_r[0]}
                : solve_res_t{false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return A.ncols(); }
  idx_t ncols_impl() const { return A.nrows(); }

  friend struct model::is_operator<dFGMRES<T, Op, Pre>>;
  friend struct model::is_solver<dFGMRES<T, Op, Pre>>;
};
#endif

} // namespace senk

#endif