#ifndef SENK_MODEL_HPP
#define SENK_MODEL_HPP

#include <functional>

#include "senk/core/tensor.hpp"
#include "senk/matrix/base.hpp"

namespace senk {

template <class Derived>
struct has_shape {
  template <typename D = Derived>
  typename D::idx_t nrows() const {
    return static_cast<Derived *>(this)->nrows();
  }
  template <typename D = Derived>
  typename D::idx_t ncols() const {
    return static_cast<Derived *>(this)->ncols();
  }
};

template <class P>
struct has_params {
  struct Params : P {};
};

struct solve_res_t {
  bool is_solved;
  int32_t res_iter;
  double res_nrm2;
  void operator+=(const solve_res_t &in) {
    is_solved = is_solved || in.is_solved;
    res_iter = res_iter + in.res_iter;
    res_nrm2 = in.res_nrm2;
  }
};

namespace model {

template <class Derived>
struct is_operator {

  template <typename T, typename in_t, typename out_t, typename loc_t,
      typename = void>
  struct has_apply_impl : std::false_type {};
  template <typename T, typename in_t, typename out_t, typename loc_t>
  struct has_apply_impl<T, in_t, out_t, loc_t,
      std::void_t<decltype(&T::template apply_impl<in_t, out_t>)>>
      : std::true_type {};

  template <typename T, typename rhs_t, typename in_t, typename out_t,
      typename loc_t, typename = void>
  struct has_residual_impl : std::false_type {};
  template <typename T, typename rhs_t, typename in_t, typename out_t,
      typename loc_t>
  struct has_residual_impl<T, rhs_t, in_t, out_t, loc_t,
      std::void_t<decltype(&T::template residual_impl<rhs_t, in_t, out_t>)>>
      : std::true_type {};

  template <typename T, bool init, typename in_t, typename out_t,
      typename loc_t, typename = void>
  struct has_solve_impl : std::false_type {};
  template <typename T, bool init, typename in_t, typename out_t,
      typename loc_t>
  struct has_solve_impl<T, init, in_t, out_t, loc_t,
      std::void_t<decltype(&T::template solve_impl<init, in_t, out_t>)>>
      : std::true_type {};

  template <typename in_t, typename out_t, typename D = Derived>
  void apply(const vector<in_t, typename D::loc_t> &in,
      vector<out_t, typename D::loc_t> &out) const {
    if constexpr (has_apply_impl<D, in_t, out_t, typename D::loc_t>::value) {
      static_cast<const Derived *>(this)->apply_impl(in, out);
    } else if constexpr (has_solve_impl<D, true, in_t, out_t,
                             typename D::loc_t>::value) {
      static_cast<const Derived *>(this)->template solve_impl<true>(
          in, out, nullptr);
    } else {
      static_cast<const Derived *>(this)->apply_impl(in, out);
    }
  }
  template <typename in_t, typename out_t, typename D = Derived>
  void apply(const vector<in_t, typename D::loc_t> &in,
      vector<out_t, typename D::loc_t> &&out) const {
    apply(in, out);
  }
  template <typename rhs_t, typename in_t, typename out_t, typename D = Derived>
  void residual(const vector<rhs_t, typename D::loc_t> &rhs,
      const vector<in_t, typename D::loc_t> &in,
      vector<out_t, typename D::loc_t> &out) const {
    if constexpr (has_residual_impl<D, rhs_t, in_t, out_t,
                      typename D::loc_t>::value) {
      static_cast<const Derived *>(this)->residual_impl(rhs, in, out);
    } else {
      static_cast<const Derived *>(this)->apply_impl(in, out);
      out = rhs - out;
    }
  }
  template <typename rhs_t, typename in_t, typename out_t, typename D = Derived>
  void residual(const vector<rhs_t, typename D::loc_t> &rhs,
      const vector<in_t, typename D::loc_t> &in,
      vector<out_t, typename D::loc_t> &&out) const {
    residual(rhs, in, out);
  }

  template <typename D = Derived>
  typename D::idx_t nrows() const {
    return static_cast<const Derived *>(this)->nrows_impl();
  }
  template <typename D = Derived>
  typename D::idx_t ncols() const {
    return static_cast<const Derived *>(this)->ncols_impl();
  }
};

template <class Derived>
struct is_solver {
  template <typename in_t, typename out_t, typename D = Derived>
  solve_res_t solve(const vector<in_t, typename D::loc_t> &in,
      vector<out_t, typename D::loc_t> &out,
      const std::function<bool(int, double)> &cond = nullptr) const {
    return static_cast<const Derived *>(this)->template solve_impl<false>(
        in, out, cond);
  }
  template <typename D = Derived>
  typename D::idx_t nrows() const {
    return static_cast<const Derived *>(this)->nrows_impl();
  }
  template <typename D = Derived>
  typename D::idx_t ncols() const {
    return static_cast<const Derived *>(this)->ncols_impl();
  }
};

template <class Derived>
struct is_invertible {
  template <typename in_t, typename out_t, typename D = Derived>
  void inverse(const vector<in_t, typename D::loc_t> &in,
      vector<out_t, typename D::loc_t> &out) const {
    static_cast<const Derived *>(this)->inverse_impl(in, out);
  }
  template <typename in_t, typename out_t, typename D = Derived>
  void inverse(const vector<in_t, typename D::loc_t> &in,
      vector<out_t, typename D::loc_t> &&out) const {
    inverse(in, out);
  }
};

} // namespace model

namespace impl {

template <class L, class F, typename I = index_t>
struct lambda_operator : public has_idx_t<I>,
                         public has_loc_t<L>,
                         public model::is_operator<lambda_operator<L, F>> {
  using typename has_idx_t<I>::idx_t;
  using typename has_loc_t<L>::loc_t;

  lambda_operator(const std::array<idx_t, 2> &shape, F func)
      : shape(shape), func(func) {}

private:
  std::array<idx_t, 2> shape;
  F func;

  template <typename in_t, typename out_t>
  void apply_impl(const vector<in_t, L> &in, vector<out_t, L> &out) const {
    func(in, out);
  }

  idx_t nrows_impl() const { return shape[0]; }
  idx_t ncols_impl() const { return shape[1]; }

  friend struct model::is_operator<lambda_operator<L, F>>;
};

template <typename T, class Lo, class Ro>
struct dual_operator : public has_val_t<T>,
                       public has_idx_t<typename Lo::idx_t>,
                       public has_loc_t<typename Lo::loc_t>,
                       public model::is_operator<dual_operator<T, Lo, Ro>> {
  static_assert(std::is_same_v<typename Lo::loc_t, typename Ro::loc_t>);
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Lo::loc_t>::loc_t;
  using typename has_idx_t<typename Lo::idx_t>::idx_t;

  Lo L;
  Ro R;

  dual_operator(const Lo &L, const Ro &R) : L(L), R(R), tmp(L.nrows()) {}

private:
  mutable vector<val_t, loc_t> tmp;

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    R.apply(in, tmp);
    L.apply(tmp, out);
  }

  idx_t nrows_impl() const { return L.nrows(); }
  idx_t ncols_impl() const { return R.ncols(); }

  friend struct model::is_operator<dual_operator<T, Lo, Ro>>;
};

template <typename T, class L>
struct diag_operator : public has_val_t<T>,
                       public has_idx_t<index_t>,
                       public has_loc_t<L>,
                       public model::is_operator<diag_operator<T, L>>,
                       public model::is_invertible<diag_operator<T, L>> {
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  using typename has_idx_t<index_t>::idx_t;

  vector<val_t, loc_t> diag;
  vector<val_t, loc_t> dinv;

  template <typename Tin, class Lin>
  diag_operator(const vector<Tin, Lin> &in) : diag(in), dinv(in.shape(0)) {
    dinv.copy(diag).inv();
  }

  template <typename in_t, typename out_t>
  void inverse_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    out = dinv * in;
  }

private:
  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    out = diag * in;
  }

  idx_t nrows_impl() const { return diag.shape(0); }
  idx_t ncols_impl() const { return diag.shape(0); }

  friend struct model::is_operator<diag_operator<T, L>>;
  friend struct model::is_invertible<diag_operator<T, L>>;
};

} // namespace impl

template <typename T, class Lo, class Ro>
auto concat(Lo &&l, Ro &&r) {
  return impl::dual_operator<T, std::decay_t<Lo>, std::decay_t<Ro>>(
      std::forward<Lo>(l), std::forward<Ro>(r));
}

template <class L, class F, typename I = index_t>
auto lambda(const std::array<I, 2> shape, F &&func) {
  return impl::lambda_operator<L, std::decay_t<F>, I>(
      shape, std::forward<F>(func));
}

template <typename T, class L>
auto diagonal(const vector<T, L> &in) {
  return impl::diag_operator<T, L>(in);
}

} // namespace senk

#endif // SENK_MODEL_HPP