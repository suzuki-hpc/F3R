#ifndef SENK_SOLVER_STATIONARY_HPP
#define SENK_SOLVER_STATIONARY_HPP

#include "senk/core/tensor.hpp"
#include "senk/models.hpp"
#include "senk/solver/base.hpp"

namespace senk {

template <typename T, typename Op, typename Pre>
struct Stationary : public has_val_t<T>,
                    public has_idx_t<typename Op::idx_t>,
                    public has_loc_t<typename Op::loc_t>,
                    public has_params<param::ss>,
                    public model::is_operator<Stationary<T, Op, Pre>>,
                    public model::is_solver<Stationary<T, Op, Pre>> {
  static_assert(std::is_same_v<typename Op::loc_t, typename Pre::loc_t>, "");
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<typename Op::loc_t>::loc_t;
  using typename has_idx_t<typename Op::idx_t>::idx_t;
  using model::is_operator<Stationary<T, Op, Pre>>::nrows;
  using model::is_operator<Stationary<T, Op, Pre>>::ncols;

  Params prm;

  Op R;
  Pre M;

  Stationary(const Op &R, const Pre &M, Params prm)
      : prm(prm), R(R), M(M), r(M.nrows()), tmp(M.nrows()), reduce(R.nrows()) {}

private:
  mutable vector<val_t, loc_t> r, tmp;
  mutable scalar<val_t, host> h_nrm_r;
  reducer<loc_t> reduce;

  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    M.apply(in, out);
    for (int k = 1; k < prm.max_iter; k++) {
      R.residual(in, out, tmp);
      M.apply(tmp, out);
    }
  }

  template <bool init, typename in_t, typename out_t>
  solve_res_t solve_impl(const vector<in_t, loc_t> &in,
      vector<out_t, loc_t> &out,
      const std::function<bool(int, double)> cond = nullptr) const {
    for (int k = 0; k < prm.max_iter; k++) {
      R.residual(in, out, tmp);
      if (cond) {
        M.inverse(out, r);
        r = tmp - r;
        h_nrm_r = reduce.norm(r);
        if (cond(k, h_nrm_r[0]))
          return {true, k, h_nrm_r[0]};
      }
      M.apply(tmp, out);
    }
    return {false, prm.max_iter, h_nrm_r[0]};
  }

  idx_t nrows_impl() const { return R.nrows(); }
  idx_t ncols_impl() const { return R.ncols(); }

  friend struct model::is_operator<Stationary<T, Op, Pre>>;
  friend struct model::is_solver<Stationary<T, Op, Pre>>;
#if 0
  static std::tuple<impl::diag_operator<val_t, loc_t>, CoefficientMatrix<L>>
  Jacobi(const CSR<double, host> &data) {
    auto v = vector<double, host>(data.nrows());
    auto res =
        CSR<double, host>(spmat(data.get_shape(), data.nnz() - data.nrows()));
    idx_t nnz = 0;
    res.rptr[0] = nnz;
    for (idx_t i = 0; i < data.nrows(); ++i) {
      for (idx_t j = data.rptr[i]; j < data.rptr[i + 1]; ++j) {
        if (data.idx[j] != i) {
          res.val[nnz] = data.val[j];
          res.idx[nnz++] = data.idx[j];
        } else {
          v[i] = 1. / data.val[j];
        }
      }
      res.rptr[i + 1] = nnz;
    }
    return {Diag(v), res};
  }

  static std::tuple<InvertibleOperator<L>, CoefficientMatrix<L>> WJacobi(
      const CSR<double, host> &data, double weight) {
    auto v = vector<double, host>(data.nrows());
    auto res = data.duplicate_val();
    for (idx_t i = 0; i < res.nrows(); ++i) {
      for (idx_t j = res.rptr[i]; j < res.rptr[i + 1]; ++j) {
        if (res.idx[j] == i) {
          v[i] = weight / res.val[j];
          res.val[j] -= res.val[j] / weight;
          break;
        }
      }
    }
    return {Diag(v), res};
  }

  static std::tuple<InvertibleOperator<L>, CoefficientMatrix<L>> GS(
      const CSR<double, host> &in) {
    idx_t l_nnz = 0, u_nnz = 0;
    for (idx_t i = 0; i < in.nrows(); i++) {
      for (idx_t j = in.rptr[i]; j < in.rptr[i + 1]; ++j) {
        l_nnz = (in.idx[j] <= i) ? l_nnz + 1 : l_nnz;
        u_nnz = (in.idx[j] > i) ? u_nnz + 1 : u_nnz;
      }
    }
    auto l = CSR<T, L>(in.get_shape(), l_nnz);
    l.spmat::copy_attrs(static_cast<spmat>(in));
    l.set_ld(impl::flags::is_lower);
    auto u = CSR<T, L>(in.get_shape(), u_nnz);
    u.spmat::copy_attrs(static_cast<spmat>(in));

    l_nnz = u_nnz = 0;
    l.rptr[0] = l_nnz;
    u.rptr[0] = u_nnz;
    for (idx_t i = 0; i < in.nrows(); i++) {
      for (idx_t j = in.rptr[i]; j < in.rptr[i + 1]; ++j) {
        if (in.idx[j] <= i) {
          l.val[l_nnz] = in.val[j];
          l.idx[l_nnz++] = in.idx[j];
        }
        if (in.idx[j] > i) {
          u.val[u_nnz] = in.val[j];
          u.idx[u_nnz++] = in.idx[j];
        }
        l.rptr[i + 1] = l_nnz;
        u.rptr[i + 1] = u_nnz;
      }
    }
    return {trsv::ld::Direct<CSR<double, host>>(l), u};
  }

  static std::tuple<InvertibleOperator<L>, CoefficientMatrix<L>> SOR(
      const CSR<double, host> &in, double weight) {
    idx_t l_nnz = 0, u_nnz = 0;
    for (idx_t i = 0; i < in.nrows(); i++) {
      for (idx_t j = in.rptr[i]; j < in.rptr[i + 1]; ++j) {
        l_nnz = (in.idx[j] <= i) ? l_nnz + 1 : l_nnz;
        u_nnz = (in.idx[j] >= i) ? u_nnz + 1 : u_nnz;
      }
    }
    auto l = CSR<T, L>(in.get_shape(), l_nnz);
    l.spmat::copy_attrs(static_cast<spmat>(in));
    l.set_ld(impl::flags::is_lower);
    auto u = CSR<T, L>(in.get_shape(), u_nnz);
    u.spmat::copy_attrs(static_cast<spmat>(in));

    l_nnz = u_nnz = 0;
    l.rptr[0] = l_nnz;
    u.rptr[0] = u_nnz;
    for (idx_t i = 0; i < in.nrows(); i++) {
      for (idx_t j = in.rptr[i]; j < in.rptr[i + 1]; ++j) {
        if (in.idx[j] <= i) {
          l.val[l_nnz] = (in.idx[j] == i) ? in.val[j] / weight : in.val[j];
          l.idx[l_nnz++] = in.idx[j];
        }
        if (in.idx[j] >= i) {
          u.val[u_nnz] =
              (in.idx[j] == i) ? in.val[j] * (1. - 1. / weight) : in.val[j];
          u.idx[u_nnz++] = in.idx[j];
        }
        l.rptr[i + 1] = l_nnz;
        u.rptr[i + 1] = u_nnz;
      }
    }
    return {trsv::ld::Direct<CSR<double, host>>(l), u};
  }
#endif
};

} // namespace senk

#endif // UMNK_SOLVER_STATIONARY_HPP
