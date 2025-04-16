#ifndef UMINEKO_SPARSE_PRECONDITIONER_AINV_HPP
#define UMINEKO_SPARSE_PRECONDITIONER_AINV_HPP

#include "umineko-sparse/matrix/csr.hpp"
#include "umineko-sparse/models.hpp"
#include "umineko-sparse/spvec.hpp"

#include "umineko-sparse/solver/richardson.hpp"

namespace kmm {

namespace impl {

template <typename T> CSR<T, host> csr_to_csc(const CSR<T, host> &A);
template <typename T> CSR<T, host> convert_to_csc(SpVec<T> *z, idx_t N);
template <typename T>
std::tuple<CSR<T, host>, CSR<T, host>> left_ainv(
    const CSR<T, host> &in, T **d, double tol, double acc);
template <typename T>
std::tuple<CSR<T, host>, CSR<T, host>> left_sdainv(
    const CSR<T, host> &A, T **d, double tol, double acc);

} // namespace impl

namespace algorithm {

template <typename T, class L>
std::tuple<CSR<T, host>, CSR<T, host>> sdainv(
    const CSR<T, L> &A, double tol, double acc) {
  T *diag;
#if 0
  if (A.is_symmetric()) {
    auto [Wt, Z] = A.is_partitioned() ? left_sdainv_sym_gp(A, &diag, tol, acc)
                   : !A.is_colored()  ? left_sdainv_sym(A, &diag, tol, acc)
                   : A.is_colored(1)  ? left_sdainv_sym_mc(A, &diag, tol, acc)
                                      : left_sdainv_sym_bmc(A, &diag, tol, acc);
    for (idx_t i = 0; i < Wt.nrows(); i++) {
      for (idx_t j = Wt.rptr[i]; j < Wt.rptr[i + 1]; j++)
        Wt.val[j] /= diag[i];
    }
    free(diag);
    return {Z, Wt};
  } else {
    auto [Wt, Zt] = A.is_partitioned() ? left_sdainv_gp(A, &diag, tol, acc)
                    : !A.is_colored()  ? left_sdainv(A, &diag, tol, acc)
                    : A.is_colored(1)  ? left_sdainv_mc(A, &diag, tol, acc)
                                       : left_sdainv_bmc(A, &diag, tol, acc);
    auto Z = csr_to_csc(Zt);
    for (idx_t i = 0; i < Wt.nrows(); i++) {
      for (idx_t j = Wt.rptr[i]; j < Wt.rptr[i + 1]; j++)
        Wt.val[j] /= diag[i];
    }
    free(diag);
    return {Z, Wt};
  }
#endif
  auto [Wt, Zt] = impl::left_sdainv(A, &diag, tol, acc);
  for (idx_t i = 0; i < Zt.nrows(); i++) {
    for (idx_t j = Zt.rptr[i]; j < Zt.rptr[i + 1]; ++j)
      Zt.val[j] /= diag[i];
  }
  auto Z = impl::csr_to_csc(Zt);
  free(diag);
  return {Z, Wt};
}

template <typename T, class L>
std::tuple<CSR<T, host>, CSR<T, host>> ainv(
    const CSR<T, L> &in, double tol, double acc) {
  static_assert(std::is_same_v<host, L>, "ainv must be on host at this time");
  T *diag;
#if 0
  if (A.is_symmetric()) {
    auto [Wt, Z] = A.is_partitioned() ? left_ainv_sym_gp(A, &diag, tol, acc)
                                      : left_ainv_sym(A, &diag, tol, acc);
    for (idx_t i = 0; i < Wt.nrows(); i++) {
      for (idx_t j = Wt.rptr[i]; j < Wt.rptr[i + 1]; j++) Wt.val[j] /= diag[i];
    }
    free(diag);
    return {Z, Wt};
  } else {
    auto [Wt, Zt] = A.is_partitioned() ? left_ainv_gp(A, &diag, tol, acc)
                                       : left_ainv(A, &diag, tol, acc);
    auto Z = csr_to_csc(Zt);
    for (idx_t i = 0; i < Wt.nrows(); i++) {
      for (idx_t j = Wt.rptr[i]; j < Wt.rptr[i + 1]; j++) Wt.val[j] /= diag[i];
    }
    free(diag);
    return {Z, Wt};
  }
#endif
  auto [Wt, Zt] = impl::left_ainv(in, &diag, tol, acc);
  for (idx_t i = 0; i < Zt.nrows(); i++) {
    for (idx_t j = Zt.rptr[i]; j < Zt.rptr[i + 1]; ++j)
      Zt.val[j] /= diag[i];
  }
  auto Z = impl::csr_to_csc(Zt);
  free(diag);
  return {Z, Wt};
}

} // namespace algorithm

template <typename T, class L> struct IZW {
  struct factories {
    std::function<Operator<L>(const CSR<double, host> &)> z;
    std::function<Operator<L>(const CSR<double, host> &)> w;
  };

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    W.operate(in, temp);
    Z.operate(temp, out);
  }

  [[nodiscard]] idx_t nrows() const { return Z.nrows(); }
  [[nodiscard]] idx_t ncols() const { return W.ncols(); }

protected:
  Operator<L> Z;
  Operator<L> W;
  vector<T, L> temp;

  IZW(std::tuple<CSR<double, host>, CSR<double, host>> &&t, factories factory)
      : Z(factory.z(std::get<0>(t))), W(factory.w(std::get<1>(t))), temp(W.nrows()) {
  }
};

template <typename T, class L> struct SDAINV : IZW<T, L> {
  struct params {
    double tol = 0.1;
    double acc = 1.0;
  };

  SDAINV(const CSR<double, host> &A, params param,
      typename IZW<T, L>::factories factory =
          {Operator<L>::template factory<CSR<T, L>>(),
              Operator<L>::template factory<CSR<T, L>>()})
      : IZW<T, L>(algorithm::sdainv(A, param.tol, param.acc), factory) {}

  template <typename _>
  static std::function<Solver<L>(const CSR<_, host> &in)> Solver(params param) {
    return [param](const CSR<_, host> &in) {
      return kmm::Solver<L>(Richardson<T, L>(in, SDAINV(in, param), {2, 1.0}));
    };
  }
};

template <typename T, class L> struct AINV : IZW<T, L> {
  struct params {
    double tol = 0.1;
    double acc = 1.0;
  };

  AINV(const CSR<double, host> &A, params param,
      typename IZW<T, L>::factories factory =
          {Operator<L>::template factory<CSR<T, L>>(),
              Operator<L>::template factory<CSR<T, L>>()})
      : IZW<T, L>(algorithm::ainv(A, param.tol, param.acc), factory) {}

  template <typename _>
  static std::function<Solver<L>(const CSR<_, host> &in)> Solver(params param) {
    return [param](const CSR<_, host> &in) {
      return kmm::Solver<L>(Richardson<T, L>(in, AINV(in, param), {2, 1.0}));
    };
  }
};

namespace impl {

template <typename T>
std::tuple<CSR<T, host>, CSR<T, host>> left_ainv(
    const CSR<T, host> &A, T **d, double tol, double acc) {
  // printf("# AINV\n");
  auto At = csr_to_csc(A);
  auto N = A.nrows();
  *d = memory<host>::alloc<T>(N);
  auto *z = memory<host>::alloc<SpVec<T>>(N);
  for (idx_t i = 0; i < N; i++) {
    new (&z[i]) SpVec<T>(8);
    z[i].append({1.0, i});
  }
  auto *w = memory<host>::alloc<SpVec<T>>(N);
  for (idx_t i = 0; i < N; i++) {
    new (&w[i]) SpVec<T>(8);
    w[i].append({1.0, i});
  }
  auto *h_z = memory<host>::alloc<SpVec<T>>(N);
  for (idx_t i = 0; i < N; i++) {
    new (&h_z[i]) SpVec<T>(8);
    h_z[i].append({1.0, i});
  }
  auto *h_w = memory<host>::alloc<SpVec<T>>(N);
  for (idx_t i = 0; i < N; i++) {
    new (&h_w[i]) SpVec<T>(8);
    h_w[i].append({1.0, i});
  }
  auto temp = SpVec<T>(8);
  auto flag_z = vector<idx_t, host>(N).fill(0.0);
  auto flag_w = vector<idx_t, host>(N).fill(0.0);
  for (idx_t i = 0; i < N; i++) {
    idx_t a_len = A.rptr[i + 1] - A.rptr[i];
    idx_t off = A.rptr[i];
    idx_t at_len = At.rptr[i + 1] - At.rptr[i];
    idx_t c_off = At.rptr[i];

    flag_w[i] = 0;
    for (idx_t j = 0; j < at_len; j++) {
      idx_t row = At.idx[c_off + j];
      if (row >= i)
        break;
      for (idx_t k = 0; k < h_z[row].size(); k++) {
        idx_t col = h_z[row].elems[k].second;
        if (flag_w[col] == i)
          continue;
        auto valu = z[col].dot(&A.val.raw()[off], &A.idx.raw()[off], a_len);
        auto alpha = -valu / (*d)[col];
        temp.set_axpy_ainv(alpha, w[col], w[i], tol, h_w, col);
        w[i].set(temp);
        flag_w[col] = i;
      }
    }

    flag_z[i] = 0;
    for (idx_t j = 0; j < a_len; j++) {
      idx_t row = A.idx[off + j];
      if (row >= i)
        break;
      for (idx_t k = 0; k < h_w[row].size(); k++) {
        idx_t col = h_w[row].elems[k].second;
        if (flag_z[col] == i)
          continue;
        auto valu = w[col].dot(&At.val.raw()[c_off], &At.idx.raw()[c_off], at_len);
        auto alpha = -valu / (*d)[col];
        temp.set_axpy_ainv(alpha, z[col], z[i], tol, h_z, col);
        z[i].set(temp);
        flag_z[col] = i;
      }
    }

    idx_t end = w[i].size() - 1;
    w[i].elems[end].first *= acc;
    (*d)[i] = w[i].dot(&At.val.raw()[c_off], &At.idx.raw()[c_off], at_len);
    w[i].elems[end].first /= acc;
    if ((*d)[i] == 0) {
      (*d)[i] = 1;
    }
  }
  auto Zt = convert_to_csc(z, N);
  auto Wt = convert_to_csc(w, N);

  for (idx_t i = N; i > 0; i--)
    z[i - 1].~SpVec();
  memory<host>::free(z);
  for (idx_t i = N; i > 0; i--)
    w[i - 1].~SpVec();
  memory<host>::free(w);
  for (idx_t i = N; i > 0; i--)
    h_z[i - 1].~SpVec();
  memory<host>::free(h_z);
  for (idx_t i = N; i > 0; i--)
    h_w[i - 1].~SpVec();
  memory<host>::free(h_w);

  return {Wt, Zt};
}

template <typename T>
std::tuple<CSR<T, host>, CSR<T, host>> left_sdainv(
    const CSR<T, host> &A, T **d, double tol, double acc) {
  // printf("# SDAINV\n");
  auto At = csr_to_csc(A);
  auto N = A.nrows();
  *d = memory<host>::alloc<T>(N);
  auto *z = memory<host>::alloc<SpVec<T>>(N);
  for (idx_t i = 0; i < N; i++) {
    new (&z[i]) SpVec<T>(8);
    z[i].append({1.0, i});
  }
  auto *w = memory<host>::alloc<SpVec<T>>(N);
  for (idx_t i = 0; i < N; i++) {
    new (&w[i]) SpVec<T>(8);
    w[i].append({1.0, i});
  }
  auto temp = SpVec<T>(8);
  for (idx_t i = 0; i < N; i++) {
    idx_t a_len = A.rptr[i + 1] - A.rptr[i];
    idx_t off = A.rptr[i];
    idx_t at_len = At.rptr[i + 1] - At.rptr[i];
    idx_t c_off = At.rptr[i];
    idx_t z_len = 0, w_len = 0;
    while (A.idx[off + w_len] < i)
      w_len++;
    while (At.idx[c_off + z_len] < i)
      z_len++;
    for (idx_t j = 0; j < w_len; j++) {
      idx_t col = A.idx[off + j];
      auto valu = z[col].dot(&A.val.raw()[off], &A.idx.raw()[off], a_len);
      auto alpha = -valu / (*d)[col];
      temp.set_axpy(alpha, w[col], w[i], tol);
      w[i].set(temp);
    }
    for (idx_t j = 0; j < z_len; j++) {
      idx_t col = At.idx[c_off + j];
      auto valu = w[col].dot(&At.val.raw()[c_off], &At.idx.raw()[c_off], at_len);
      auto alpha = -valu / (*d)[col];
      temp.set_axpy(alpha, z[col], z[i], tol);
      z[i].set(temp);
    }
    idx_t end = w[i].size() - 1;
    w[i].elems[end].first *= acc;
    (*d)[i] = w[i].dot(&At.val.raw()[c_off], &At.idx.raw()[c_off], at_len);
    w[i].elems[end].first /= acc;
    if ((*d)[i] == 0) {
      (*d)[i] = 1;
    }
  }
  auto Zt = convert_to_csc(z, N);
  auto Wt = convert_to_csc(w, N);

  for (idx_t i = N - 1; i >= 0; i--)
    z[i].~SpVec();
  memory<host>::free(z);
  for (idx_t i = N - 1; i >= 0; i--)
    w[i].~SpVec();
  memory<host>::free(w);

  return {Wt, Zt};
}

template <typename T> inline CSR<T, host> csr_to_csc(const CSR<T, host> &A) {
  auto N = A.nrows();
  auto M = A.ncols();
  idx_t nnz = A.nnz();
  auto res = CSR<T, host>({M, N}, nnz);
  idx_t *num = (idx_t *)std::calloc(N, sizeof(idx_t));
  for (idx_t i = 0; i < N; i++) {
    for (idx_t j = A.rptr[i]; j < A.rptr[i + 1]; j++)
      num[A.idx[j]]++;
  }
  res.rptr[0] = 0;
  for (idx_t i = 0; i < M; i++) {
    res.rptr[i + 1] = res.rptr[i] + num[i];
    num[i] = 0;
  }
  for (idx_t i = 0; i < N; i++) {
    for (idx_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
      idx_t off = res.rptr[A.idx[j]];
      idx_t pos = num[A.idx[j]];
      res.idx[off + pos] = i;
      res.val[off + pos] = A.val[j];
      num[A.idx[j]]++;
    }
  }
  free(num);
  return res;
}

template <typename T> inline CSR<T, host> convert_to_csc(SpVec<T> *z, idx_t N) {
  auto ptr = vector<idx_t, host>(N + 1);
  ptr[0] = 0;
  for (idx_t i = 0; i < N; i++)
    ptr[i + 1] = ptr[i] + z[i].elems.size();
  auto res = CSR<T, host>({N, N}, ptr[N]);
  res.rptr = ptr;
  for (idx_t i = 0; i < N; i++) {
    idx_t off = res.rptr[i];
    idx_t len = z[i].elems.size();
    for (idx_t j = 0; j < len; j++) {
      res.val[off + j] = z[i].elems[j].first;
      res.idx[off + j] = z[i].elems[j].second;
    }
  }
  return res;
}

} // namespace impl

} // namespace kmm

#endif // UMINEKO_SPARSE_PRECONDITIONER_AINV_HPP
