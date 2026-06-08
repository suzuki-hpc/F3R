#ifndef SENK_PRECONDITIONER_AMG_SA_HPP
#define SENK_PRECONDITIONER_AMG_SA_HPP

#include "senk/core/tensor.hpp"
#include "senk/matrix/csr.hpp"

namespace senk {

namespace impl {

inline auto aggregate(const CSR<double, host> &);
inline auto get_prolongator(const CSR<double, host> &,
    const vector<CSR<double, host>::idx_t, host> &, int);
inline auto get_restrictor(const CSR<double, host> &);
inline auto get_coarse_matrix(const CSR<double, host> &,
    const CSR<double, host> &, const CSR<double, host> &);

} // namespace impl

namespace impl {

inline auto aggregate(const CSR<double, host> &A, double eps) {
  using idx_t = std::remove_cvref_t<decltype(A)>::idx_t;
  auto max = vector<double, host>(A.nrows());
  for (idx_t i = 0; i < A.nrows(); i++) {
    for (idx_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
      if (i == A.col[j])
        continue;
      max[i] = std::max(max[i], abs(A.val[j]));
    }
    max[i] *= eps;
  }
  auto aggr = vector<idx_t, host>(A.nrows());
  idx_t now = 1;

  std::function<void(idx_t, int)> add_aggr;
  add_aggr = [&](idx_t u, int depth) {
    if (depth == 0)
      return;
    for (idx_t ptr = A.rptr[u]; ptr < A.rptr[u + 1]; ptr++) {
      auto v = A.col[ptr];
      if (aggr[v] != 0)
        continue;
      if (max[u] <= std::abs(A.val[ptr])) {
        aggr[v] = now;
        add_aggr(v, depth - 1);
      }
    }
  };

  for (idx_t i = 0; i < A.nrows(); i++) {
    if (aggr[i] != 0)
      continue;
    aggr[i] = now;
    add_aggr(i, 3);
    now++;
  }
  aggr -= 1;
  return std::make_tuple(aggr, now - 1);
}

inline auto spgemm(const CSR<double, host> &A, const CSR<double, host> &B) {
  auto N = A.nrows();
  auto t_rptr = vector<int, host>(N + 1);
  t_rptr[0] = 0;
  index_t res_nnz = 0;
#pragma omp parallel
  {
    auto marker = std::vector<index_t>(B.ncols(), -1);
#pragma omp for
    for (index_t i = 0; i < N; i++) {
      index_t nnz = 0;
      for (index_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
        index_t a_c = A.col[j];
        for (index_t k = B.rptr[a_c]; k < B.rptr[a_c + 1]; k++) {
          index_t b_c = B.col[k];
          if (marker[b_c] != i) {
            marker[b_c] = i;
            nnz++;
          }
        }
      }
      t_rptr[i + 1] = nnz;
    }
#pragma omp for reduction(+ : res_nnz)
    for (index_t i = 1; i < N + 1; i++) {
      res_nnz += t_rptr[i];
    }
  }
  auto t_val = vector<double, host>(res_nnz);
  auto t_idx = vector<int, host>(res_nnz);
  for (index_t i = 0; i < N; i++)
    t_rptr[i + 1] += t_rptr[i];

  index_t zero_cnt = 0;
#pragma omp parallel
  {
    auto marker = std::vector<index_t>(B.ncols(), -1);
#pragma omp for
    for (index_t i = 0; i < N; i++) {
      index_t row_beg = t_rptr[i];
      index_t row_end = row_beg;
      for (index_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
        index_t a_c = A.col[j];
        double a_v = A.val[j];

        for (index_t k = B.rptr[a_c]; k < B.rptr[a_c + 1]; k++) {
          index_t b_c = B.col[k];
          double b_v = B.val[k];

          if (marker[b_c] < row_beg) {
            marker[b_c] = row_end;
            t_idx[row_end] = b_c;
            t_val[row_end] = a_v * b_v;
            row_end++;
          } else {
            t_val[marker[b_c]] += a_v * b_v;
          }
        }
      }
      sort::pack_sort<sort::order::asc>(
          row_beg, row_end, t_idx.raw(), t_val.raw());
      for (index_t ii = row_beg; ii < row_end; ii++) {
        if (t_val[ii] == 0) {
#pragma omp critical
          zero_cnt++;
        }
      }
    }
  }

  if (zero_cnt != 0) {
    res_nnz = 0;
    index_t cnt = 0;
    for (index_t i = 0; i < N; i++) {
      for (index_t j = t_rptr[i] + cnt; j < t_rptr[i + 1]; j++) {
        if (t_val[j] == 0) {
          cnt++;
          continue;
        }
        t_val[res_nnz] = t_val[j];
        t_idx[res_nnz] = t_idx[j];
        res_nnz++;
      }
      t_rptr[i + 1] = res_nnz;
    }
  }

  auto res = CSR<double, host>({N, B.ncols()}, res_nnz);
  res.val.copy(t_val);
  res.col.copy(t_idx);
  res.rptr.copy(t_rptr);

  return res;
}

inline auto get_prolongator(const CSR<double, host> &A,
    const vector<CSR<double, host>::idx_t, host> &aggr, int deg) {
  using idx_t = std::remove_cvref_t<decltype(A)>::idx_t;
  auto n = A.nrows();
  auto m = deg;
  auto P = CSR<double, host>({n, m}, n);
  P.rptr.iota(0);
  P.val.fill(1.);
  P.col.copy(aggr);

  auto J = A.duplicate();
  for (idx_t i = 0; i < n; i++) {
    double diag = 0.;
    for (idx_t j = J.rptr[i]; j < J.rptr[i + 1]; j++) {
      if (J.col[j] == i) {
        diag = J.val[j];
        break;
      }
    }
    for (idx_t j = J.rptr[i]; j < J.rptr[i + 1]; j++) {
      J.val[j] = -J.val[j] / diag / 1.5;
      if (J.col[j] == i)
        J.val[j] += 1;
    }
  }
  return spgemm(J, P);
}

inline auto get_restrictor(const CSR<double, host> &I) {
  using idx_t = std::remove_cvref_t<decltype(I)>::idx_t;
  auto N = I.nrows();
  auto M = I.ncols();
  auto NNZ = I.rptr[I.nrows()];
  auto num = vector<id_t, host>(M).fill(0);
  for (index_t i = 0; i < N; i++) {
    for (index_t j = I.rptr[i]; j < I.rptr[i + 1]; j++)
      num[I.col[j]]++;
  }
  auto It = CSR<double, host>({M, N}, NNZ);
  It.rptr[0] = 0;
  for (index_t i = 0; i < M; i++) {
    It.rptr[i + 1] = It.rptr[i] + num[i];
    num[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    for (int j = I.rptr[i]; j < I.rptr[i + 1]; j++) {
      int st = It.rptr[I.col[j]];
      int off = num[I.col[j]];
      It.val[st + off] = I.val[j];
      It.col[st + off] = i;
      num[I.col[j]]++;
    }
  }
  return It;
}

inline auto get_coarse_matrix(const CSR<double, host> &A,
    const CSR<double, host> &I, const CSR<double, host> &It) {
  auto temp = spgemm(It, A);
  auto CA = spgemm(temp, I);
  return CA;
}

} // namespace impl

namespace expr {

inline auto sa(const CSR<double, host> &A) {
  auto [aggr, dim] = impl::aggregate(A, 0.);
  auto P = impl::get_prolongator(A, aggr, dim);
  auto R = impl::get_restrictor(P);
  auto C = impl::get_coarse_matrix(A, P, R);
  // printf("%d %d %d\n", C.nrows(), C.ncols(), C.nnz());
  return std::make_tuple(C, P, R);
}

} // namespace expr

} // namespace senk

#endif