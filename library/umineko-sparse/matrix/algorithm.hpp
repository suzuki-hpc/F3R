#ifndef UMINEKO_SPARSE_MATRIX_ALGORITHM_HPP
#define UMINEKO_SPARSE_MATRIX_ALGORITHM_HPP

#include "umineko-sparse/matrix/csr.hpp"

namespace kmm::algorithm {

template <typename T, class L>
std::tuple<CSR<T, L>, CSR<T, L>> split(const CSR<T, L> &in) {
  static_assert(std::is_same_v<host, L>, "split must be on host");
  idx_t l_nnz = 0, u_nnz = 0;
  for (idx_t i = 0; i < in.nrows(); i++) {
    for (idx_t j = in.rptr[i]; j < in.rptr[i + 1]; ++j) {
      l_nnz = (in.idx[j] <= i) ? l_nnz + 1 : l_nnz;
      u_nnz = (in.idx[j] >= i) ? u_nnz + 1 : u_nnz;
    }
  }
  auto l = CSR<T, L>(in.get_shape(), l_nnz);
  l.spmat::copy_attrs(static_cast<spmat>(in));
  l.form = spmat::Form::lower;
  auto u = CSR<T, L>(in.get_shape(), u_nnz);
  u.spmat::copy_attrs(static_cast<spmat>(in));
  u.form = spmat::Form::d_upper;

  l_nnz = u_nnz = 0;
  l.rptr[0] = l_nnz;
  u.rptr[0] = u_nnz;
  for (idx_t i = 0; i < in.nrows(); i++) {
    for (idx_t j = in.rptr[i]; j < in.rptr[i + 1]; ++j) {
      if (in.idx[j] <= i) {
        l.val[l_nnz] = (in.idx[j] < i) ? in.val[j] : 1.0;
        l.idx[l_nnz++] = in.idx[j];
      }
      if (in.idx[j] >= i) {
        u.val[u_nnz] = in.val[j];
        u.idx[u_nnz++] = in.idx[j];
      }
      l.rptr[i + 1] = l_nnz;
      u.rptr[i + 1] = u_nnz;
    }
  }
  return {l, u};
}

template <typename T, class L> void scaling(CSR<T, L> &in) {
  auto diag = vector<T, host>(in.nrows()).fill(1);
  exec<L>::para_for(in.nrows(),
      [=, in_val = in.val, in_idx = in.idx, in_rptr = in.rptr](auto i) mutable {
        for (auto j = in_rptr[i]; j < in_rptr[i + 1]; ++j)
          if (in_idx[j] == idx_t(i))
            diag[i] = std::sqrt(std::abs(in_val[j]));
      });
  exec<L>::para_for(in.nrows(),
      [=, in_val = in.val, in_idx = in.idx, in_rptr = in.rptr](auto i) mutable {
        for (auto j = in_rptr[i]; j < in_rptr[i + 1]; ++j) {
          in_val[j] /= diag[i] * diag[in_idx[j]];
        }
      });
}

template <typename T, class L> void scaling_asym(CSR<T, L> &in) {
  exec<L>::para_for(
      in.nrows(), [=, in_val = in.val, in_rptr = in.rptr](auto i) mutable {
        T max = 0;
        for (auto j = in_rptr[i]; j < in_rptr[i + 1]; ++j) {
          if (std::abs(in_val[j]) > max)
            max = std::abs(in_val[j]);
        }
        for (auto j = in_rptr[i]; j < in_rptr[i + 1]; ++j)
          in_val[j] /= max;
      });
  in.sym = spmat::Sym::general;
}

template <typename T, class L> CSR<T, L> conjugate_transpose(CSR<T, L> &in) {
  static_assert(std::is_same_v<host, L>, "conjugate_transpose must be on host");
  auto res = CSR<T, L>(static_cast<spmat>(in));
  auto t_num = vector<idx_t, host>(in.ncols()).fill(0);
  for (idx_t i = 0; i < in.nnz(); ++i)
    ++t_num[in.idx[i]];
  res.rptr[0] = 0;
  for (idx_t i = 0; i < in.ncols(); ++i) {
    res.rptr[i + 1] = res.rptr[i] + t_num[i];
    t_num[i] = 0;
  }
  for (idx_t i = 0; i < in.nrows(); ++i) {
    for (idx_t j = in.rptr[i]; j < in.rptr[i + 1]; ++j) {
      auto off = res.rptr[in.idx[j]];
      auto pos = t_num[in.idx[j]];
      res.idx[off + pos] = i;
      if constexpr (impl::is_complex_v<T>)
        res.val[off + pos] = std::conj(in.val[j]);
      else
        res.val[off + pos] = in.val[j];
      ++t_num[in.idx[j]];
    }
  }
  return res;
}

} // namespace kmm::algorithm

#endif // UMINEKO_SPARSE_MATRIX_ALGORITHM_HPP
