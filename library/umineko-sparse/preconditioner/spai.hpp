#ifndef UMINEKO_SPARSE_PRECONDITIONER_SPAI_HPP
#define UMINEKO_SPARSE_PRECONDITIONER_SPAI_HPP

#include "umineko-sparse/matrix/csr.hpp"

namespace kmm {

template <typename T, class L> struct SPAI0 {
  vector<T, L> d;

  struct params {};

  SPAI0(const CSR<double, host> &A, [[maybe_unused]] params param) : d(A.nrows()) {
    auto t = vector<double, host>(A.nrows());
    for (idx_t i = 0; i < A.nrows(); i++) {
      double sum = 0;
      for (idx_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
        if (A.idx[j] == i)
          t[i] = A.val[j];
        sum += A.val[j] * A.val[j];
      }
      t[i] /= sum;
    }
    d.copy(t);
  }

  [[nodiscard]] idx_t nrows() const { return d.size(0); }
  [[nodiscard]] idx_t ncols() const { return d.size(0); }

  template <typename in_t, typename out_t>
  void operate(const in_t &in, out_t out) const {
    out = d * in;
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_PRECONDITIONER_SPAI_HPP
