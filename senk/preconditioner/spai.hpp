#ifndef SENK_PRECONDITIONER_HPP
#define SENK_PRECONDITIONER_HPP

#include "senk/core/tensor.hpp"
#include "senk/matrix/csr.hpp"
#include "senk/models.hpp"

namespace senk {

namespace param {

struct empty {};

} // namespace param

template <typename T, class L>
struct SPAI0 : public has_val_t<T>,
               public has_idx_t<index_t>,
               public has_loc_t<L>,
               public has_params<param::empty>,
               public model::is_operator<SPAI0<T, L>> {
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  using typename has_idx_t<index_t>::idx_t;

  Params prm;

  vector<T, L> d;

  SPAI0(const CSR<double, host> &A, [[maybe_unused]] Params prm = {})
      : d(A.nrows()) {
    auto t = vector<double, host>(A.nrows());
#pragma omp parallel for
    for (idx_t i = 0; i < A.nrows(); i++) {
      double sum = 0;
      for (idx_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
        if (A.col[j] == i)
          t[i] = A.val[j];
        sum += A.val[j] * A.val[j];
      }
      t[i] /= sum;
    }
    d.copy(t);
  }

private:
  template <typename in_t, typename out_t>
  void apply_impl(
      const vector<in_t, loc_t> &in, vector<out_t, loc_t> &out) const {
    out = d * in;
  }

  idx_t nrows_impl() const { return d.shape(0); }
  idx_t ncols_impl() const { return d.shape(0); }

  friend class model::is_operator<SPAI0<T, L>>;
};

} // namespace senk

#endif