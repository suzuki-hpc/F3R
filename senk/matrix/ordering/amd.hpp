#if (__has_include("amd.h"))
#ifndef SENK_MATRIX_ORDERING_AMD_HPP
#define SENK_MATRIX_ORDERING_AMD_HPP

#include "senk/matrix/ordering/permutation.hpp"

#include "amd.h"

namespace senk {

namespace impl {

struct amd_ordering_params {};

} // namespace impl

struct AMD : Reordering<host>, has_params<impl::amd_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  AMD(const CSR<T, host> &A, Params prm = Params{})
      : Reordering<host>(get_reorderer(A, prm)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A, Params prm) {
    using idx_t = typename CSR<T, host>::idx_t;
    idx_t *P = static_cast<idx_t *>(malloc(A.nrows() * sizeof(int)));
    double Control[AMD_CONTROL], Info[AMD_INFO];
    Control[AMD_DENSE] = -1.0;
    amd_defaults(Control);
    // int status =
    // amd_order(A.nrows(), A.rptr.raw(), A.col.raw(), P, Control, Info);
    amd_order(A.nrows(), A.rptr.raw(), A.col.raw(), P, Control, Info);
    auto p = Permutation<host>(A.nrows());
    auto pt = Permutation<host>(A.ncols());
    for (int i = 0; i < A.nrows(); i++) {
      p[i] = P[i];
      pt[P[i]] = i;
    }
    free(P);
    return {A.attr, p, pt};
  }
};

} // namespace senk

#endif

#endif