#ifndef SENK_MATRIX_ORDERING_LOCO_HPP
#define SENK_MATRIX_ORDERING_LOCO_HPP

#include "senk/matrix/ordering/permutation.hpp"

namespace senk {

namespace impl {

struct loco_ordering_params {};

} // namespace impl

struct LOCO : Reordering<host>, has_params<impl::loco_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  LOCO(const CSR<T, host> &A, Params prm = Params{})
      : Reordering<host>(get_reorderer(A, prm)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A, [[maybe_unused]] Params prm) {
    auto index = vector<int, host>(A.nrows()).iota(0);
    auto loc = vector<int, host>(A.nrows()).iota(0);
    auto flag = vector<bool, host>(A.nrows()).fill(true);
    int cnt = 0;
    for (int i = 0; i < A.nrows(); i++) {
      auto row = index[i];
      for (int j = A.rptr[row]; j < A.rptr[row + 1]; j++) {
        if (flag[A.col[j]]) {
          loc[index[cnt]] = loc[A.col[j]];
          auto tmp = index[cnt];
          index[cnt] = index[loc[A.col[j]]];
          index[loc[A.col[j]]] = tmp;
          flag[A.col[j]] = false;
          cnt++;
        }
      }
    }
    auto p = Permutation<host>(A.nrows());
    auto pt = Permutation<host>(A.ncols());
    for (int i = 0; i < A.nrows(); i++) {
      p[i] = index[i];
      pt[index[i]] = i;
    }
    return {A.attr, p, pt};
  }
};

} // namespace senk

#endif