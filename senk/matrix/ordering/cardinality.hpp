#ifndef SENK_MATRIX_ORDERING_CARDINAL_HPP
#define SENK_MATRIX_ORDERING_CARDINAL_HPP

#include <algorithm>

#include "senk/core/sort.hpp"
#include "senk/matrix/ordering/permutation.hpp"

namespace senk {

namespace impl {

struct cardinal_ordering_params {
  sort::order order = sort::order::asc;
};

} // namespace impl

struct Cardinal : Reordering<host>, has_params<impl::cardinal_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  Cardinal(const CSR<T, host> &A, Params prm = Params{})
      : Reordering<host>(get_reorderer(A, prm)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A, Params prm) {
    auto card = std::vector<std::pair<int, int>>(A.nrows());
    for (int i = 0; i < A.nrows(); i++)
      card[i] = {A.rptr[i + 1] - A.rptr[i], i};

    if (prm.order == sort::order::asc)
      std::stable_sort(card.begin(), card.end(),
          [](auto left, auto right) { return left.first < right.first; });
    else
      std::stable_sort(card.begin(), card.end(),
          [](auto left, auto right) { return left.first > right.first; });
    auto p = Permutation<host>(A.nrows());
    auto pt = Permutation<host>(A.nrows());

    for (int i = 0; i < A.nrows(); i++) {
      p[i] = card[i].second;
      pt[p[i]] = i;
    }
    return {A.attr, p, pt};
  }
};

} // namespace senk

#endif