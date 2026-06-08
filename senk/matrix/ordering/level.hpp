#ifndef SENK_MATRIX_ORDERING_LEVEL_HPP
#define SENK_MATRIX_ORDERING_LEVEL_HPP

#include "senk/matrix/ordering/graph.hpp"
#include "senk/matrix/ordering/permutation.hpp"

namespace senk {

namespace impl {

struct level_ordering_params {};

} // namespace impl

struct Level : Reordering<host>, has_params<impl::level_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  Level(const CSR<T, host> &A, Params param)
      : Reordering<host>(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A, Params param = {}) {
    using idx_t = typename CSR<T, host>::idx_t;
    auto G = Graph(A).remove_direction();

    int level_max = 0;
    auto level = vector<int, host>(G.nrows());
    for (idx_t i = 0; i < G.nrows(); i++) {
      int max = -1;
      for (int j = G.rptr[i]; j < G.rptr[i + 1]; j++) {
        if (G.col[j] == i)
          break;
        if (G.col[j] < i)
          max = (max < level[G.col[j]]) ? level[G.col[j]] : max;
      }
      level[i] = max + 1;
      level_max = std::max(level_max, level[i]);
    }
    printf("level: %d\n", level_max + 1);
    auto p = Permutation<host>(G.nrows());
    auto pt = Permutation<host>(G.ncols());
    p.iota(0);
    sort::pack_sort<sort::order::asc>(0, G.nrows(), level.raw(), p.raw());

    auto segm = vector<idx_t, host>(level_max + 2).fill(0);

    idx_t _l = 0;
    for (idx_t i = 0; i < G.nrows(); i++) {
      pt[p[i]] = i;
      if (_l != level[i]) {
        segm[_l + 1] = i;
        _l++;
      }
    }
    segm[level_max + 1] = G.nrows();

    auto attr = attribute(A.attr.flags | impl::flags::is_colored, segm);
    return {attr, p, pt};
  }
};

} // namespace senk

#endif