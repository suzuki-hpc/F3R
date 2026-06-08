#ifndef SENK_MATRIX_ORDERING_CM_HPP
#define SENK_MATRIX_ORDERING_CM_HPP

#include <queue>

#include "senk/matrix/ordering/graph.hpp"
#include "senk/matrix/ordering/permutation.hpp"

namespace senk {

namespace impl {

struct cm_ordering_params {
  bool reverse = true;
};

} // namespace impl

struct CM : Reordering<host>, has_params<impl::cm_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  CM(const CSR<T, host> &A, Params prm)
      : Reordering<host>(get_reorderer(A, prm)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A, Params prm) {
    using idx_t = typename CSR<T, host>::idx_t;
    auto G = Graph(A);
    auto p = Permutation<host>(G.nrows());
    auto pt = Permutation<host>(G.ncols());

    auto degree_key = vector<idx_t, host>(G.nrows());
    auto degree = vector<idx_t, host>(G.nrows());
    for (idx_t i = 0; i < G.nrows(); i++)
      degree[i] = degree_key[i] = G.rptr[i + 1] - G.rptr[i];
    auto order =
        vector<idx_t, host>(G.nrows()).iota(0); // 添え字 To 元のインデックス
    sort::pack_sort<sort::order::asc>(
        0, G.nrows(), degree_key.raw(), order.raw());

    pt.fill(-2);
    auto cnt = idx_t{0};
    auto tmp_idx = std::vector<idx_t>();
    auto tmp_deg = std::vector<idx_t>();
    auto Q = std::queue<idx_t>();

    for (idx_t k = 0; k < G.nrows(); k++) {
      if (pt[order[k]] >= 0)
        continue;
      Q.push(order[k]);
      while (true) {
        if (Q.empty())
          break;
        idx_t i = Q.front(); // 元のインデックス
        Q.pop();
        if (pt[i] >= 0)
          continue;
        // printf("[%d] : %ld\n", i, Q.size());
        if (!prm.reverse)
          pt[i] = cnt;
        else
          pt[i] = G.nrows() - 1 - cnt;
        cnt++;
        tmp_idx.clear();
        tmp_deg.clear();
        for (auto j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
          if (pt[G.col[j]] >= -1)
            continue;
          tmp_idx.push_back(G.col[j]);
          tmp_deg.push_back(degree[G.col[j]]);
          pt[G.col[j]] = -1;
        }
        if (tmp_idx.size() == 0)
          continue;
        sort::pack_sort<sort::order::asc>(
            0, tmp_idx.size(), tmp_deg.data(), tmp_idx.data());
        // for (auto j = 0; j < tmp_idx.size(); j++)
        // printf("%d ", tmp_idx[j]);
        // printf("\n");
        for (auto &s : tmp_idx)
          Q.push(s);
      }
      if (cnt == G.nrows())
        break;
    }
    for (idx_t i = 0; i < G.nrows(); i++)
      p[pt[i]] = i;
    // return {A.spmat::duplicate(), p, pt};
    return {A.attr, p, pt};
  }
};

} // namespace senk

#endif