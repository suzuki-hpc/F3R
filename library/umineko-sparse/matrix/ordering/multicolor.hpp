#ifndef UMINEKO_SPARSE_MATRIX_ORDERING_MULTICOLOR_HPP
#define UMINEKO_SPARSE_MATRIX_ORDERING_MULTICOLOR_HPP

#include "umineko-core/sort.hpp"
#include "umineko-sparse/matrix/ordering/graph.hpp"

namespace kmm {

struct MC : reorderer<host> {
  using reorderer::attr, reorderer::p, reorderer::pt;

  struct params {
    coloring c_type;
  };

  template <typename T>
  MC(const CSR<T, host> &A, params param) : reorderer(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<spmat, permuter<host>, permuter<host>> get_reorderer(
      const CSR<T, host> &A, params param) {
    auto G = Graph<host>(A).remove_direction();
    auto [c_map, c_size, c_num] = G.make_coloring_set(param.c_type);
    // print_color_size(c_size.ptr, c_num);
    auto p = permuter<host>(G.nrows());
    auto pt = permuter<host>(G.ncols());
    auto num = vector<idx_t, host>(c_num);
    for (idx_t i = 0; i < A.nrows(); i++) {
      const auto cid = c_map[i] - 1;
      p.pi[c_size[cid] + num[cid]] = i;
      pt.pi[i] = c_size[cid] + num[cid];
      ++num[cid];
    }
    auto attr = static_cast<spmat>(A);
    attr.pattern = spmat::Pattern::colored;
    attr.pattern_set = {1, c_num, c_size};
    return {attr, p, pt};
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_MATRIX_ORDERING_MULTICOLOR_HPP
