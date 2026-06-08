#ifndef SENK_MATRIX_ORDERING_COLOR_HPP
#define SENK_MATRIX_ORDERING_COLOR_HPP

#include "senk/matrix/ordering/graph.hpp"
#include "senk/matrix/ordering/permutation.hpp"

namespace senk {

namespace impl {

struct mc_ordering_params {
  coloring c_type;
  int max = 1000;
};

struct bmc_ordering_params {
  blocking b_type;
  int b_size;
  coloring c_type;
};

} // namespace impl

struct MC : Reordering<host>, has_params<impl::mc_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  MC(const CSR<T, host> &A, Params param)
      : Reordering<host>(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(
      const CSR<T, host> &A, Params param = {coloring::greedy, 1000}) {
    using idx_t = typename CSR<T, host>::idx_t;
    auto G = Graph(A).remove_direction();
    auto [c_map, c_size, c_num] = G.make_coloring_set(param.c_type, param.max);

    auto p = Permutation<host>(G.nrows());
    auto pt = Permutation<host>(G.ncols());
    auto num = vector<idx_t, host>(c_num);
    for (idx_t i = 0; i < A.nrows(); i++) {
      const auto cid = c_map[i] - 1;
      p[c_size[cid] + num[cid]] = i;
      pt[i] = c_size[cid] + num[cid];
      ++num[cid];
    }
    auto attr = attribute(A.attr.flags | impl::flags::is_colored, c_size);
    return {attr, p, pt};
  }
};

struct BMC : Reordering<host>, has_params<impl::bmc_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  BMC(const CSR<T, host> &A, Params param)
      : Reordering<host>(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A,
      Params param = {blocking::simple, 8, coloring::greedy}) {
    using idx_t = typename CSR<T, host>::idx_t;
    auto b_size = param.b_size;
    auto G =
        Graph(A).remove_direction().make_blocked_graph(b_size, param.b_type);
    auto [c_map, c_size, c_num] = G.make_coloring_set(param.c_type, 1000);

    auto p = Permutation<host>(A.nrows());
    auto pt = Permutation<host>(A.ncols());
    auto num = vector<idx_t, host>(c_num);
    for (idx_t i = 0; i < G.nrows(); i++) {
      const auto cid = c_map[i] - 1;
      for (int j = 0; j < b_size; j++) {
        auto to = c_size[cid] * b_size + num[cid];
        auto from = G.list[i * b_size + j];
        p[to] = from;
        pt[from] = to;
        ++num[cid];
      }
    }
    auto attr = attribute(A.attr.flags | impl::flags::is_colored, c_size);
    return {attr, p, pt};
  }
};

struct TMC : Reordering<host>, has_params<impl::bmc_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  TMC(const CSR<T, host> &A, Params param)
      : Reordering<host>(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A,
      Params param = {blocking::simple, 8, coloring::greedy}) {
    using idx_t = typename CSR<T, host>::idx_t;
    auto b_size = param.b_size;
    auto G = Graph(A).remove_direction().make_pseudo_blocked_graph(
        b_size, param.b_type);
    auto [c_map, c_size, c_num] = G.make_coloring_set(param.c_type, 1000);
    auto tmp = vector<idx_t, host>(A.nrows());
    auto p = Permutation<host>(A.nrows());
    auto pt = Permutation<host>(A.ncols());
    auto num = vector<idx_t, host>(c_num);
    for (idx_t i = 0; i < G.nrows(); i++) {
      const auto cid = c_map[i] - 1;
      for (int j = 0; j < b_size; j++) {
        tmp[c_size[cid] * b_size + num[cid]] = G.list[i * b_size + j];
        ++num[cid];
      }
    }
    for (idx_t i = 0; i < c_num; i++) {
      int size = c_size[i + 1] - c_size[i];
      int off = c_size[i] * b_size;
      for (idx_t j = 0; j < size; j++) {
        for (idx_t k = 0; k < b_size; k++) {
          p[off + k * size + j] = tmp[off + j * b_size + k];
          pt[tmp[off + j * b_size + k]] = off + k * size + j;
        }
      }
    }

    auto c_size2 = vector<idx_t, host>(c_num * b_size + 1);
    c_size2[0] = 0;
    for (int i = 0; i < c_num * b_size; i++) {
      auto seg = c_size[i / b_size + 1] - c_size[i / b_size];
      c_size2[i + 1] = c_size2[i] + seg;
    }

    auto attr = attribute(A.attr.flags | impl::flags::is_colored, c_size2);
    return {attr, p, pt};
  }
};

struct HBMC : Reordering<host>, has_params<impl::bmc_ordering_params> {
  using Reordering<host>::attr;
  using Reordering<host>::p;
  using Reordering<host>::pt;

  Params prm;

  template <typename T>
  HBMC(const CSR<T, host> &A, Params param)
      : Reordering<host>(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<attribute, Permutation<host>, Permutation<host>>
  get_reorderer(const CSR<T, host> &A,
      Params param = {blocking::simple, 8, coloring::greedy}) {
    using idx_t = typename CSR<T, host>::idx_t;
    auto b_size = param.b_size;
    auto G =
        Graph(A).remove_direction().make_blocked_graph(b_size, param.b_type);
    auto [c_map, c_size, c_num] = G.make_coloring_set(param.c_type, 1000);

    auto tmp = vector<idx_t, host>(A.nrows());
    auto p = Permutation<host>(A.nrows());
    auto pt = Permutation<host>(A.ncols());
    auto num = vector<idx_t, host>(c_num);
    for (idx_t i = 0; i < G.nrows(); i++) {
      const auto cid = c_map[i] - 1;
      for (int j = 0; j < b_size; j++) {
        tmp[c_size[cid] * b_size + num[cid]] = G.list[i * b_size + j];
        ++num[cid];
      }
    }
    for (idx_t i = 0; i < c_num; i++) {
      int size = c_size[i + 1] - c_size[i];
      int off = c_size[i] * b_size;
      for (idx_t j = 0; j < size; j++) {
        for (idx_t k = 0; k < b_size; k++) {
          p[off + k * size + j] = tmp[off + j * b_size + k];
          pt[tmp[off + j * b_size + k]] = off + k * size + j;
        }
      }
    }

    auto c_size2 = vector<idx_t, host>(c_num * b_size);
    c_size2[0] = 0;
    for (int i = 0; i < c_num * b_size; i++) {
      c_size2[i + 1] = c_size2[i] + c_size[i / b_size];
    }

    auto attr = attribute(A.attr.flags | impl::flags::is_colored, c_size2);
    return {attr, p, pt};
  }
};

} // namespace senk

#endif