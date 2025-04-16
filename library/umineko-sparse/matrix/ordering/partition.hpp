#ifndef UMINEKO_SPARSE_MATRIX_ORDERING_PARTITION_HPP
#define UMINEKO_SPARSE_MATRIX_ORDERING_PARTITION_HPP

#if __has_include(<metis.h>)
#include <metis.h>

#include "umineko-sparse/matrix/ordering/graph.hpp"

namespace kmm {

namespace impl {

template <class L> vector<idx_t, L> metis_kway(const Graph<L> &G, int n_parts);

}

struct KWAY : reorderer<host> {
  using reorderer::attr, reorderer::p, reorderer::pt;

  struct params {
    uint16_t n_parts;
  };

  template <typename T>
  KWAY(const CSR<T, host> &A, params param) : reorderer(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<spmat, permuter<host>, permuter<host>> get_reorderer(
      const CSR<T, host> &A, params param) {
    int p_num = param.n_parts;
    auto G = Graph<host>(A).remove_direction();
    auto p_map = impl::metis_kway(G, p_num);

    auto nums = vector<idx_t, host>(p_num).fill(0);
    for (int i = 0; i < G.nrows(); i++)
      ++nums[p_map[i]];

    for (int i = 0; i < G.nrows(); i++) {
      if (p_map[i] == p_num)
        continue;
      for (int j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
        if (G.idx[j] == i || p_map[G.idx[j]] == p_num)
          continue;
        if (p_map[G.idx[j]] != p_map[i]) {
          if (nums[p_map[i]] >= nums[p_map[G.idx[j]]]) {
            --nums[p_map[i]];
            p_map[i] = p_num;
            break;
          }
          --nums[p_map[G.idx[j]]];
          p_map[G.idx[j]] = p_num;
        }
      }
    }
    p_num++;

    auto p_size = vector<idx_t, host>(p_num + 1);
    for (int i = 0; i < A.nrows(); i++)
      ++p_size[p_map[i] + 1];
    for (int i = 0; i < p_num; i++)
      p_size[i + 1] += p_size[i];
    auto p = permuter<host>(G.nrows());
    auto pt = permuter<host>(G.ncols());
    auto num = vector<idx_t, host>(p_num);
    for (int i = 0; i < A.nrows(); i++) {
      int pid = p_map[i];
      p.pi[p_size[pid] + num[pid]] = i;
      pt.pi[i] = p_size[pid] + num[pid];
      ++num[pid];
    }
    auto attr = static_cast<spmat>(A);
    attr.pattern = spmat::Pattern::partitioned;
    attr.pattern_set = {1, p_num, p_size};
    return {attr, p, pt};
  }
};

struct KWAY2 : reorderer<host> {
  using reorderer::attr, reorderer::p, reorderer::pt;

  struct params {
    uint16_t n_parts;
  };

  template <typename T>
  KWAY2(const CSR<T, host> &A, params param) : reorderer(get_reorderer(A, param)) {}

private:
  template <typename T>
  static std::tuple<spmat, permuter<host>, permuter<host>> get_reorderer(
      const CSR<T, host> &A, params param) {
    int p_num = param.n_parts;
    auto G = Graph<host>(A).remove_direction();
    auto p_map = impl::metis_kway(G, p_num);
    auto p_size = vector<idx_t, host>(p_num + 1);
    for (int i = 0; i < A.nrows(); i++)
      ++p_size[p_map[i] + 1];
    for (int i = 0; i < p_num; i++)
      p_size[i + 1] += p_size[i];
    auto p = permuter<host>(G.nrows());
    auto pt = permuter<host>(G.ncols());
    auto num = vector<idx_t, host>(p_num);
    for (int i = 0; i < A.nrows(); i++) {
      int pid = p_map[i];
      p.pi[p_size[pid] + num[pid]] = i;
      pt.pi[i] = p_size[pid] + num[pid];
      ++num[pid];
    }
    auto attr = static_cast<spmat>(A);
    attr.pattern = spmat::Pattern::partitioned;
    attr.pattern_set = {1, p_num, p_size};
    return {attr, p, pt};
  }
};

namespace impl {

template <class L>
vector<idx_t, L> metis_kway(const Graph<L> &G, const int n_parts) {
  /* Adjacency matrix without diagonal elements */
  int N = G.nrows();
  auto xadj = vector<idx_t, host>(N + 1);
  auto adjncy = vector<idx_t, host>(G.rptr[N]);
  int cnt = 0;
  xadj[0] = cnt;
  for (int i = 0; i < N; i++) {
    for (int j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
      if (G.idx[j] == i)
        continue;
      adjncy[cnt] = G.idx[j];
      cnt++;
    }
    xadj[i + 1] = cnt;
  }
  idx_t nvtxs[1] = {N}; /* The number of vertices. */
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_UFACTOR] = 30;
  options[METIS_OPTION_SEED] = 0;

  idx_t ncon[1] = {1};
  idx_t nparts[1] = {n_parts};
  idx_t objval[1];
  auto part = vector<idx_t, host>(N);
  int metis_res = METIS_PartGraphKway(nvtxs, ncon, xadj.raw(), adjncy.raw(), NULL,
      NULL, NULL, nparts, NULL, NULL, options, objval, part.raw());
  // for (int i = 0; i < part.size(0); i++)
  //   printf("%d %d\n", i, part[i]);
#if 0
  {
    idx_t sepsize;
    int ret = METIS_ComputeVertexSeparator(
        nvtxs, xadj.raw(), adjncy.raw(), NULL, NULL, &sepsize, part.raw());
    if (ret == METIS_OK) {
      std::cout << "ノードセパレータサイズ: " << sepsize << "\n";
      std::cout << "分割結果:\n";
      for (int i = 0; i < nvtxs[0]; i++) {
        std::cout << "頂点 " << i << ": パーティション " << part[i] << "\n";
      }
    } else {
      std::cerr << "METISエラー\n";
    }
  }
#endif
  if (metis_res != 1) {
    printf("METIS Erorr\n");
    exit(1);
  }

  return part;
}

} // namespace impl

} // namespace kmm

#endif

#endif // UMINEKO_SPARSE_MATRIX_ORDERING_PARTITION_HPP
