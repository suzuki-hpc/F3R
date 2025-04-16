#ifndef UMINEKO_SPARSE_MATRIX_ORDERING_GRAPH_HPP
#define UMINEKO_SPARSE_MATRIX_ORDERING_GRAPH_HPP

#include <algorithm>

#include "umineko-core/tensor.hpp"
#include "umineko-sparse/matrix/base.hpp"

namespace kmm {

enum class coloring { greedy, cyclic, hybrid };
enum class blocking { simple, connect, stack };

template <typename> struct Graph;
template <typename> struct BlockGraph;

namespace impl {

template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> color_greedy(
    const Graph<L> &G);
template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> color_cyclic(
    const Graph<L> &G);
template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> color_hybrid(
    const Graph<L> &G);
template <class L> vector<idx_t, L> block_simple(const Graph<L> &G, idx_t b_size);
template <class L> vector<idx_t, L> block_connect(const Graph<L> &G, idx_t b_size);
template <class L> void block_stack(const Graph<L> &G, idx_t b_size, idx_t **b_map);
template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>> get_block_adjacency(
    const Graph<L> &G, const vector<idx_t, L> &b_map, const vector<idx_t, L> &b_list,
    const idx_t b_size);
template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>> get_pseudo_block_adjacency(
    const Graph<L> &G, const vector<idx_t, L> &b_map, const vector<idx_t, L> &b_list,
    const idx_t b_size);

} // namespace impl

template <class L> struct Graph : spmat {
  vector<idx_t, L> idx;
  vector<idx_t, L> rptr;

  explicit Graph(const spmat &base) : spmat(base), idx(nnz()), rptr(nrows() + 1) {}
  // template <typename _, class L2>
  // explicit Graph(const CSR<_, L2> &in) : spmat(in), idx(in.idx), rptr(in.rptr) {}
  template <typename _, class L2>
  explicit Graph(const CSR<_, L2> &in) : spmat(in), idx(nnz()), rptr(nrows() + 1) {
    idx.copy(in.idx);
    rptr.copy(in.rptr);
  }
  template <class L2>
  explicit Graph(const Graph<L2> &in) : spmat(in), idx(in.idx), rptr(in.rptr) {}
  Graph(const impl::shape &s, const idx_t nz)
      : spmat(s, nz), idx(nnz()), rptr(nrows() + 1) {}
  Graph(const Graph &in) = default;
  Graph &operator=(const Graph &in) = default;

  template <class L2> Graph &copy(const Graph<L2> &in) {
    spmat::copy(static_cast<spmat>(in));
    idx.copy(in.idx);
    rptr.copy(in.rptr);
    return *this;
  }
  Graph clone() const { return Graph(static_cast<spmat>(*this)).copy(*this); }
  Graph transpose() {
    static_assert(std::is_same_v<L, host>, "device is not supported at this time");
    if (is_symmetric())
      return *this;
    auto t_rptr = vector<idx_t, L>(rptr.size(0));
    auto t_idx = vector<idx_t, L>(idx.size(0));
    auto t_num = vector<idx_t, L>(ncols()).fill(0);
    for (idx_t i = 0; i < nnz(); ++i) {
      ++t_num[idx[i]];
    }
    t_rptr[0] = 0;
    for (idx_t i = 0; i < ncols(); ++i) {
      t_rptr[i + 1] = t_rptr[i] + t_num[i];
      t_num[i] = 0;
    }
    for (idx_t i = 0; i < nrows(); ++i) {
      for (idx_t j = rptr[i]; j < rptr[i + 1]; ++j) {
        auto off = t_rptr[idx[j]];
        auto pos = t_num[idx[j]];
        t_idx[off + pos] = i;
        ++t_num[idx[j]];
      }
    }
    idx.copy(t_idx);
    rptr.copy(t_rptr);
    return *this;
  }
  Graph remove_direction() {
    if (is_symmetric())
      return *this;
    auto tmp = clone().transpose();
    idx_t nnz = 0;
    for (idx_t i = 0; i < ncols(); i++) {
      idx_t ptr = rptr[i];
      idx_t tmp_ptr = tmp.rptr[i];
      while (ptr < rptr[i + 1] || tmp_ptr < tmp.rptr[i + 1]) {
        idx_t col = (ptr < rptr[i + 1]) ? idx[ptr] : ncols();
        idx_t tmp_col = (tmp_ptr < tmp.rptr[i + 1]) ? tmp.idx[tmp_ptr] : ncols();
        ptr = (col <= tmp_col) ? ptr + 1 : ptr;
        tmp_ptr = (col >= tmp_col) ? tmp_ptr + 1 : tmp_ptr;
        nnz++;
      }
    }
    auto res = Graph(get_shape(), nnz);
    res.spmat::copy(static_cast<spmat>(*this));
    res.sym = Sym::symmetric;
    nnz = 0;
    res.rptr[0] = nnz;
    for (idx_t i = 0; i < ncols(); i++) {
      idx_t ptr = rptr[i];
      idx_t tmp_ptr = tmp.rptr[i];
      while (ptr < rptr[i + 1] || tmp_ptr < tmp.rptr[i + 1]) {
        idx_t col = (ptr < rptr[i + 1]) ? idx[ptr] : ncols();
        idx_t tmp_col = (tmp_ptr < tmp.rptr[i + 1]) ? tmp.idx[tmp_ptr] : ncols();
        res.idx[nnz++] = (col <= tmp_col) ? col : tmp_col;
        ptr = (col <= tmp_col) ? ptr + 1 : ptr;
        tmp_ptr = (col >= tmp_col) ? tmp_ptr + 1 : tmp_ptr;
      }
      res.rptr[i + 1] = nnz;
    }
    return res;
  }

  std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> make_coloring_set(
      const coloring type) {
    if (type == coloring::greedy)
      return impl::color_greedy(*this);
    if (type == coloring::cyclic)
      return impl::color_cyclic(*this);
    // if (type == Coloring::Hybrid) {
    return impl::color_hybrid(*this);
  }

  BlockGraph<L> make_blocked_graph(idx_t b_size, const blocking type) {
    auto b_map = [&]() {
      if (type == blocking::simple)
        return impl::block_simple(*this, b_size);
      if (type == blocking::connect)
        return impl::block_connect(*this, b_size);
      return impl::block_connect(*this, b_size);
    }();
    auto b_list = vector<idx_t, L>(nrows());
    auto offset = vector<idx_t, L>(nrows() / b_size);
    for (idx_t i = 0; i < nrows(); i++) {
      auto bid = b_map[i] - 1;
      b_list[bid * b_size + offset[bid]] = i;
      ++offset[bid];
    }
    auto [b_idx, b_rptr] = impl::get_block_adjacency(*this, b_map, b_list, b_size);
    auto nnz = b_rptr[nrows() / b_size];
    auto res =
        BlockGraph<L>(spmat({nrows() / b_size, ncols() / b_size}, nnz), b_size);
    res.b_list = b_list;
    res.idx = b_idx;
    res.rptr = b_rptr;
    return res;
  }

  BlockGraph<L> make_pseudo_blocked_graph(idx_t b_size, const blocking type) {
    auto b_map = [&]() {
      if (type == blocking::simple)
        return impl::block_simple(*this, b_size);
      if (type == blocking::connect)
        return impl::block_connect(*this, b_size);
      return impl::block_connect(*this, b_size);
    }();
    auto b_list = vector<idx_t, L>(nrows());
    auto offset = vector<idx_t, L>(nrows() / b_size);
    for (idx_t i = 0; i < nrows(); i++) {
      auto bid = b_map[i] - 1;
      b_list[bid * b_size + offset[bid]] = i;
      ++offset[bid];
    }
    auto [b_idx, b_rptr] =
        impl::get_pseudo_block_adjacency(*this, b_map, b_list, b_size);
    auto nnz = b_rptr[nrows() / b_size];
    auto res =
        BlockGraph<L>(spmat({nrows() / b_size, ncols() / b_size}, nnz), b_size);
    res.b_list = b_list;
    res.idx = b_idx;
    res.rptr = b_rptr;
    return res;
  }
};

template <class L> struct BlockGraph : Graph<L> {
  using Graph<L>::idx;
  using Graph<L>::rptr;
  idx_t b_size;
  vector<idx_t, L> b_list; // list from index to block id.

  BlockGraph(const spmat &base, idx_t b_size)
      : Graph<L>(base), b_size(b_size), b_list(Graph<L>::nrows() * b_size) {}
};

template <class L> struct permuter {
  static_assert(std::is_same_v<L, host>, "device is not supported at this time");
  vector<idx_t, L> pi;
  explicit permuter(idx_t n) : pi(n) { pi.iota(0); }
  template <typename _> void apply_from_right(CSR<_, L> &mat) {
    for (idx_t i = 0; i < mat.nrows(); i++) {
      for (idx_t j = mat.rptr[i]; j < mat.rptr[i + 1]; ++j)
        mat.idx[j] = pi[mat.idx[j]];
      sort::quick<sort::order::asc>(
          mat.rptr[i], mat.rptr[i + 1], mat.idx.raw(), mat.val.raw());
    }
  }
  template <typename _> void apply_from_left(CSR<_, L> &mat) {
    auto t_val = vector<_, L>(mat.nnz());
    auto t_idx = vector<idx_t, L>(mat.nnz());
    auto t_rptr = vector<idx_t, L>(mat.nrows() + 1);
    idx_t nnz = 0;
    t_rptr[0] = nnz;
    for (idx_t i = 0; i < mat.nrows(); ++i) {
      for (idx_t j = mat.rptr[pi[i]]; j < mat.rptr[pi[i] + 1]; ++j) {
        t_val[nnz] = mat.val[j];
        t_idx[nnz++] = mat.idx[j];
      }
      t_rptr[i + 1] = nnz;
    }
    mat.val.copy(t_val);
    mat.idx.copy(t_idx);
    mat.rptr.copy(t_rptr);
  }
  template <typename _> void apply_from_left(vector<_, L> &vec) {
    auto t_vec = vector<_, L>(vec.size(0));
    for (idx_t i = 0; i < vec.size(0); ++i)
      t_vec[i] = vec[pi[i]];
    vec.copy(t_vec);
  }
};

template <class L> struct reorderer {
  spmat attr;
  permuter<L> p;
  permuter<L> pt;
  reorderer(spmat &attr, const permuter<L> &p, const permuter<L> &pt)
      : attr(attr), p(p), pt(pt) {}
  explicit reorderer(std::tuple<spmat, permuter<L>, permuter<L>> &&t)
      : attr(std::get<0>(t)), p(std::get<1>(t)), pt(std::get<2>(t)) {}
  template <typename T> reorderer apply(CSR<T, L> &mat) {
    p.apply_from_left(mat);
    pt.apply_from_right(mat);
    auto tmp = static_cast<spmat>(mat);
    mat.spmat::copy_attrs(attr);
    return reorderer(tmp, pt, p);
  }
  template <typename T, typename T2>
  reorderer apply(CSR<T, L> &mat, vector<T2, L> &vec) {
    p.apply_from_left(mat);
    pt.apply_from_right(mat);
    p.apply_from_left(vec);
    auto tmp = static_cast<spmat>(mat);
    mat.spmat::copy_attrs(attr);
    return reorderer(tmp, pt, p);
  }
  template <typename T> void apply(vector<T, L> vec) { p.apply_from_left(vec); }
};

namespace impl {

template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> color_greedy(
    const Graph<L> &G) {
  auto c_map = vector<idx_t, host>(G.nrows());
  idx_t c_num = 0;
  auto _c_size = std::vector<idx_t>(c_num + 2);
  _c_size[0] = 0;
  while (true) {
    idx_t num_colored_nodes = 0;
    for (idx_t i = 0; i < G.nrows(); i++) {
      if (c_map[i] != 0)
        continue;
      bool adjacent_to_colored_node = false;
      for (idx_t j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
        if (c_map[G.idx[j]] == c_num + 1) {
          adjacent_to_colored_node = true;
          break;
        }
      }
      if (adjacent_to_colored_node)
        continue;
      c_map[i] = c_num + 1;
      num_colored_nodes++;
    }
    if (!num_colored_nodes)
      break;
    c_num++;
    _c_size.resize(c_num + 2);
    _c_size[c_num] = _c_size[c_num - 1] + num_colored_nodes;
  }
  auto c_size = vector<idx_t, host>(c_num + 1).copy(_c_size.data());
  return {c_map, c_size, c_num};
}

template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> color_cyclic(
    const Graph<L> &G) {
  auto c_num = G.rptr[1];
  for (idx_t i = 1; i < G.nrows(); i++)
    c_num = (c_num < G.rptr[i + 1] - G.rptr[i]) ? G.rptr[i + 1] - G.rptr[i] : c_num;
  auto c_map = vector<idx_t, host>(G.nrows());
  auto c_size = vector<idx_t, host>(c_num + 1);
  auto adja = vector<idx_t, host>(c_num).fill(0);
  idx_t adja_len;
  idx_t color_id = 0;
  for (idx_t i = 0; i < G.nrows(); i++) {
    adja_len = 0;
    bool isAdjacent = false;
    for (idx_t j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
      if (G.idx[j] >= i)
        continue;
      adja[adja_len] = c_map[G.idx[j]];
      adja_len++;
      if (c_map[G.idx[j]] == color_id + 1)
        isAdjacent = true;
    }
    if (isAdjacent) {
      while (true) {
        color_id = (color_id + 1) % c_num;
        bool flag = false;
        for (idx_t j = 0; j < adja_len; j++) {
          if (adja[j] == color_id + 1) {
            flag = true;
            break;
          }
        }
        if (!flag)
          break;
      }
    }
    ++c_size[color_id + 1];
    c_map[i] = color_id + 1;
    color_id = (color_id + 1) % c_num;
  }
  for (idx_t i = 0; i < c_num; i++)
    c_size[i + 1] += c_size[i];
  return {c_map, c_size, c_num};
}

template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> color_hybrid(
    const Graph<L> &G) {
  auto c_map = vector<idx_t, host>(G.nrows());
  idx_t c_num = 0;
  while (true) {
    idx_t hasOccurred = 0;
    for (idx_t i = 0; i < G.nrows(); i++) {
      if (c_map[i] != 0)
        continue;
      bool isAdjacent = false;
      for (idx_t j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
        if (c_map[G.idx[j]] == c_num + 1) {
          isAdjacent = true;
          break;
        }
      }
      if (isAdjacent) {
        continue;
      }
      c_map[i] = c_num + 1;
      hasOccurred++;
    }
    if (!hasOccurred) {
      break;
    }
    c_num++;
  }
  auto c_num_max = c_num + 30;
  auto _c_size = std::vector<idx_t>(c_num_max + 1);
  while (true) {
    for (idx_t i = 0; i < G.nrows(); ++i)
      c_map[i] = 0;
    for (idx_t i = 0; i < c_num + 1; ++i)
      _c_size[i] = 0;
    idx_t color_id = 0;
    int64_t loop_count = 0;
    for (idx_t i = 0; i < G.nrows(); ++i) {
      loop_count = 0;
      while (true) {
        if (loop_count >= c_num) {
          loop_count = -1;
          break;
        }
        bool isAdjacent = false;
        for (idx_t j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
          if (G.idx[j] >= i)
            continue;
          if (c_map[G.idx[j]] == color_id + 1) {
            isAdjacent = true;
            break;
          }
        }
        if (!isAdjacent) {
          _c_size[color_id + 1]++;
          c_map[i] = color_id + 1;
          color_id = (color_id + 1) % c_num;
          break;
        }
        color_id = (color_id + 1) % c_num;
        loop_count++;
      }
      if (loop_count == -1)
        break;
    }
    if (loop_count == -1) {
      c_num += 1;
      if (c_num > c_num_max) {
        printf("Overflow in Coloring::Hybrid.");
        exit(1);
      }
      continue;
    }
    break;
  }
  auto c_size = vector<idx_t, host>(c_num + 1);
  c_size[0] = 0;
  for (idx_t i = 0; i < c_num; i++) {
    c_size[i + 1] = _c_size[i + 1] + c_size[i];
  }
  return {c_map, c_size, c_num};
}

template <class L>
vector<idx_t, L> block_simple(const Graph<L> &G, const idx_t b_size) {
  auto b_map = vector<idx_t, L>(G.nrows());
  for (idx_t i = 0; i < G.nrows(); i++)
    b_map[i] = i / b_size + 1;
  return b_map;
}

template <class L> vector<idx_t, L> block_connect(const Graph<L> &G, idx_t b_size) {
  auto b_map = vector<idx_t, L>(G.nrows());
  auto b_num = G.nrows() / b_size;
  auto seed_q = vector<idx_t, L>(b_size);
  idx_t seed = 0, prev_seed = 0;
  idx_t count = 0;
  bool isSame = false;
  for (idx_t i = 0; i < b_num; i++) {
    if (!isSame)
      count = 0;
    seed = prev_seed; // Selecting a new seed node.
    while (b_map[seed] != 0)
      seed++;
    prev_seed = seed;
    idx_t head = 0;
    idx_t tail = 0; // Initializing the queue.
    // Assigning the current block ID to the seed node.
    b_map[seed] = i + 1;
    count++;
    if (count == b_size) {
      isSame = false;
      continue;
    }
    // Assigning the current block ID to
    // the nodes adjacent to the seed node
    // until 'b_size' nodes are assigned.
    while (count < b_size) {
      for (idx_t j = G.rptr[seed]; j < G.rptr[seed + 1]; ++j) {
        idx_t id = G.idx[j];
        if (b_map[id] == 0) {
          b_map[id] = i + 1;
          count++;
          seed_q[tail] = id;
          tail++; // Enqueue
          // When 'b_size' nodes are assigned the current ID,
          // go to the next ID assignment.
          if (count == b_size) {
            isSame = false;
            break;
          }
        }
      }
      // If the queue is empty,
      // go back to the initial seed selection.
      if (head == tail) {
        isSame = true;
        i--;
        break;
      }
      seed = seed_q[head];
      head++; // Dequeue
    }
  }
  return b_map;
}

template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>> get_block_adjacency(
    const Graph<L> &G, const vector<idx_t, L> &b_map, const vector<idx_t, L> &b_list,
    const idx_t b_size) {
  idx_t b_num = G.nrows() / b_size;
  auto _b_idx = vector<idx_t, L>(G.nnz());
  auto b_rptr = vector<idx_t, L>(b_num + 1);

  std::vector<idx_t> t;
  b_rptr[0] = 0;
  for (idx_t bid = 0; bid < b_num; bid++) {
    t.clear();
    for (idx_t j = 0; j < b_size; j++) {
      idx_t id = b_list[bid * b_size + j];
      for (idx_t k = G.rptr[id]; k < G.rptr[id + 1]; k++)
        t.push_back(b_map[G.idx[k]] - 1);
    }
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    b_rptr[bid + 1] = b_rptr[bid] + t.size();
    for (size_t j = 0; j < t.size(); j++)
      _b_idx[b_rptr[bid] + j] = t[j];
  }
  auto b_idx = vector<idx_t, L>(b_rptr[b_num]).copy(_b_idx);
  return {b_idx, b_rptr};
}

template <class L>
std::tuple<vector<idx_t, host>, vector<idx_t, host>> get_pseudo_block_adjacency(
    const Graph<L> &G, const vector<idx_t, L> &b_map, const vector<idx_t, L> &b_list,
    const idx_t b_size) {
  idx_t b_num = G.nrows() / b_size;
  auto _b_idx = vector<idx_t, L>(G.nnz());
  auto b_rptr = vector<idx_t, L>(b_num + 1);

  std::vector<idx_t> t;
  b_rptr[0] = 0;
  for (idx_t bid = 0; bid < b_num; bid++) {
    t.clear();
    for (idx_t j = 0; j < b_size; j++) {
      // idx_t id = b_list[bid * b_size + j];
      // for (idx_t k = G.rptr[id]; k < G.rptr[id + 1]; k++)
      //   t.push_back(b_map[G.idx[k]] - 1);
      idx_t id = b_list[bid * b_size + j];
      for (idx_t k = G.rptr[id]; k < G.rptr[id + 1]; k++) {
        idx_t b = b_map[G.idx[k]] - 1;
        if (b_list[b * b_size + j] != G.idx[k])
          continue;
        t.push_back(b_map[G.idx[k]] - 1);
      }
    }
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    b_rptr[bid + 1] = b_rptr[bid] + t.size();
    for (size_t j = 0; j < t.size(); j++)
      _b_idx[b_rptr[bid] + j] = t[j];
  }
  auto b_idx = vector<idx_t, L>(b_rptr[b_num]).copy(_b_idx);
  return {b_idx, b_rptr};
}

#if 0

template <typename T>
void block_stack(const Graph<T> &G, idx_t b_size, idx_t **b_map) {
  *b_map = Calloc<On::CPU, idx_t>(G.size._nrows);
  idx_t b_num = G.size._nrows / b_size;
  std::set<idx_t> st;
  idx_t seed = 0, prev_seed = 0;
  idx_t now_size = 0;
  bool isSame = false;
  for (idx_t bid = 0; bid < b_num; bid++) {
    now_size = (isSame) ? now_size : 0;
    isSame = false;
    seed = prev_seed; // Selecting a new seed node.
    while ((*b_map)[seed] != 0)
      seed++;
    prev_seed = seed;
    st.clear();
    st.insert(seed);

    while (now_size < b_size) {
      if (st.empty()) {
        isSame = true;
        bid--;
        break;
      }
      auto id = *st.begin();
      st.erase(st.begin());
      // if ((*b_map)[id] != 0)
      //   continue;
      (*b_map)[id] = bid + 1;
      now_size++;
      for (idx_t j = G.rptr[id]; j < G.rptr[id + 1]; j++) {
        if ((*b_map)[G.cind[j]] == 0)
          st.insert(G.cind[j]);
      }
    }
  }
}

#endif

} // namespace impl

} // namespace kmm

#endif // UMINEKO_SPARSE_MATRIX_ORDERING_GRAPH_HPP
