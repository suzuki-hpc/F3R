#ifndef SENK_MATRIX_ORDERING_GRAPH_HPP
#define SENK_MATRIX_ORDERING_GRAPH_HPP

#include <algorithm>
#include <set>

#include "senk/matrix/csr.hpp"

namespace senk {

enum class coloring { greedy, greedy2, cyclic, cyclic2, gradient, hybrid };
enum class blocking { simple, connect, stack };

template <typename T, class L, typename I, typename S>
struct Graph;

namespace impl {

template <typename T, class L, typename I, typename S>
std::tuple<vector<I, L>, vector<I, L>, I> color_greedy(
    const Graph<T, L, I, S> &G);
template <typename T, class L, typename I, typename S>
std::tuple<vector<I, L>, vector<I, L>, I> color_cyclic(
    const Graph<T, L, I, S> &G);

template <typename T, class L, typename I, typename S>
vector<I, L> block_simple(const Graph<T, L, I, S> &G, I b_size);
template <typename T, class L, typename I, typename S>
vector<I, L> block_connect(const Graph<T, L, I, S> &G, I b_size);
template <typename T, class L, typename I, typename S>
vector<I, L> block_stack(const Graph<T, L, I, S> &G, I b_size);

template <typename T, class L, typename I, typename S>
Graph<T, L, I, S> get_block_adjacency(const Graph<T, L, I, S> &G,
    const vector<I, L> &b_map, const vector<I, L> &b_list, I b_size);
template <typename T, class L, typename I, typename S>
Graph<T, L, I, S> get_pseudo_block_adjacency(const Graph<T, L, I, S> &G,
    const vector<I, L> &b_map, const vector<I, L> &b_list, I b_size);

} // namespace impl

template <typename T, class L, typename I = index_t, typename S = serial_t>
struct Graph : CSR<T, L, I, S> {
  static_assert(std::is_same_v<L, host>);
  using Base = CSR<T, L, I, S>;
  using Base::attr;
  using Base::col;
  using Base::rptr;
  using Base::shape;
  using Base::val;
  using typename Base::idx_t;
  using typename Base::loc_t;
  using typename Base::srl_t;
  using typename Base::val_t;

  vector<I, L> list;

  Graph(const CSR<T, L, I, S> &in) : CSR<T, L, I, S>(in), list(1) {}
  Graph(std::array<idx_t, 2> shape, srl_t nnz, attribute attr = attribute{})
      : CSR<T, L, I, S>(shape, nnz, attr), list(1) {}
  Graph(const CSR<T, L, I, S> &in, const vector<I, L> &list)
      : CSR<T, L, I, S>(in), list(list) {}

  Graph transpose() {
    if (attr.is_symmetric())
      return *this;
    auto nnz = val.shape(0);
    auto t_rptr = vector<srl_t, L>(rptr.shape(0));
    auto t_col = vector<idx_t, L>(col.shape(0));
    auto t_val = vector<val_t, L>(val.shape(0));
    auto t_num = vector<idx_t, L>(this->ncols()).fill(0);
    for (idx_t i = 0; i < nnz; ++i) {
      ++t_num[col[i]];
    }
    t_rptr[0] = 0;
    for (idx_t i = 0; i < this->ncols(); ++i) {
      t_rptr[i + 1] = t_rptr[i] + t_num[i];
      t_num[i] = 0;
    }
    for (idx_t i = 0; i < this->nrows(); ++i) {
      for (auto j = rptr[i]; j < rptr[i + 1]; ++j) {
        auto off = t_rptr[col[j]];
        auto pos = t_num[col[j]];
        t_col[off + pos] = i;
        t_val[off + pos] = val[j];
        ++t_num[col[j]];
      }
    }
    val.copy(t_val);
    col.copy(t_col);
    rptr.copy(t_rptr);
    return *this;
  }

  Graph remove_direction() {
    if (attr.is_symmetric())
      return *this;
    auto tmp = Graph(this->duplicate()).transpose();
    srl_t nnz = 0;
    for (idx_t i = 0; i < this->ncols(); i++) {
      auto ptr = rptr[i];
      auto tmp_ptr = tmp.rptr[i];
      while (ptr < rptr[i + 1] || tmp_ptr < tmp.rptr[i + 1]) {
        auto colm = (ptr < rptr[i + 1]) ? col[ptr] : this->ncols();
        auto tmp_colm =
            (tmp_ptr < tmp.rptr[i + 1]) ? tmp.col[tmp_ptr] : this->ncols();
        ptr = (colm <= tmp_colm) ? ptr + 1 : ptr;
        tmp_ptr = (colm >= tmp_colm) ? tmp_ptr + 1 : tmp_ptr;
        nnz++;
      }
    }
    auto res = Graph(shape, nnz, attr);
    // res.spmat::copy_attrs(static_cast<spmat>(*this));
    res.attr.set_flag(impl::flags::is_symmetric);
    nnz = 0;
    res.rptr[0] = nnz;
    for (idx_t i = 0; i < this->ncols(); i++) {
      auto ptr = rptr[i];
      auto tmp_ptr = tmp.rptr[i];
      while (ptr < rptr[i + 1] || tmp_ptr < tmp.rptr[i + 1]) {
        auto colm = (ptr < rptr[i + 1]) ? col[ptr] : this->ncols();
        auto tmp_colm =
            (tmp_ptr < tmp.rptr[i + 1]) ? tmp.col[tmp_ptr] : this->ncols();
        if (colm < tmp_colm) {
          res.val[nnz] = val[ptr];
          res.col[nnz++] = colm;
        } else if (colm == tmp_colm) {
          res.val[nnz] = (val[ptr] + tmp.val[tmp_ptr]) / 2;
          res.col[nnz++] = colm;
        } else {
          res.val[nnz] = tmp.val[tmp_ptr];
          res.col[nnz++] = tmp_colm;
        }
        ptr = (colm <= tmp_colm) ? ptr + 1 : ptr;
        tmp_ptr = (colm >= tmp_colm) ? tmp_ptr + 1 : tmp_ptr;
      }
      res.rptr[i + 1] = nnz;
    }
    return res;
  }

  std::tuple<vector<idx_t, host>, vector<idx_t, host>, idx_t> make_coloring_set(
      [[maybe_unused]] const coloring type,
      [[maybe_unused]] const idx_t max = std::numeric_limits<idx_t>::max()) {
    if (type == coloring::greedy)
      return impl::color_greedy(*this);
    // if (type == coloring::greedy2)
    //   return impl::color_greedy_with_max(*this, max);
    if (type == coloring::cyclic)
      return impl::color_cyclic(*this);
    // if (type == coloring::cyclic2)
    //   return impl::color_cyclic_with_max(*this, max);
    // // if (type == Coloring::Hybrid) {
    // return impl::color_hybrid(*this);
    return impl::color_greedy(*this);
  }

  Graph make_blocked_graph(idx_t b_size, const blocking type) {
    if (this->nrows() % b_size != 0)
      throw std::runtime_error(
          "graph size must be a multiple of blocking size");
    auto nrows = this->nrows();
    auto b_map = [&]() {
      if (type == blocking::simple)
        return impl::block_simple(*this, b_size);
      if (type == blocking::connect)
        return impl::block_connect(*this, b_size);
      // return impl::block_connect(*this, b_size);
      return impl::block_simple(*this, b_size);
    }();
    auto b_list = vector<idx_t, L>(nrows);
    auto offset = vector<idx_t, L>(nrows / b_size);
    for (idx_t i = 0; i < nrows; i++) {
      auto bid = b_map[i] - 1;
      b_list[bid * b_size + offset[bid]] = i;
      ++offset[bid];
    }
    auto BG = impl::get_block_adjacency(*this, b_map, b_list, b_size);
    return Graph{BG, b_list};
  }

  Graph make_pseudo_blocked_graph(idx_t b_size, const blocking type) {
    if (this->nrows() % b_size != 0)
      throw std::runtime_error(
          "graph size must be a multiple of blocking size");
    auto nrows = this->nrows();
    auto b_map = [&]() {
      if (type == blocking::simple)
        return impl::block_simple(*this, b_size);
      if (type == blocking::connect)
        return impl::block_connect(*this, b_size);
      if (type == blocking::stack)
        return impl::block_stack(*this, b_size);
      // return impl::block_connect(*this, b_size);
      return impl::block_simple(*this, b_size);
    }();
    auto b_list = vector<idx_t, L>(nrows);
    auto offset = vector<idx_t, L>(nrows / b_size);
    for (idx_t i = 0; i < nrows; i++) {
      auto bid = b_map[i] - 1;
      b_list[bid * b_size + offset[bid]] = i;
      ++offset[bid];
    }
    auto BG = impl::get_pseudo_block_adjacency(*this, b_map, b_list, b_size);
    return Graph{BG, b_list};
  }
};

namespace impl {

template <typename T, class L, typename I, typename S>
std::tuple<vector<I, L>, vector<I, L>, I> color_greedy(
    const Graph<T, L, I, S> &G) {
  static_assert(std::is_same_v<L, host>, "");
  using idx_t = I;
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
      for (auto j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
        if (c_map[G.col[j]] == c_num + 1) {
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

template <typename T, class L, typename I, typename S>
std::tuple<vector<I, L>, vector<I, L>, I> color_cyclic(
    const Graph<T, L, I, S> &G) {
  static_assert(std::is_same_v<L, host>, "");
  using idx_t = I;
  auto c_num = G.rptr[1];
  for (idx_t i = 1; i < G.nrows(); i++)
    c_num = std::max(c_num, G.rptr[i + 1] - G.rptr[i]);
  auto c_map = vector<idx_t, host>(G.nrows());
  auto c_size = vector<idx_t, host>(c_num + 1);
  auto adja = vector<idx_t, host>(c_num).fill(0);
  idx_t adja_len;
  idx_t color_id = 0;
  for (idx_t i = 0; i < G.nrows(); i++) {
    adja_len = 0;
    bool isAdjacent = false;
    for (auto j = G.rptr[i]; j < G.rptr[i + 1]; ++j) {
      if (G.col[j] >= i)
        continue;
      adja[adja_len] = c_map[G.col[j]];
      adja_len++;
      if (c_map[G.col[j]] == color_id + 1)
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

template <typename T, class L, typename I, typename S>
vector<I, L> block_simple(const Graph<T, L, I, S> &G, I b_size) {
  using idx_t = I;
  auto b_map = vector<idx_t, L>(G.nrows());
  for (idx_t i = 0; i < G.nrows(); i++)
    b_map[i] = i / b_size + 1;
  return b_map;
}

template <typename T, class L, typename I, typename S>
vector<I, L> block_connect(const Graph<T, L, I, S> &G, I b_size) {
  using idx_t = I;
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
      for (auto j = G.rptr[seed]; j < G.rptr[seed + 1]; ++j) {
        idx_t id = G.col[j];
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

template <typename T, class L, typename I, typename S>
vector<I, L> block_stack(const Graph<T, L, I, S> &G, I b_size) {
  using idx_t = I;
  auto b_map = vector<idx_t, L>(G.nrows());
  auto b_num = G.nrows() / b_size;
  std::set<idx_t> st;
  idx_t seed = 0, prev_seed = 0;
  idx_t now_size = 0;
  bool isSame = false;
  for (idx_t bid = 0; bid < b_num; bid++) {
    now_size = (isSame) ? now_size : 0;
    isSame = false;
    seed = prev_seed; // Selecting a new seed node.
    while (b_map[seed] != 0)
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
      b_map[id] = bid + 1;
      now_size++;
      for (idx_t j = G.rptr[id]; j < G.rptr[id + 1]; j++) {
        if (b_map[G.col[j]] == 0)
          st.insert(G.col[j]);
      }
    }
  }
  return b_map;
}

template <typename T, class L, typename I, typename S>
Graph<T, L, I, S> get_block_adjacency(const Graph<T, L, I, S> &G,
    const vector<I, L> &b_map, const vector<I, L> &b_list, I b_size) {
  using idx_t = I;
  idx_t b_num = G.nrows() / b_size;
  auto _b_idx = vector<idx_t, L>(G.col.shape(0));
  auto b_rptr = vector<idx_t, L>(b_num + 1);

  std::vector<idx_t> t;
  b_rptr[0] = 0;
  for (idx_t bid = 0; bid < b_num; bid++) {
    t.clear();
    for (idx_t j = 0; j < b_size; j++) {
      idx_t id = b_list[bid * b_size + j];
      for (auto k = G.rptr[id]; k < G.rptr[id + 1]; k++)
        t.push_back(b_map[G.col[k]] - 1);
    }
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    b_rptr[bid + 1] = b_rptr[bid] + t.size();
    for (size_t j = 0; j < t.size(); j++)
      _b_idx[b_rptr[bid] + j] = t[j];
  }

  auto res = Graph<T, L, I>({b_num, b_num}, b_rptr[b_num]);
  res.val.fill(1.);
  res.col.copy(_b_idx.raw());
  res.rptr.copy(b_rptr.raw());

  return res;
}

template <typename T, class L, typename I, typename S>
Graph<T, L, I, S> get_pseudo_block_adjacency(const Graph<T, L, I, S> &G,
    const vector<I, L> &b_map, const vector<I, L> &b_list, I b_size) {
  using idx_t = I;
  idx_t b_num = G.nrows() / b_size;
  auto _b_idx = vector<idx_t, L>(G.col.shape(0));
  auto b_rptr = vector<idx_t, L>(b_num + 1);

  std::vector<idx_t> t;
  b_rptr[0] = 0;
  for (idx_t bid = 0; bid < b_num; bid++) {
    t.clear();
    for (idx_t j = 0; j < b_size; j++) {
      idx_t id = b_list[bid * b_size + j];
      for (auto k = G.rptr[id]; k < G.rptr[id + 1]; k++) {
        idx_t b = b_map[G.col[k]] - 1;
        if (b_list[b * b_size + j] != G.col[k])
          continue;
        t.push_back(b_map[G.col[k]] - 1);
      }
    }
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    b_rptr[bid + 1] = b_rptr[bid] + t.size();
    for (size_t j = 0; j < t.size(); j++)
      _b_idx[b_rptr[bid] + j] = t[j];
  }

  auto res = Graph<T, L, I, S>({b_num, b_num}, b_rptr[b_num]);
  res.val.fill(1.);
  res.col.copy(_b_idx.raw());
  res.rptr.copy(b_rptr.raw());

  return res;
}

} // namespace impl

} // namespace senk

#endif