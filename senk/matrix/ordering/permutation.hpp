#ifndef SENK_MATRIX_ORDERING_PERMUTATION_HPP
#define SENK_MATRIX_ORDERING_PERMUTATION_HPP

#include "senk/core/sort.hpp"
#include "senk/core/tensor.hpp"
#include "senk/core/tools.hpp"
#include "senk/matrix/csr.hpp"

namespace senk {

template <class L, typename I = index_t>
struct Permutation : public vector<I, host> {
  using loc_t = L;
  using idx_t = I;
  static_assert(std::is_same_v<L, host>);
  explicit Permutation(idx_t n) : vector<I, host>(n) { this->iota(0); }
  template <typename _>
  void apply_from_right(CSR<_, L> &mat) {
    for (idx_t i = 0; i < mat.nrows(); i++) {
      for (idx_t j = mat.rptr[i]; j < mat.rptr[i + 1]; ++j)
        mat.col[j] = this->operator[](mat.col[j]);
      sort::pack_sort<sort::order::asc>(
          mat.rptr[i], mat.rptr[i + 1], mat.col.raw(), mat.val.raw());
    }
  }
  template <typename _>
  void apply_from_left(CSR<_, L> &mat) {
    auto t_val = vector<_, L>(mat.val.shape(0));
    auto t_col = vector<idx_t, L>(mat.val.shape(0));
    auto t_rptr = vector<idx_t, L>(mat.nrows() + 1);
    idx_t nnz = 0;
    t_rptr[0] = nnz;
    for (idx_t i = 0; i < mat.nrows(); ++i) {
      for (idx_t j = mat.rptr[this->operator[](i)];
          j < mat.rptr[this->operator[](i) + 1]; ++j) {
        t_val[nnz] = mat.val[j];
        t_col[nnz++] = mat.col[j];
      }
      t_rptr[i + 1] = nnz;
    }
    mat.val.copy(t_val);
    mat.col.copy(t_col);
    mat.rptr.copy(t_rptr);
  }
  template <typename _>
  void apply_from_left(vector<_, L> &vec) {
    auto t_vec = vector<_, L>(vec.shape(0));
    for (decltype(vec.shape(0)) i = 0; i < vec.shape(0); ++i)
      t_vec[i] = vec[this->operator[](i)];
    vec.copy(t_vec);
  }
};

template <class L>
struct Reordering {
  attribute attr;
  Permutation<L> p;
  Permutation<L> pt;
  Reordering(attribute &attr, const Permutation<L> &p, const Permutation<L> &pt)
      : attr(attr), p(p), pt(pt) {}
  explicit Reordering(std::tuple<attribute, Permutation<L>, Permutation<L>> &&t)
      : attr(std::get<0>(t)), p(std::get<1>(t)), pt(std::get<2>(t)) {}
  template <typename T>
  Reordering apply(CSR<T, L> &mat) {
    p.apply_from_left(mat);
    pt.apply_from_right(mat);
    auto tmp = mat.attr;
    mat.attr = attr;
    return Reordering(tmp, pt, p);
  }
  template <typename T, typename T2>
  Reordering apply(CSR<T, L> &mat, vector<T2, L> &vec) {
    p.apply_from_left(mat);
    pt.apply_from_right(mat);
    p.apply_from_left(vec);
    auto tmp = mat.attr;
    mat.attr = attr;
    return Reordering(tmp, pt, p);
  }
  template <typename T>
  void apply(vector<T, L> &vec) {
    p.apply_from_left(vec);
  }
};

} // namespace senk

#endif