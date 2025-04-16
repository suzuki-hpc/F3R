#ifndef UMINEKO_SPARSE_MATRIX_CSR_HPP
#define UMINEKO_SPARSE_MATRIX_CSR_HPP

#include "umineko-core/tensor.hpp"
#include "umineko-sparse/matrix/base.hpp"
#include "umineko-sparse/matrix/io.hpp"

#include "umineko-sparse/models.hpp"

namespace kmm {

template <typename T, class L> struct CSR : spmat {
  using loc_t = L;
  using val_t = T;
  vector<T, L> val;
  vector<idx_t, L> idx;
  vector<idx_t, L> rptr;

  explicit CSR(const io::CSR<val_t> &in)
      : spmat(in), val(in.val), idx(in.idx), rptr(in.rptr) {}
  explicit CSR(const spmat &base)
      : spmat(base), val(nnz()), idx(nnz()), rptr(nrows() + 1) {}
  template <typename T2, class L2>
  explicit CSR(const CSR<T2, L2> &in)
      : spmat(in), val(in.val), idx(in.idx), rptr(in.rptr) {}
  CSR(const impl::shape &s, const idx_t nz)
      : spmat(s, nz), val(nnz()), idx(nnz()), rptr(nrows() + 1) {}

  CSR(const CSR &in) = default;
  CSR &operator=(const CSR &in) = default;

  template <typename in_t, typename out_t>
  void operate(const vector<in_t, L> &in, vector<out_t, L> out) const {
    exec<L>::para_for(out.size(0), [=, *this](idx_t i) mutable {
      auto t = decltype(std::declval<T &>() * std::declval<in_t &>()){0};
      for (auto j = rptr[i]; j < rptr[i + 1]; ++j)
        t += val[j] * in[idx[j]];
      out[i] = t;
    });
  }
  template <typename rhs_t, typename in_t, typename res_t>
  void compute_residual(const vector<rhs_t, L> &rhs, const vector<in_t, L> &in,
      vector<res_t, L> res) const {
    exec<L>::para_for(res.size(0), [=, *this](idx_t i) mutable {
      auto t = decltype(std::declval<T &>() * std::declval<in_t &>()){0};
      for (auto j = rptr[i]; j < rptr[i + 1]; ++j)
        t += val[j] * in[idx[j]];
      res[i] = rhs[i] - t;
    });
  }

  template <typename T2, class L2> CSR &copy(const CSR<T2, L2> &in) {
    spmat::copy(static_cast<spmat>(in));
    val.copy(in.val);
    idx.copy(in.idx);
    rptr.copy(in.rptr);
    return *this;
  }

  CSR &inverse_last() {
    exec<L>::para_for(nrows(), [=, *this](const idx_t i) mutable {
      val[rptr[i + 1] - 1] = 1. / val[rptr[i + 1] - 1];
    });
    return *this;
  }
  CSR &rotate180() {
    exec<L>::para_for(
        (nrows() + 1 + 1) / 2, [=, n = nrows(), *this](idx_t i) mutable {
          auto tmp = rptr[i];
          rptr[i] = nz - rptr[n - i];
          rptr[n - i] = nz - tmp;
        });
    exec<L>::para_for((nnz() + 1) / 2, [=, n = nrows(), *this](idx_t i) mutable {
      auto tmp = val[i];
      val[i] = val[nz - 1 - i];
      val[nz - 1 - i] = tmp;
      auto tmp2 = idx[i];
      idx[i] = n - 1 - idx[nz - 1 - i];
      idx[nz - 1 - i] = n - 1 - tmp2;
    });
    if (pattern == Pattern::block_diagonal) {
      auto new_set = pattern_set.clone().reverse();
      pattern_set = new_set;
    }
    return *this;
  }

  CSR duplicate() const { return CSR(static_cast<spmat>(*this)).copy(*this); }
  CSR duplicate_val() const {
    auto res = CSR(*this);
    auto vec = vector<T, L>(res.nnz()).copy(val);
    res.val = vec;
    return res;
  }
  CSR duplicate_rotate180() const {
    auto res = CSR(get_shape(), nnz());
    auto n = nrows();
    auto nz = nnz();
    exec<L>::para_for(nrows() + 1, [=, *this, p = res.rptr.raw()](idx_t i) mutable {
      p[i] = nz - rptr[n - i];
    });
    exec<L>::para_for(
        nnz(), [=, *this, p1 = res.val.raw(), p2 = res.idx.raw()](idx_t i) mutable {
          p1[i] = val[nz - 1 - i];
          p2[i] = n - 1 - idx[nz - 1 - i];
        });
    return res;
  }
  CSR duplicate_block_diagonal(const idx_t block_num, const idx_t unit_size) const {
    if (block_num * unit_size > nrows() || nrows() % unit_size != 0) {
      printf("Error: mismatch between block_num * unit_size and N\n");
      exit(EXIT_FAILURE);
    }
    /* Calculate the size of each block. */
    idx_t unit_num = nrows() / unit_size;
    idx_t block_size = (unit_num + block_num - 1) / block_num * unit_size;
    auto b_size = vector<idx_t, host>(block_num + 1);
    b_size[0] = 0;
    for (idx_t i = 0; i < block_num; ++i) {
      b_size[i + 1] = (i != block_num - 1) ? b_size[i] + block_size : nrows();
    }
    /* Count the number of elements in the blocks. */
    idx_t nnz = 0;
    for (idx_t bi = 0; bi < block_num; bi++) {
      auto left = b_size[bi];
      auto right = b_size[bi + 1];
      for (idx_t i = left; i < right; i++) {
        for (idx_t j = rptr[i]; j < rptr[i + 1]; ++j) {
          if (left <= idx[j] && idx[j] < right)
            nnz++;
        }
      }
    }
    /* Store nonzero elements into the blocks. */
    auto attr = spmat::duplicate().change_pattern(Pattern::block_diagonal);
    attr.nz = nnz;
    attr.pattern_set = {unit_size, block_num, b_size};
    auto res = CSR(attr);
    nnz = 0;
    res.rptr[0] = nnz;
    for (idx_t bi = 0; bi < block_num; bi++) {
      auto left = b_size[bi];
      auto right = b_size[bi + 1];
      for (idx_t i = left; i < right; i++) {
        for (idx_t j = rptr[i]; j < rptr[i + 1]; ++j) {
          if (left <= idx[j] && idx[j] < right) {
            res.val[nnz] = val[j];
            res.idx[nnz++] = idx[j];
          }
        }
        res.rptr[i + 1] = nnz;
      }
    }
    return res;
  }

  CSR duplicate_only_partitioned() const {
    if (pattern != Pattern::partitioned)
      printf("error: duplicate_only_partitioned\n");
    int cnt = 0;
    for (int ip = 0; ip < pattern_set.idx_size; ++ip) {
      auto s = pattern_set.idx_ptr[ip];
      auto e = pattern_set.idx_ptr[ip + 1];
      for (int i = s; i < e; ++i) {
        for (int j = rptr[i]; j < rptr[i + 1]; ++j) {
          if (idx[j] < s || e <= idx[j])
            continue;
          ++cnt;
        }
      }
    }
    auto res = CSR<double, host>(get_shape(), cnt);
    res.copy_attrs(*this);
    cnt = 0;
    res.rptr[0] = cnt;
    for (int ip = 0; ip < pattern_set.idx_size; ++ip) {
      auto s = pattern_set.idx_ptr[ip];
      auto e = pattern_set.idx_ptr[ip + 1];
      for (int i = s; i < e; ++i) {
        for (int j = rptr[i]; j < rptr[i + 1]; ++j) {
          if (idx[j] < s || e <= idx[j])
            continue;
          res.val[cnt] = val[j];
          res.idx[cnt] = idx[j];
          ++cnt;
        }
        res.rptr[i + 1] = cnt;
      }
    }
    return res;
  }
};

namespace impl {

#if defined(KMM_WITH_CUDA)
#define KMM_LOC __host__ __device__
#else
#define KMM_LOC
#endif
#define KMM_ENABULER(cond) std::enable_if_t<cond, std::nullptr_t> = nullptr
#define KMM_IS_SAME_V(cond1, cond2) std::is_same_v<cond1, cond2>

template <typename> struct is_csr {
  static constexpr bool value = false;
};
template <class T> inline constexpr bool is_csr_v = is_csr<T>::value;
template <typename T, class L> struct is_csr<CSR<T, L>> {
  static constexpr bool value = true;
};

template <typename L, typename R, KMM_ENABULER(is_csr_v<L>),
    KMM_ENABULER(is_vector_expr_v<R>)>
struct CSRMV {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<typename L::val_t &>() *
                         std::declval<typename R::val_t &>());
  L l;
  R r;
  CSRMV(const L &l, const R &r) : l(l), r(r) {}
  KMM_LOC auto eval(idx_t i = 0, [[maybe_unused]] idx_t j = 0) const {
    auto t = val_t{0};
    for (auto k = l.rptr[i]; k < l.rptr[i + 1]; ++k)
      t += l.val[k] * r.eval(l.idx[k]);
    return t;
  }
  [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return r.size(i); }
};
template <typename T1, typename T2> struct is_vector_expr<CSRMV<T1, T2>> {
  static constexpr bool value = true;
};

#undef KMM_ENABULER
#undef KMM_IS_SAME_V
#undef KMM_LOC

} // namespace impl

} // namespace kmm

template <typename L, typename R> kmm::impl::CSRMV<L, R> operator*(L v1, R v2) {
  return kmm::impl::CSRMV<L, R>(v1, v2);
}

#endif // UMINEKO_SPARSE_MATRIX_CSR_HPP
