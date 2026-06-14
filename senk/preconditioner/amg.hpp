#ifndef SENK_PRECONDITIONER_AMG_HPP
#define SENK_PRECONDITIONER_AMG_HPP

#include "senk/core/tensor.hpp"
#include "senk/matrix/csr.hpp"

#if __has_include("amgcl/amg.hpp") && __has_include(                           \
                                          "boost/property_tree/ptree.hpp")
#include "amgcl/amg.hpp"
#include "amgcl/backend/builtin.hpp"
#include "amgcl/coarsening/aggregation.hpp"
#include "amgcl/coarsening/ruge_stuben.hpp"
#include "amgcl/coarsening/smoothed_aggregation.hpp"
#endif

namespace senk {

namespace impl {

using mat_t = CSR<double, host>;
using vec_t = vector<index_t, host>;

inline std::tuple<mat_t, mat_t> get_strong_weak_set(const mat_t &, double);
inline mat_t get_influence_set(const mat_t &);
inline vec_t first_pass(const mat_t &, const mat_t &);
inline int second_pass(const mat_t &, const mat_t &, vec_t &);
inline std::tuple<mat_t, mat_t> separate_strong_to_coarse_fine(
    const mat_t &, const vec_t &);
inline mat_t get_interpolation_matrix(const mat_t &, const mat_t &,
    const mat_t &, const mat_t &, const vec_t &, index_t);
inline mat_t transpose_interpolation_matrix(const mat_t &);
inline mat_t get_coarse_matrix(const mat_t &, const mat_t &, const mat_t &);

#if __has_include("amgcl/amg.hpp") && __has_include(                           \
                                          "boost/property_tree/ptree.hpp")
template <typename ValueType>
using Backend = amgcl::backend::builtin<ValueType, index_t, index_t>;

template <typename T>
using SharedAMGCLCSR =
    std::shared_ptr<amgcl::backend::crs<T, index_t, index_t>>;

template <typename T>
SharedAMGCLCSR<T> toAMGCL(const CSR<T, host> &A) {
  SharedAMGCLCSR<T> res(new amgcl::backend::crs<T, index_t, index_t>);
  res->nrows = A.nrows();
  res->ncols = A.ncols();
  res->val = A.val.raw();
  res->col = A.col.raw();
  res->ptr = A.rptr.raw();
  return res;
}

template <typename T>
CSR<T, host> fromAMGCL(SharedAMGCLCSR<T> &A) {
  auto res = CSR<T, host>(
      mat_t({(index_t)A->nrows, (index_t)A->ncols}, A->ptr[A->nrows]));
  res.val.copy(A->val);
  res.col.copy(A->col);
  res.rptr.copy(A->ptr);
  return res;
}
#endif

} // namespace impl

// template <typename T, class L> struct AMG {};

inline std::tuple<CSR<double, host>, CSR<double, host>, CSR<double, host>>
rs_abs(const CSR<double, host> &A) {
  auto [Stg, Week] = impl::get_strong_weak_set(A, 0.6);
  auto Stg_t = impl::get_influence_set(Stg);
  auto is_coarse = impl::first_pass(Stg, Stg_t);
  auto num_c = impl::second_pass(Stg, Stg_t, is_coarse);
  auto [C_Stg, DS_Stg] = impl::separate_strong_to_coarse_fine(Stg, is_coarse);
  auto I =
      impl::get_interpolation_matrix(A, C_Stg, DS_Stg, Week, is_coarse, num_c);
  auto It = impl::transpose_interpolation_matrix(I);
  auto C = impl::get_coarse_matrix(A, I, It);
  // printf("%d %d %d\n", C.nrows(), C.ncols(), C.nnz());
  return {C, I, It};
}

#if __has_include("amgcl/amg.hpp") && __has_include(                           \
                                          "boost/property_tree/ptree.hpp")
inline std::tuple<CSR<double, host>, CSR<double, host>, CSR<double, host>> sa(
    const CSR<double, host> &A) {
  auto coarsener =
      new amgcl::coarsening::smoothed_aggregation<impl::Backend<double>>();
  auto amgcl_A = impl::toAMGCL(A);
  impl::SharedAMGCLCSR<double> t_I, t_It, t_C;
  std::tie(t_I, t_It) = coarsener->transfer_operators(*amgcl_A);
  sort_rows(*t_I);
  sort_rows(*t_It);
  t_C = coarsener->coarse_operator(*amgcl_A, *t_I, *t_It);
  sort_rows(*t_C);
  auto C = impl::fromAMGCL(t_C);
  auto I = impl::fromAMGCL(t_I);
  auto It = impl::fromAMGCL(t_It);
  amgcl_A.get()->val = nullptr;
  amgcl_A.get()->col = nullptr;
  amgcl_A.get()->ptr = nullptr;
  // printf("%d %d %d\n", C.nrows(), C.ncols(), C.nnz());
  return {C, I, It};
}
#endif

namespace impl {

inline std::tuple<mat_t, mat_t> get_strong_weak_set(
    const mat_t &A, double theta) {
  index_t N = A.nrows();
  index_t stgNNZ = 0;
  auto max = vector<double, host>(N).fill(0.0);

  for (index_t i = 0; i < N; i++) {
    for (index_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
      if (A.col[j] == i)
        continue;
      if (std::abs(A.val[j]) > max[i])
        max[i] = std::abs(A.val[j]);
    }
    for (index_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
      if (A.col[j] == i)
        continue;
      if (std::abs(A.val[j]) > theta * max[i])
        stgNNZ++;
    }
  }
  index_t weakNNZ = A.rptr[N] - stgNNZ;
  auto Stg = mat_t({N, A.ncols()}, stgNNZ);
  auto Weak = mat_t({N, A.ncols()}, weakNNZ);
  stgNNZ = 0;
  weakNNZ = 0;
  Stg.rptr[0] = stgNNZ;
  Weak.rptr[0] = weakNNZ;
  for (int i = 0; i < N; i++) {
    for (int j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
      if (A.col[j] != i && std::abs(A.val[j]) > theta * max[i]) {
        Stg.val[stgNNZ] = A.val[j];
        Stg.col[stgNNZ] = A.col[j];
        stgNNZ++;
      } else {
        Weak.val[weakNNZ] = A.val[j];
        Weak.col[weakNNZ] = A.col[j];
        weakNNZ++;
      }
    }
    Stg.rptr[i + 1] = stgNNZ;
    Weak.rptr[i + 1] = weakNNZ;
  }
  return {Stg, Weak};
}

inline mat_t get_influence_set(const mat_t &Stg) {
  int N = Stg.nrows();
  int M = Stg.ncols();
  auto num = vector<index_t, host>(M).fill(0);
  for (int i = 0; i < N; i++) {
    for (int j = Stg.rptr[i]; j < Stg.rptr[i + 1]; j++)
      num[Stg.col[j]]++;
  }
  auto Stg_t = mat_t({M, N}, Stg.rptr[Stg.nrows()]);
  Stg_t.rptr[0] = 0;
  for (int i = 0; i < M; i++) {
    Stg_t.rptr[i + 1] = Stg_t.rptr[i] + num[i];
    num[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    for (int j = Stg.rptr[i]; j < Stg.rptr[i + 1]; j++) {
      int idx = Stg.col[j];
      Stg_t.col[Stg_t.rptr[idx] + num[idx]] = i;
      num[idx]++;
    }
  }
  return Stg_t;
}

inline vec_t first_pass(const mat_t &Stg, const mat_t &Stg_t) {
  int N = Stg.nrows();
  auto is_coarse = vec_t(N).fill(0);
  auto is_visited = vec_t(N).fill(0);
  auto st_crdnl = vec_t(N).fill(0);
  auto s_crdnl = vec_t(N).fill(0);
  auto influence = vec_t(N).fill(0);
  auto index = vec_t(N).fill(0);
  auto where = vec_t(N).fill(0);

  for (int i = 0; i < N; i++) {
    st_crdnl[i] = Stg_t.rptr[i + 1] - Stg_t.rptr[i];
    s_crdnl[i] = Stg.rptr[i + 1] - Stg.rptr[i];
    index[i] = i;
    if (st_crdnl[i] == 0 && s_crdnl[i] == 0) {
      is_coarse[i] = 0;
      is_visited[i] = 1;
    }
    for (int j = Stg_t.rptr[i]; j < Stg_t.rptr[i + 1]; j++) {
      int idx = Stg_t.col[j];
      influence[i] += Stg.rptr[idx + 1] - Stg.rptr[idx];
    }
  }
  sort::pack_sort<sort::order::desc>(
      0, N, st_crdnl.raw(), s_crdnl.raw(), influence.raw(), index.raw());

  int max_crdnl = st_crdnl[0];
  std::vector<int> crdnl_ptr(max_crdnl + 2);
  crdnl_ptr[max_crdnl + 1] = 0;
  int strt = 0;
  for (int i = max_crdnl; i >= 0; i--) {
    int num = 0;
    bool flag = false;
    for (int j = strt; j < N; j++) {
      if (st_crdnl[j] == i) {
        num++;
      } else {
        crdnl_ptr[i] = crdnl_ptr[i + 1] + num;
        strt = j;
        flag = true;
        break;
      }
    }
    if (flag)
      continue;
    crdnl_ptr[i] = crdnl_ptr[i + 1] + num;
  }
  for (int i = max_crdnl; i >= 0; i--) {
    if (crdnl_ptr[i] - crdnl_ptr[i + 1] > 1)
      sort::pack_sort<sort::order::asc>(
          crdnl_ptr[i + 1], crdnl_ptr[i], influence.raw(), index.raw());
  }

  for (int i = 0; i < N; i++)
    where[index[i]] = i;
  int pos = 0;
  while (pos < N) {
    while (pos < N && is_visited[index[pos]])
      pos++;
    if (pos >= N)
      break;
    int c_idx = index[pos];
    is_visited[c_idx] = 1;
    is_coarse[c_idx] = 1; // 1 = Coarse
    for (int i = Stg_t.rptr[c_idx]; i < Stg_t.rptr[c_idx + 1]; i++) {
      int f_idx = Stg_t.col[i];
      if (is_visited[f_idx])
        continue;
      is_visited[f_idx] = 1;
      is_coarse[f_idx] = 0; // 0 = Fine;

      for (int j = Stg.rptr[f_idx]; j < Stg.rptr[f_idx + 1]; j++) {
        if (is_visited[Stg.col[j]])
          continue;
        int idx = where[Stg.col[j]];
        if (idx <= pos)
          printf("Warning 1: %d %d\n", idx, pos);
        st_crdnl[idx]++;
        int obj;
        if (st_crdnl[idx] > max_crdnl) {
          max_crdnl = st_crdnl[idx];
          if (crdnl_ptr[st_crdnl[idx]] > pos + 1)
            printf("Warning 2: %d %d\n", crdnl_ptr[st_crdnl[idx]], pos + 1);
          obj = pos + 1;
          crdnl_ptr.push_back(pos + 1);
          crdnl_ptr[max_crdnl] = pos + 2;
        } else {
          if (crdnl_ptr[st_crdnl[idx]] > pos + 1) {
            obj = crdnl_ptr[st_crdnl[idx]];
            crdnl_ptr[st_crdnl[idx]] += 1;
          } else {
            obj = pos + 1;
            crdnl_ptr[st_crdnl[idx]] = pos + 2;
          }
        }
        std::swap<int>(st_crdnl[idx], st_crdnl[obj]);
        std::swap<int>(index[idx], index[obj]);
        std::swap<int>(where[index[idx]], where[index[obj]]);
      }
    }
    for (int i = Stg.rptr[c_idx]; i < Stg.rptr[c_idx + 1]; i++) {
      if (is_visited[Stg.col[i]])
        continue;
      int idx = where[Stg.col[i]];
      if (st_crdnl[idx] == 0)
        continue;
      st_crdnl[idx]--;
      int obj;
      obj = crdnl_ptr[st_crdnl[idx] + 1] - 1;
      crdnl_ptr[st_crdnl[idx] + 1] -= 1;
      std::swap<int>(st_crdnl[idx], st_crdnl[obj]);
      std::swap<int>(index[idx], index[obj]);
      std::swap<int>(where[index[idx]], where[index[obj]]);
    }
    pos++;
  }
  return is_coarse;
}

inline int second_pass(
    const mat_t &Stg, [[maybe_unused]] const mat_t &Stg_t, vec_t &is_coarse) {
  int N = Stg.nrows();
  int num_coarse = 0;
  for (int i = 0; i < N; i++) {
    if (is_coarse[i])
      continue;
    // i is a fine point.
    for (int j = Stg.rptr[i]; j < Stg.rptr[i + 1]; j++) {
      if (is_coarse[Stg.col[j]])
        continue;
      int f_idx = Stg.col[j];
      int flag = 0;
      int main_ptr = Stg.rptr[i];
      int sub_ptr = Stg.rptr[f_idx];
      while (main_ptr < Stg.rptr[i + 1] && sub_ptr < Stg.rptr[f_idx + 1]) {
        if (Stg.col[sub_ptr] < Stg.col[main_ptr]) {
          sub_ptr++;
        } else if (Stg.col[sub_ptr] == Stg.col[main_ptr]) {
          if (is_coarse[Stg.col[main_ptr]]) {
            flag = 1;
            break;
          }
          sub_ptr++;
          main_ptr++;
        } else {
          main_ptr++;
        }
      }
      if (flag)
        continue;
      is_coarse[i] = 1; // Change into a C-point
      break;
    }
  }
  for (int i = 0; i < N; i++) {
    if (is_coarse[i])
      num_coarse++;
  }
  // printf("%d\n", num_coarse);
  return num_coarse;
}

inline std::tuple<mat_t, mat_t> separate_strong_to_coarse_fine(
    const mat_t &Stg, const vec_t &is_coarse) {
  int N = Stg.nrows();
  int cNNZ = 0, dsNNZ = 0;
  for (int i = 0; i < N; i++) {
    for (int j = Stg.rptr[i]; j < Stg.rptr[i + 1]; j++) {
      if (is_coarse[Stg.col[j]])
        cNNZ++;
      else
        dsNNZ++;
    }
  }
  auto C_Stg = mat_t({Stg.nrows(), Stg.ncols()}, cNNZ);
  auto DS_Stg = mat_t({Stg.nrows(), Stg.ncols()}, dsNNZ);
  cNNZ = 0;
  dsNNZ = 0;
  C_Stg.rptr[0] = cNNZ;
  DS_Stg.rptr[0] = dsNNZ;
  for (int i = 0; i < N; i++) {
    for (int j = Stg.rptr[i]; j < Stg.rptr[i + 1]; j++) {
      if (is_coarse[Stg.col[j]]) {
        C_Stg.val[cNNZ] = Stg.val[j];
        C_Stg.col[cNNZ] = Stg.col[j];
        cNNZ++;
      } else {
        DS_Stg.val[dsNNZ] = Stg.val[j];
        DS_Stg.col[dsNNZ] = Stg.col[j];
        dsNNZ++;
      }
    }
    C_Stg.rptr[i + 1] = cNNZ;
    DS_Stg.rptr[i + 1] = dsNNZ;
  }
  return {C_Stg, DS_Stg};
}

inline mat_t get_interpolation_matrix(const mat_t &A, const mat_t &C_Stg,
    const mat_t &DS_Stg, const mat_t &Week, const vec_t &_is_coarse,
    index_t num_c) {
  int N = A.nrows();
  auto col = vec_t(N);
  auto is_coarse = _is_coarse;
  int count = 0;
  for (int i = 0; i < N; i++) {
    if (is_coarse[i]) {
      col[i] = count;
      count++;
    }
  }
  int nnz = 0;
  for (int i = 0; i < N; i++) {
    if (is_coarse[i]) {
      nnz++;
    } else {
      for (int j = C_Stg.rptr[i]; j < C_Stg.rptr[i + 1]; j++)
        nnz++;
    }
  }
  auto I = mat_t({N, num_c}, nnz);
  nnz = 0;
  I.rptr[0] = nnz;
  for (int i = 0; i < N; i++) {
    if (is_coarse[i]) {
      I.val[nnz] = 1;
      I.col[nnz] = col[i];
      nnz++;
    } else {
      double deno = 0;
      for (int j = Week.rptr[i]; j < Week.rptr[i + 1]; j++)
        deno += Week.val[j];
      for (int j = C_Stg.rptr[i]; j < C_Stg.rptr[i + 1]; j++) {
        double aij = C_Stg.val[j];
        for (int m = DS_Stg.rptr[i]; m < DS_Stg.rptr[i + 1]; m++) {
          double amj = 0;
          for (int l = A.rptr[DS_Stg.col[m]]; l < A.rptr[DS_Stg.col[m] + 1];
              l++) {
            if (A.col[l] == C_Stg.col[j]) {
              amj = A.val[l];
              break;
            }
          }
          bool flag = false;
          double amk = 0;
          int m_ptr = A.rptr[DS_Stg.col[m]];
          int k_ptr = C_Stg.rptr[i];
          while (
              m_ptr < A.rptr[DS_Stg.col[m] + 1] && k_ptr < C_Stg.rptr[i + 1]) {
            if (A.col[m_ptr] < C_Stg.col[k_ptr]) {
              m_ptr++;
            } else if (A.col[m_ptr] == C_Stg.col[k_ptr]) {
              amk += A.val[m_ptr];
              flag = true;
              m_ptr++;
              k_ptr++;
            } else {
              k_ptr++;
            }
          }
          if (amk == 0) {
            if (flag) {
              printf("1 amk = 0\n"); // exit(1);
              amk = 0.01;
            } else {
              printf("2 amk = 0\n");
              exit(1);
            }
          }
          aij += DS_Stg.val[m] * amj / amk;
        }
        I.val[nnz] = -aij / deno;
        I.col[nnz] = col[C_Stg.col[j]];
        nnz++;
      }
    }
    I.rptr[i + 1] = nnz;
  }
  return I;
}

inline mat_t transpose_interpolation_matrix(const mat_t &I) {
  auto N = I.nrows();
  auto M = I.ncols();
  auto NNZ = I.rptr[I.nrows()];
  auto num = vec_t(M).fill(0);
  for (index_t i = 0; i < N; i++) {
    for (index_t j = I.rptr[i]; j < I.rptr[i + 1]; j++)
      num[I.col[j]]++;
  }
  auto It = mat_t({M, N}, NNZ);
  It.rptr[0] = 0;
  for (index_t i = 0; i < M; i++) {
    It.rptr[i + 1] = It.rptr[i] + num[i];
    num[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    for (int j = I.rptr[i]; j < I.rptr[i + 1]; j++) {
      int st = It.rptr[I.col[j]];
      int off = num[I.col[j]];
      It.val[st + off] = I.val[j];
      It.col[st + off] = i;
      num[I.col[j]]++;
    }
  }
  return It;
}

inline mat_t spgemm(const mat_t &A, const mat_t &B) {
  auto N = A.nrows();
  auto t_rptr = vec_t(N + 1);
  t_rptr[0] = 0;
  index_t res_nnz = 0;
#pragma omp parallel
  {
    auto marker = std::vector<index_t>(B.ncols(), -1);
#pragma omp for
    for (index_t i = 0; i < N; i++) {
      index_t nnz = 0;
      for (index_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
        index_t a_c = A.col[j];
        for (index_t k = B.rptr[a_c]; k < B.rptr[a_c + 1]; k++) {
          index_t b_c = B.col[k];
          if (marker[b_c] != i) {
            marker[b_c] = i;
            nnz++;
          }
        }
      }
      t_rptr[i + 1] = nnz;
    }
#pragma omp for reduction(+ : res_nnz)
    for (index_t i = 1; i < N + 1; i++) {
      res_nnz += t_rptr[i];
    }
  }
  auto t_val = vector<double, host>(res_nnz);
  auto t_idx = vec_t(res_nnz);
  for (index_t i = 0; i < N; i++)
    t_rptr[i + 1] += t_rptr[i];

  index_t zero_cnt = 0;
#pragma omp parallel
  {
    auto marker = std::vector<index_t>(B.ncols(), -1);
#pragma omp for
    for (index_t i = 0; i < N; i++) {
      index_t row_beg = t_rptr[i];
      index_t row_end = row_beg;
      for (index_t j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
        index_t a_c = A.col[j];
        double a_v = A.val[j];

        for (index_t k = B.rptr[a_c]; k < B.rptr[a_c + 1]; k++) {
          index_t b_c = B.col[k];
          double b_v = B.val[k];

          if (marker[b_c] < row_beg) {
            marker[b_c] = row_end;
            t_idx[row_end] = b_c;
            t_val[row_end] = a_v * b_v;
            row_end++;
          } else {
            t_val[marker[b_c]] += a_v * b_v;
          }
        }
      }
      sort::pack_sort<sort::order::asc>(
          row_beg, row_end, t_idx.raw(), t_val.raw());
      for (index_t ii = row_beg; ii < row_end; ii++) {
        if (t_val[ii] == 0) {
#pragma omp critical
          zero_cnt++;
        }
      }
    }
  }

  if (zero_cnt != 0) {
    res_nnz = 0;
    index_t cnt = 0;
    for (index_t i = 0; i < N; i++) {
      for (index_t j = t_rptr[i] + cnt; j < t_rptr[i + 1]; j++) {
        if (t_val[j] == 0) {
          cnt++;
          continue;
        }
        t_val[res_nnz] = t_val[j];
        t_idx[res_nnz] = t_idx[j];
        res_nnz++;
      }
      t_rptr[i + 1] = res_nnz;
    }
  }

  auto res = mat_t({N, B.ncols()}, res_nnz);
  res.val.copy(t_val);
  res.col.copy(t_idx);
  res.rptr.copy(t_rptr);

  return res;
}

inline mat_t get_coarse_matrix(
    const mat_t &A, const mat_t &I, const mat_t &It) {
  auto temp = spgemm(It, A);
  auto CA = spgemm(temp, I);
  return CA;
}

} // namespace impl

} // namespace senk

#endif