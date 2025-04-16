#ifndef UMINEKO_SPARSE_PRECONDITIONER_ILU_HPP
#define UMINEKO_SPARSE_PRECONDITIONER_ILU_HPP

#include "umineko-sparse/matrix/algorithm.hpp"
#include "umineko-sparse/matrix/bcsr.hpp"

#include "umineko-sparse/models.hpp"
#include "umineko-sparse/trsv/baj.hpp"
#include "umineko-sparse/trsv/direct.hpp"
#include "umineko-sparse/trsv/jacobi.hpp"

#include "umineko-sparse/solver/richardson.hpp"

namespace kmm {

namespace impl {

template <typename T, class L>
CSR<T, L> _ilu0(const CSR<T, L> &in, double acc, uint16_t b_num, uint16_t unit);
template <typename T, class L>
CSR<T, L> _ilup(const CSR<T, L> &in, int level, double acc);

} // namespace impl

namespace algorithm {

template <typename T, class L>
CSR<T, L> ilup(const CSR<T, L> &in, int level, double acc, uint16_t b_num = 1,
    uint16_t unit = 1) {
  static_assert(std::is_same_v<host, L>, "ilup must be on host at this time");
  return (level == 0) ? impl::_ilu0(in, acc, b_num, unit)
                      : impl::_ilup(in, level, acc);
}

template <uint16_t bnl, uint16_t bnw, typename T, class L>
CSR<T, L> ilub(const CSR<T, L> &in, int level, double acc, uint16_t b_num = 1,
    uint16_t unit = bnl) {
  static_assert(std::is_same_v<host, L>, "ilub must be on host at this time");
  auto bcsr = BCSR<bnl, bnw, T, L>(in.duplicate_block_diagonal(b_num, unit));
  auto t = impl::bcsr_to_csr(bcsr);
  auto res = CSR<T, L>(static_cast<spmat>(bcsr));
  res.val.copy(std::get<2>(t));
  res.idx.copy(std::get<3>(t));
  res.rptr.copy(std::get<4>(t));
  return (level == 0) ? impl::_ilu0(res, acc, b_num, unit)
                      : impl::_ilu0(res, acc, b_num, unit);
}

} // namespace algorithm

template <typename T, class L> struct ILU {
  struct factories {
    std::function<Operator<L>(const CSR<double, host> &)> l;
    std::function<Operator<L>(const CSR<double, host> &)> u;
  };

  template <typename val1_t, typename val2_t>
  void operate(const vector<val1_t, L> &in, vector<val2_t, L> out) const {
    Lower.operate(in, temp);
    Upper.operate(temp, out);
  }

  [[nodiscard]] idx_t nrows() const { return Lower.nrows(); }
  [[nodiscard]] idx_t ncols() const { return Upper.ncols(); }

protected:
  Operator<L> Lower;
  Operator<L> Upper;
  vector<T, L> temp;

  ILU(std::tuple<CSR<double, host>, CSR<double, host>> &&t, factories factory)
      : Lower(factory.l(std::get<0>(t))), Upper(factory.u(std::get<1>(t))),
        temp(Lower.nrows()) {}
};

template <typename T, class L> struct ILUp : ILU<T, L> {
  struct params {
    int p = 0;
    double alpha = 1.0;
    uint16_t b_num = 1;
    uint16_t unit = 1;
  };

  ILUp(const CSR<double, host> &A, params param,
      typename ILU<T, L>::factories factory =
          {Operator<L>::template factory<trsv::l::Direct<CSR<double, host>>>(),
              Operator<L>::template factory<trsv::du::Direct<CSR<double, host>>>()})
      : ILU<T, L>(algorithm::split(algorithm::ilup(
                      A, param.p, param.alpha, param.b_num, param.unit)),
            factory) {}

  template <typename _>
  static std::function<Solver<L>(const CSR<_, host> &in)> Solver(params param) {
    return [param](const CSR<_, host> &in) {
      return kmm::Solver<L>(Richardson<T, L>(in, ILUp(in, param), {2, 1.0}));
    };
  }
};

template <uint16_t bnl, uint16_t bnw, typename T, class L> struct ILUBp : ILU<T, L> {
  struct params {
    int p = 0;
    double alpha = 1.0;
    uint16_t b_num = 1;
    uint16_t unit = 1;
  };

  ILUBp(const CSR<double, host> &A, params param,
      typename ILU<T, L>::factories factory =
          {Operator<L>::template factory<
               trsv::l::Direct<BCSR<bnl, bnw, double, host>>>(),
              Operator<L>::template factory<
                  trsv::du::Direct<BCSR<bnl, bnw, double, host>>>()})
      : ILU<T, L>(algorithm::split(algorithm::ilub<bnl, bnw>(
                      A.duplicate(), param.p, param.alpha, param.b_num, param.unit)),
            factory) {}

  template <typename _>
  static std::function<Solver<L>(const CSR<_, host> &in)> Solver(params param) {
    return [param](const CSR<_, host> &in) {
      return kmm::Solver<L>(Richardson<T, L>(in, ILUBp(in, param), {2, 1.0}));
    };
  }
};

namespace impl {

template <typename T, class L>
CSR<T, L> _ilu0(const CSR<T, L> &in, double acc, uint16_t b_num, uint16_t unit) {
  auto res = in.duplicate_block_diagonal(b_num, unit);
  auto val = res.val.raw();
  auto cind = res.idx.raw();
  auto rptr = res.rptr.raw();
  res.val[0] *= acc; // if cind[0] == 0

#pragma omp parallel for
  for (idx_t id = 0; id < res.pattern_set.idx_size; id++) {
    auto s = id == 0 ? 1 : res.pattern_set.idx_ptr[id];
    auto e = res.pattern_set.idx_ptr[id + 1];
    for (idx_t i = s; i < e; ++i) {
      auto d = res.rptr[i];
      while (cind[d] != i)
        ++d;
      val[d] *= acc;
      for (idx_t pvt = rptr[i]; pvt < rptr[i + 1] && cind[pvt] < i; ++pvt) {
        auto k = cind[pvt];
        auto ref = rptr[k];
        while (cind[ref] != cind[pvt])
          ++ref;
        val[pvt] = val[pvt] / val[ref]; // if val[l] == 0, zero div
        auto base = pvt + 1;
        while (base < rptr[i + 1] && ref < rptr[k + 1]) {
          if (cind[ref] < cind[base])
            ++ref;
          else if (cind[ref] > cind[base])
            ++base;
          else
            val[base++] -= val[pvt] * val[ref++];
        }
      }
    }
  }
  return res;
}

template <typename T, class L>
CSR<T, L> _ilup(const CSR<T, L> &in, int level, double acc) {
  auto val = in.val.raw();
  auto cind = in.idx.raw();
  auto rptr = in.rptr.raw();
  // val, col, level
  std::vector<std::tuple<T, idx_t, idx_t>> temp;
  std::vector<std::tuple<T, idx_t, idx_t>> temp2;

  T pivot = 0;
  idx_t pivot_lev = 0;
  auto N = in.nrows();

  idx_t first_len = rptr[1] - rptr[0];
  std::vector<T> new_val(first_len);
  std::vector<idx_t> new_cind(first_len);
  std::vector<idx_t> new_lev(first_len, 0);
  std::vector<idx_t> new_rptr(N + 1);
  idx_t new_len = first_len;
  // 一行目
  for (idx_t i = 0; i < first_len; i++) {
    if (cind[i] == 0)
      new_val[i] = val[i] * acc;
    else
      new_val[i] = val[i];
    new_cind[i] = cind[i];
  }
  new_rptr[0] = 0;
  new_rptr[1] = first_len;

  for (idx_t i = 1; i < N; i++) {
    idx_t off = rptr[i];
    idx_t now_len = rptr[i + 1] - off;
    temp.clear();
    for (idx_t l = 0; l < now_len; l++) {
      std::tuple<T, idx_t, idx_t> tup;
      if (cind[off + l] == i)
        tup = std::make_tuple(val[off + l] * acc, cind[off + l], 0);
      else {
        tup = std::make_tuple(val[off + l], cind[off + l], 0);
      }
      temp.push_back(tup);
    }
    idx_t count = 0;
    while (count < (idx_t)temp.size() && std::get<1>(temp[count]) < i) {
      idx_t k_ptr = count;
      if (std::get<2>(temp[k_ptr]) > level) {
        count++;
        continue;
      }
      idx_t k = std::get<1>(temp[k_ptr]);
      temp2.clear();
      for (idx_t l = 0; l < count; l++)
        temp2.push_back(temp[l]);
      idx_t j = new_rptr[k];
      for (; j < new_rptr[k + 1]; j++) {
        if (new_cind[j] == k) {
          pivot = std::get<0>(temp[k_ptr]) / new_val[j];
          pivot_lev = std::get<2>(temp[k_ptr]);
          break;
        }
      }
      j++;
      k_ptr++;
      temp2.push_back(std::make_tuple(pivot, k, 0));
      while (k_ptr < (idx_t)temp.size() || j < new_rptr[k + 1]) {
        T t_val;
        idx_t t_lev, t_col, t_col1, t_col2;
        t_col1 = (k_ptr < (idx_t)temp.size()) ? std::get<1>(temp[k_ptr]) : N + 1;
        t_col2 = (j < new_rptr[k + 1]) ? new_cind[j] : N + 1;
        if (t_col1 < t_col2) {
          t_val = std::get<0>(temp[k_ptr]);
          t_col = std::get<1>(temp[k_ptr]);
          t_lev = std::get<2>(temp[k_ptr]);
          k_ptr++;
        } else if (t_col1 == t_col2) {
          t_val = std::get<0>(temp[k_ptr]) - pivot * new_val[j];
          t_col = std::get<1>(temp[k_ptr]);
          t_lev = std::get<2>(temp[k_ptr]);
          if (t_lev > pivot_lev + new_lev[j] + 1)
            t_lev = pivot_lev + new_lev[j] + 1;
          k_ptr++;
          j++;
        } else { // (k>j) fill-in
          t_val = -pivot * new_val[j];
          t_col = new_cind[j];
          t_lev = pivot_lev + new_lev[j] + 1;
          j++;
        }
        temp2.push_back(std::make_tuple(t_val, t_col, t_lev));
      }
      count++;
      temp.clear();
      for (idx_t l = 0; l < (idx_t)temp2.size(); l++)
        temp.push_back(temp2[l]);
    }
    for (idx_t l = 0; l < (idx_t)temp.size(); l++) {
      if (std::get<2>(temp[l]) <= level) {
        new_val.push_back(std::get<0>(temp[l]));
        new_cind.push_back(std::get<1>(temp[l]));
        new_lev.push_back(std::get<2>(temp[l]));
        new_len++;
      }
    }
    new_rptr[i + 1] = new_len;
  }
  auto res = CSR<T, host>({in.nrows(), in.ncols()}, new_len);
  res.val.copy(&new_val[0]);
  res.idx.copy(&new_cind[0]);
  res.rptr.copy(&new_rptr[0]);
  return res;
}

} // namespace impl

} // namespace kmm

#endif // UMINEKO_SPARSE_PRECONDITIONER_ILU_HPP
