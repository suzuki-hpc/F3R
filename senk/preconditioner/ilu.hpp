#ifndef SENK_PRECONDITIONER_ILU_HPP
#define SENK_PRECONDITIONER_ILU_HPP

#include "senk/core/tensor.hpp"
#include "senk/matrix/csr.hpp"
#include "senk/models.hpp"

#include "senk/trsv/csr.hpp"

namespace senk {

namespace impl {

struct ilup_params {
  int16_t level;
  double alpha;
};

template <typename T, class L>
CSR<T, L> _ilu0(const CSR<T, L> &in, double acc, uint16_t b_num, uint16_t unit);
template <typename T, class L>
CSR<T, L> _ilup(const CSR<T, L> &in, int level, double acc);

} // namespace impl

namespace algorithm {

auto ilup(const CSR<double, host> &in, impl::ilup_params prm) {
  auto LU = (prm.level == 0) ? impl::_ilu0(in, prm.alpha, 1, 1)
                             : impl::_ilup(in, prm.level, prm.alpha);
  return LU.split_l1_du();
}

} // namespace algorithm

template <typename T, class L, class strat = strategy::direct>
auto ILUp(const CSR<double, host> &in, impl::ilup_params prm,
    typename has_params<strat>::Params stprm = {}) {
  auto [l, u] = algorithm::ilup(in, prm);
  auto Linv = trsv::l::CSR<T, L, strat>(l, stprm);
  auto Uinv = trsv::du::CSR<T, L, strat>(u, stprm);
  return concat<T>(Uinv, Linv);
}

template <typename T, class L, class lstrat, class ustrat>
auto ILUp(const CSR<double, host> &in, impl::ilup_params prm,
    typename has_params<lstrat>::Params lstprm = {},
    typename has_params<ustrat>::Params ustprm = {}) {
  auto [l, u] = algorithm::ilup(in, prm);
  auto Linv = trsv::l::CSR<T, L, lstrat>(l, lstprm);
  auto Uinv = trsv::du::CSR<T, L, ustrat>(u, ustprm);
  return concat<T>(Uinv, Linv);
}

namespace impl {

template <typename T, class L>
CSR<T, L> _ilu0(const CSR<T, L> &in, double acc,
    [[maybe_unused]] uint16_t b_num, [[maybe_unused]] uint16_t unit) {
  using idx_t = typename CSR<T, L>::idx_t;
  auto res = in.duplicate();
  auto val = res.val.raw();
  auto cind = res.col.raw();
  auto rptr = res.rptr.raw();

  for (idx_t i = 0; i < in.nrows(); ++i) {
    auto d = res.rptr[i];
    while (cind[d] != i)
      ++d;
    val[d] *= acc;
    for (idx_t pvt = rptr[i]; pvt < rptr[i + 1] && cind[pvt] < i; ++pvt) {
      auto k = cind[pvt];
      auto ref = rptr[k];
      while (cind[ref] != cind[pvt])
        ++ref;
      if (val[ref] == 0) {
        printf("ILU; division by zero\n");
        exit(1);
      }
      val[pvt] = val[pvt] / val[ref];
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

  return res;
}

template <typename T, class L>
CSR<T, L> _ilup(const CSR<T, L> &in, int level, double acc) {
  using idx_t = typename CSR<T, L>::idx_t;
  using srl_t = typename CSR<T, L>::srl_t;
  auto val = in.val.raw();
  auto cind = in.col.raw();
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
  std::vector<srl_t> new_rptr(N + 1);
  idx_t new_len = first_len;

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
      srl_t j = new_rptr[k];
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
        t_col1 =
            (k_ptr < (idx_t)temp.size()) ? std::get<1>(temp[k_ptr]) : N + 1;
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
  res.col.copy(&new_cind[0]);
  res.rptr.copy(&new_rptr[0]);
  return res;
}

} // namespace impl

} // namespace senk

#endif