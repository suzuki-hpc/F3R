#include "doctest/doctest.h"

#include "senk/matrix/bsr.hpp"
#include "senk/matrix/csr.hpp"
#include "senk/matrix/sell.hpp"

#include "senk/matrix/ordering/cm.hpp"
#include "senk/matrix/ordering/color.hpp"
#include "senk/matrix/ordering/loco.hpp"

#include "senk/spmv/bsr.hpp"
#include "senk/spmv/csr.hpp"
#include "senk/trsv/csr.hpp"

namespace senk {

template <typename T>
bool operator==(const vector<T, host> &lhs, const vector<T, host> &rhs) {
  double tol = 1.e-12;
  if (std::is_same_v<T, float>)
    tol = 1.e-6;
  for (size_t i = 0; i < lhs.shape(0); ++i) {
    if (std::abs(lhs[i] - rhs[i]) > tol)
      if (std::abs((lhs[i] - rhs[i]) / lhs[i]) > tol)
        return false;
  }
  return true;
}

} // namespace senk

TEST_CASE_TEMPLATE("SpMV", T, double, float) {
  using namespace senk;
  auto data = CSR<T, host>("../data/wang3.mtx");
  auto A = CSR<T, device>(data);
  auto AA = CSR<T, device, index_t>(A);
  auto AAA = CSR<T, device, index_t>(A);
  auto sA = SIGMA<4, CSR<T, device>>(data);
  auto B = SELL32<T, device>(data);
  auto sB = SIGMA<4, SELL32<T, device>>(data);

  auto v = vector<T, device>(A.ncols()).random(std::ranlux24_base(0), -1, 1);

  auto w1 = vector<T, device>(A.nrows());
  auto w2 = vector<T, device>(A.nrows());
  auto w3 = vector<T, device>(A.nrows());
  auto w4 = vector<T, device>(A.nrows());
  auto w5 = vector<T, device>(A.nrows());
  auto w6 = vector<T, device>(A.nrows());

  A.apply(v, w1);
  sA.apply(v, w2);
  B.apply(v, w3);
  sB.apply(v, w4);
  AA.apply(v, w5);
  AAA.apply(v, w6);

  auto hw1 = vector<T, host>(w1);
  auto hw2 = vector<T, host>(w2);
  auto hw3 = vector<T, host>(w3);
  auto hw4 = vector<T, host>(w4);
  auto hw5 = vector<T, host>(w5);
  auto hw6 = vector<T, host>(w6);

  CHECK(hw1 == hw2);
  CHECK(hw1 == hw3);
  CHECK(hw1 == hw4);
  CHECK(hw1 == hw5);
  CHECK(hw1 == hw6);
}

TEST_CASE_TEMPLATE("SpMV2", T, double, float) {
  using namespace senk;
  auto _csr = CSR<T, host>("../data/apache1.mtx");
  auto csr = CSR<T, device>(_csr);

  auto nrows = csr.nrows();
  auto ncols = csr.ncols();

  auto A0 = SpMV(csr);
  auto A1 = SpMV<spmv::algo_default>(csr);
  auto A3 = SpMV<spmv::algo_reduce_opt>(csr);
  auto v = vector<T, device>(ncols).random(std::ranlux24_base(0), -1, 1);

  auto w1 = vector<T, device>(nrows);
  auto w2 = vector<T, device>(nrows);
  auto w3 = vector<T, device>(nrows);
  auto w4 = vector<T, device>(nrows);

  csr.apply(v, w1);
  A0.apply(v, w2);
  A1.apply(v, w3);
  A3.apply(v, w4);

  auto hw1 = vector<T, host>(w1);
  auto hw2 = vector<T, host>(w2);
  auto hw3 = vector<T, host>(w3);
  auto hw4 = vector<T, host>(w4);

  CHECK(hw1 == hw2);
  CHECK(hw1 == hw3);
  CHECK(hw1 == hw4);
}

TEST_CASE_TEMPLATE("SpTRSV", T, double) {
  using namespace senk;

  auto data = CSR<T, host>("../data/wang3.mtx");
  auto A = CSR<T, device>(data);

  auto v = vector<T, device>(A.ncols()).random(std::ranlux24_base(0), -1, 1);
  auto w1 = vector<T, device>(A.nrows());
  auto w2 = vector<T, device>(A.nrows());
  auto w3 = vector<T, device>(A.nrows());

  auto [l, u] = data.split_l1_du();
  auto L = CSR<T, device>(l);
  auto U = CSR<T, device>(u);
  auto L1 = trsv::l::CSR<T, device, strategy::direct>(l);
  auto L2 = trsv::ld::CSR<T, device, strategy::direct>(l);
  auto U1 = trsv::du::CSR<T, device, strategy::direct>(u);

  auto I1 = concat<T>(L, L1);
  auto I2 = concat<T>(L, L2);
  auto I3 = concat<T>(U, U1);
  I1.apply(v, w1);
  I2.apply(v, w2);
  I3.apply(v, w3);

  auto hv = vector<T, host>(v);
  auto hw1 = vector<T, host>(w1);
  auto hw2 = vector<T, host>(w2);
  auto hw3 = vector<T, host>(w3);

  CHECK(hv == hw1);
  CHECK(hv == hw2);
  CHECK(hv == hw3);
}