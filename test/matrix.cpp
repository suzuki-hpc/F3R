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

TEST_CASE("IO & reordering") {
  using namespace senk;
  auto A = CSR<double, host>("../data/wang3.mtx");

  auto v = vector<double, host>(A.ncols()).random(std::ranlux24_base(0), -1, 1);
  auto w1 = vector<double, host>(A.nrows());
  auto w2 = vector<double, host>(A.nrows());
  A.apply(v, w1);

  {
    auto R = LOCO(A, {});
    auto rev = R.apply(A, v);
    A.apply(v, w2);
    rev.apply(w2);
    CHECK(w1 == w2);
    rev.apply(A, v);
  }
  {
    auto R = CM(A, {true});
    auto rev = R.apply(A, v);
    A.apply(v, w2);
    rev.apply(w2);
    CHECK(w1 == w2);
    rev.apply(A, v);
  }
  {
    auto R = MC(A, {coloring::greedy});
    auto rev = R.apply(A, v);
    A.apply(v, w2);

    // for (int i = 0; i < A.nrows(); i++) {
    //   for (int j = A.rptr[i]; j < A.rptr[i + 1]; j++) {
    //     printf("%d ", A.col[j]);
    //   }
    //   printf("\n");
    // }

    rev.apply(w2);
    CHECK(w1 == w2);
    rev.apply(A, v);
  }
}

TEST_CASE_TEMPLATE("SpMV", T, double, float) {
  using namespace senk;
  auto A = CSR<T, host>("../data/wang3.mtx");
  // auto A = CSR<T, host>("/Users/kengo/matrix/G3_circuit.mtx");
  auto AA = CSR<T, host, index_t>(A);
  auto AAA = CSR<T, host, index_t>(AA);
  auto sA = SIGMA<4, CSR<T, host>>(A);
  auto B = SELL32<T, host>(A);
  auto sB = SIGMA<4, SELL32<T, host>>(A);
  auto C = BSR<8, 8, T, host>(A);

  auto v = vector<T, host>(A.ncols()).random(std::ranlux24_base(0), -1, 1);

  auto w1 = vector<T, host>(A.nrows());
  auto w2 = vector<T, host>(A.nrows());
  auto w3 = vector<T, host>(A.nrows());
  auto w4 = vector<T, host>(A.nrows());
  auto w5 = vector<T, host>(A.nrows());
  auto w6 = vector<T, host>(A.nrows());

  A.apply(v, w1);
  sA.apply(v, w2);
  B.apply(v, w3);
  sB.apply(v, w4);
  C.apply(v, w5);
  AA.apply(v, w6);

  CHECK(w1 == w2);
  CHECK(w1 == w3);
  CHECK(w1 == w4);
  CHECK(w1 == w5);
  CHECK(w1 == w6);
}

TEST_CASE_TEMPLATE("SpMV2", T, double, float) {
  using namespace senk;
  auto csr = CSR<T, host>("../data/wang3.mtx");
  // auto csr = CSR<T, host>("/Users/kengo/matrix/G3_circuit.mtx");
  // auto data = CSR<T, host>("/Users/kengo/matrix/Bump_2911.mtx");
  // auto data = CSR<T, host>("/Users/kengo/matrix/audikw_1.mtx");

  auto bsr = BSR<2, 1, T, host>(csr);

  auto nrows = csr.nrows();
  auto ncols = csr.ncols();

  auto A0 = SpMV(csr);
  auto A1 = SpMV<spmv::algo_default>(csr);
  auto A2 = SpMV<spmv::algo_reduce<1>>(csr);
  auto A3 = SpMV<spmv::algo_reduce_opt>(csr);

  auto B0 = SpMV(bsr);

  auto v = vector<T, host>(ncols).random(std::ranlux24_base(0), -1, 1);
  auto w = vector<T, host>(ncols).random(std::ranlux24_base(1), -1, 1);

  auto w1 = vector<T, host>(nrows);
  auto w2 = vector<T, host>(nrows);
  auto w3 = vector<T, host>(nrows);
  auto w4 = vector<T, host>(nrows);
  auto w5 = vector<T, host>(nrows);
  auto w6 = vector<T, host>(nrows);

  csr.apply(v, w1);
  A0.apply(v, w2);
  A1.apply(v, w3);
  A2.apply(v, w4);
  A3.apply(v, w5);
  CHECK(w1 == w2);
  CHECK(w1 == w3);
  CHECK(w1 == w4);
  CHECK(w1 == w5);
  B0.apply(v, w2);
  CHECK(w1 == w2);

  csr.residual(w, v, w1);
  A0.residual(w, v, w2);
  A1.residual(w, v, w3);
  A2.residual(w, v, w4);
  A3.residual(w, v, w5);

  CHECK(w1 == w2);
  CHECK(w1 == w3);
  CHECK(w1 == w4);
  CHECK(w1 == w5);
}

TEST_CASE_TEMPLATE("SpTRSV", T, double, float) {
  using namespace senk;

  auto A = CSR<T, host>("../data/wang3.mtx");

  auto v = vector<T, host>(A.ncols()).random(std::ranlux24_base(0), -1, 1);
  auto w1 = vector<T, host>(A.nrows());
  auto w2 = vector<T, host>(A.nrows());
  auto w3 = vector<T, host>(A.nrows());
  auto w4 = vector<T, host>(A.nrows());

  auto [L, U] = A.split_l1_du();
  auto L1 = trsv::l::CSR<T, host, strategy::direct>(L);
  auto L2 = trsv::ld::CSR<T, host, strategy::direct>(L);
  auto U1 = trsv::du::CSR<T, host, strategy::direct>(U);

  auto I1 = concat<T>(L, L1);
  auto I2 = concat<T>(L, L2);
  auto I3 = concat<T>(U, U1);
  I1.apply(v, w2);
  CHECK(v == w2);
  I2.apply(v, w2);
  CHECK(v == w2);
  I3.apply(v, w2);
  CHECK(v == w2);
}

TEST_CASE_TEMPLATE("SpTRSV-ordering", T, double, float) {
  using namespace senk;

  auto A = CSR<T, host>("../data/wang3.mtx");

  auto R = MC(A, {coloring::greedy});
  auto rev = R.apply(A);

  auto v = vector<T, host>(A.ncols()).random(std::ranlux24_base(0), -1, 1);
  auto w1 = vector<T, host>(A.nrows());
  auto w2 = vector<T, host>(A.nrows());
  auto w3 = vector<T, host>(A.nrows());
  auto w4 = vector<T, host>(A.nrows());

  auto [L, U] = A.split_l1_du();

  {
    auto L1 = trsv::l::CSR<T, host, strategy::level>(L);
    auto L2 = trsv::l::CSR<T, host, strategy::direct>(L);
    L1.apply(v, w1);
    L2.apply(v, w2);
    CHECK(w1 == w2);
  }
  {
    auto L1 = trsv::ld::CSR<T, host, strategy::level>(L);
    auto L2 = trsv::ld::CSR<T, host, strategy::direct>(L);
    L1.apply(v, w1);
    L2.apply(v, w2);
    CHECK(w1 == w2);
  }
  {
    auto U1 = trsv::du::CSR<T, host, strategy::level>(U);
    auto U2 = trsv::du::CSR<T, host, strategy::direct>(U);
    U1.apply(v, w1);
    U2.apply(v, w2);
    CHECK(w1 == w2);
  }
}
