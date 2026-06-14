#include "doctest/doctest.h"

#include "senk/matrix/csr.hpp"
#include "senk/trsv/csr.hpp"

#include "senk/preconditioner/ilu.hpp"

#include "senk/solver/stationary.hpp"

#include "senk/solver/cg.hpp"
#include "senk/solver/cr.hpp"
#include "senk/solver/minres.hpp"

#include "senk/solver/bicgstab.hpp"
#include "senk/solver/gmres.hpp"

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

TEST_CASE("solver Jacobi/GS") {
  using namespace senk;
  auto data = CSR<double, host>("../data/wang3.mtx");
  data.scaling();

  auto A = CSR<double, device>(data);
  auto ans = vector<double, device>(A.ncols());
  auto x = vector<double, device>(A.ncols()).fill(0);
  auto b = vector<double, device>(A.nrows());
  auto r = vector<double, device>(A.nrows());
  auto reduce = reducer<device>(A.nrows());
  auto norm_b = scalar<double, host>();
  auto norm_r = scalar<double, host>();

  ans.random(std::ranlux24_base(0), 0, 1);
  A.apply(ans, b);
  norm_b = reduce.norm(b);

  auto solver_test = [&](auto solver) {
    x.fill(0.);
    auto flag = solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    while (!flag.is_solved && flag.res_iter < 8000)
      flag += solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    flag = solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    A.residual(b, x, r);
    norm_r = reduce.norm(r);
    norm_r /= norm_b;
    CHECK(norm_r[0] < 1.e-5);
  };

  {
    auto [d, lu] = data.split_d_lu();
    auto D = diagonal(vector<double, device>(d));
    auto LU = CSR<double, device>(lu);
    solver_test(Solver<Stationary, double>(LU, D, {1000, 1.0}));
  }
#if defined(SENK_WITH_CUDA)
  {
    auto [ld, u] = data.split_ld_u();
    auto LD = trsv::ld::CSR<double, device, strategy::direct>(ld);
    auto U = CSR<double, device>(u);
    solver_test(Solver<Stationary, double>(U, LD, {1000, 1.0}));
  }
  {
    auto [l, du] = data.split_l_du();
    auto L = CSR<double, device>(l);
    auto DU = trsv::du::CSR<double, device, strategy::direct>(du);
    solver_test(Solver<Stationary, double>(L, DU, {1000, 1.0}));
  }
#endif
}

TEST_CASE("solver CG/CR/MIRES") {
  using namespace senk;
  auto data = CSR<double, host>("../data/apache1.mtx");
  data.scaling();

  auto A = CSR<double, device>(data);

  auto ans = vector<double, device>(A.ncols());
  auto x = vector<double, device>(A.ncols()).fill(0);
  auto b = vector<double, device>(A.nrows());
  auto r = vector<double, device>(A.nrows());
  auto reduce = reducer<device>(A.nrows());
  auto norm_b = scalar<double, host>();
  auto norm_r = scalar<double, host>();

  ans.random(std::ranlux24_base(0), 0, 1);
  A.apply(ans, b);
  norm_b = reduce.norm(b);

  // auto M = lambda<host>(A.shape, [](auto in, auto out) { out.copy(in); });
  auto M = ILUp<double, device, strategy::jacobi>(A, {0, 1.0}, {5});

  auto solver_test = [&](auto solver) {
    x.fill(0.);
    auto flag = solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    while (!flag.is_solved && flag.res_iter < 5000)
      flag += solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    flag = solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    A.residual(b, x, r);
    norm_r = reduce.norm(r);
    norm_r /= norm_b;
    CHECK(norm_r[0] < 1.e-5);
  };

  solver_test(Solver<CG, double>(A, M, {1000}));
  solver_test(Solver<CR, double>(A, M, {1000}));
  solver_test(Solver<MINRES, double>(A, M, {1000, false}));
}

TEST_CASE("solver BiCGStab/GMRES/FGMRES") {
  using namespace senk;
  auto data = CSR<double, host>("../data/wang3.mtx");
  data.scaling();

  auto A = CSR<double, device>(data);

  auto ans = vector<double, device>(A.ncols());
  auto x = vector<double, device>(A.ncols()).fill(0);
  auto b = vector<double, device>(A.nrows());
  auto r = vector<double, device>(A.nrows());
  auto reduce = reducer<device>(A.nrows());
  auto norm_b = scalar<double, host>();
  auto norm_r = scalar<double, host>();

  ans.random(std::ranlux24_base(0), 0, 1);
  A.apply(ans, b);
  norm_b = reduce.norm(b);

  // auto M = ILUp<double, device>(A, {0, 1.0});
  auto M = ILUp<double, device, strategy::jacobi>(A, {0, 1.0}, {5});

  auto solver_test = [&](auto solver) {
    x.fill(0.);
    auto flag = solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    while (!flag.is_solved && flag.res_iter < 5000)
      flag += solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    flag = solver.solve(b, x, converg::rrn(norm_b[0], 1.e-5));
    A.residual(b, x, r);
    norm_r = reduce.norm(r);
    norm_r /= norm_b;
    CHECK(norm_r[0] < 1.e-5);
  };

  solver_test(Solver<BiCGStab, double>(A, M, {1000}));
  solver_test(Solver<GMRES, double>(A, M, {50, false}));
  solver_test(Solver<FGMRES, double>(A, M, {50, false}));
}
