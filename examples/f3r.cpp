#include "senk/core/timer.hpp"

#include "senk/matrix/csr.hpp"
#include "senk/preconditioner/ilu.hpp"
#include "senk/solver/gmres.hpp"
#include "senk/solver/richardson.hpp"

using namespace senk;

int main(int argc, char *argv[]) {
  auto coeff = CSR<double, host>(argv[1]);
  auto rhs =
      vector<double, host>(coeff.nrows()).random(std::mt19937_64(0), 0, 1);
  coeff.scaling(rhs);
  auto [l, u] = algorithm::ilup(coeff.duplicate_block(112), {0, 1.0});

  using tag = host;

  auto A = CSR<double, tag>(coeff);
  auto b = vector<double, tag>(rhs);

  auto x = vector<double, tag>(A.nrows()).fill(0.0);
  auto r = vector<double, tag>(A.nrows());

  auto reduce = reducer<tag>(A.nrows());
  scalar<double, host> nrm_b, nrm_r;
  nrm_b = reduce.norm(b);

  auto A32 = CSR<float, tag>(A);
  auto A16 = CSR<half, tag>(A);
  auto L = trsv::l::CSR<half, host, strategy::partition>(l);
  auto U = trsv::du::CSR<half, host, strategy::partition>(u);
  auto M = concat<half>(U, L);
  auto R = Solver<RichardsonAdapt, half>(A, M, {2, 1.0, 64});
  auto F = Solver<FGMRES, float>(A16, R, {4, true});
  auto F2 = Solver<FGMRES, float>(A32, F, {8, true});
  auto solver = Solver<FGMRES, double>(A, F2, {100, false});

  auto t = timer();
  t.tick();
  auto res = solver.solve(b, x, converg::rrn_log(nrm_b[0], 1.e-8));
  t.tock();

  A.residual(b, x, r);
  nrm_r = reduce.norm(r);

  printf("%d %e\n", res.res_iter, nrm_r[0] / nrm_b[0]);
  t.print("Time [s]: ");

  return 0;
}