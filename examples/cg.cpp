#include "senk/core/timer.hpp"

#include "senk/matrix/csr.hpp"
#include "senk/preconditioner/ilu.hpp"
#include "senk/solver/cg.hpp"

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
  auto L = trsv::l::CSR<double, tag, strategy::partition>(l);
  auto U = trsv::du::CSR<double, tag, strategy::partition>(u);
  auto M = concat<double>(U, L);

  auto x = vector<double, tag>(A.nrows()).fill(0.0);
  auto r = vector<double, tag>(A.nrows());

  auto reduce = reducer<tag>(A.nrows());
  scalar<double, host> nrm_b, nrm_r;
  nrm_b = reduce.norm(b);
  auto solver = Solver<CG, double>(A, M, {19200});

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