#include "senk/core/timer.hpp"

#include "senk/matrix/csr.hpp"
#include "senk/matrix/sell.hpp"
#include "senk/preconditioner/ainv.hpp"
#include "senk/solver/bicgstab.hpp"
#include "senk/solver/cg.hpp"
#include "senk/solver/gmres.hpp"

using namespace senk;

const double eps = 1.e-8;

#define _STR(x) #x
#define STR(x) _STR(x)

const std::string type_name = STR(TYPE);
const std::string precond_name = "SDAINV";
using precond_type = TYPE;

int main(int argc, char *argv[]) {
  std::string path = std::string("../matrix/") + argv[1];
  double acc = atof(argv[2]);

  int suite_iter = atoi(argv[3]);

  [[maybe_unused]] int m2 = 0;
  [[maybe_unused]] int m3 = 0;
  [[maybe_unused]] int m4 = 0;
  [[maybe_unused]] int c = 0;

  auto _name = std::string(argv[1]);
  auto name = _name.substr(0, _name.size() - 4);
  auto precond = precond_name + "," + argv[2] + ",";

  auto data = CSR<double, host>(path);
  data.scaling();
  auto rhs =
      vector<double, host>(data.nrows()).random(std::mt19937_64(0), 0, 1);

  using tag = device;

  auto A = SELL32<double, tag>(data);
  auto b = vector<double, tag>(rhs);
  auto x = vector<double, tag>(A.nrows());
  auto r = vector<double, tag>(A.nrows());

  auto test = [&A, &x, &r, &suite_iter](auto solver, auto b, auto &t, bool ff,
                  int &itr_sum, int ww = 1) {
    x.fill(0.0);
    auto reduce = reducer<tag>(A.nrows());
    scalar<double, host> nrm_b, nrm_r;
    nrm_b = reduce.norm(b);

    auto cond = converg::rrn(nrm_b[0], eps);

    t.tick();
    auto flag = solver.solve(b, x, cond);
    t.tock();

    itr_sum += flag.res_iter;

    if (ff | !flag.is_solved) {
      double sum = 0;
      for (const auto &d : t.durations)
        sum += d.count();
      if (!flag.is_solved) {
        printf("%e,", sum);
        printf("%d,%e,", itr_sum * ww, flag.res_nrm2 / nrm_b[0]);
      } else {
        printf("%e,", sum / suite_iter);
        printf("%d,%e,", itr_sum * ww / suite_iter, flag.res_nrm2 / nrm_b[0]);
      }
      A.residual(b, x, r);
      nrm_r = reduce.norm(r);
      printf("%e\n", nrm_r[0] / nrm_b[0]);
    }

    return flag.is_solved;
  };

  auto [z, w] = algorithm::sdainv(data, 0.1, acc);
  auto W = SELL<32, precond_type, tag>(w);
  auto Z = SELL<32, precond_type, tag>(z);
  auto M = concat<double>(Z, W);

#if defined(SOLV_BiCG)
  const std::string solver_name = "BiCGStab";
  auto bicg = Solver<BiCGStab, double>(A, M, {19200});
  auto solver = [&bicg](
                    auto b, auto x, auto cnv) { return bicg.solve(b, x, cnv); };
  int ww = 2;
#endif

#if defined(SOLV_CG)
  const std::string solver_name = "CG";
  auto cg = Solver<CG, double>(A, M, {19200});
  auto solver = [&cg](auto b, auto x, auto cnv) { return cg.solve(b, x, cnv); };
  int ww = 1;
#endif

#if defined(SOLV_GM)
  const std::string solver_name = "GMRES";
  auto gmres = Solver<tag>(FGMRES<double, tag>(A, M, {64, true}));
  auto solver = [&gmres](auto b, auto x, auto cnv) {
    auto res = gmres.solve(b, x, cnv);
    while (!res.is_solved && res.res_iter < 19200)
      res += gmres.solve(b, x, cnv);
    return res;
  };
  int ww = 1;
#endif

  std::cout << name << "," << solver_name << "," << type_name << "," << m2
            << "," << m3 << "," << m4 << "," << c << "," << precond;

  auto t = timer();
  int itr_sum = 0;
  for (int i = 0; i < suite_iter; i++) {
    bool is_solved = (i == suite_iter - 1)
                         ? test(solver, b, t, true, itr_sum, ww)
                         : test(solver, b, t, false, itr_sum, ww);
    if (!is_solved)
      break;
  }

  return 0;
}
