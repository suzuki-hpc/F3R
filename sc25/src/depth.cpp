#include "senk/core/timer.hpp"
#include "senk/matrix/csr.hpp"

#include "senk/solver/gmres.hpp"
#include "senk/solver/richardson.hpp"

#include "senk/preconditioner/ilu.hpp"
#include "senk/trsv/csr.hpp"

using namespace senk;

const double eps = 1.e-8;

#define _STR(x) #x
#define STR(x) _STR(x)

const std::string type_name = STR(TYPE);
const std::string precond_name = "BJILU0";
using precond_type = TYPE;

int main(int argc, char *argv[]) {
  std::string path = std::string("matrix/") + argv[1];
  // std::string path = argv[1];
  double acc = atof(argv[2]);

  int suite_iter = atoi(argv[3]);

  int m2 = atoi(argv[4]);
  int m3 = atoi(argv[5]);
  int m4 = atoi(argv[6]);
  int c = atoi(argv[7]);
  int restart = atoi(argv[8]);

  auto _name = std::string(argv[1]);
  auto name = _name.substr(0, _name.size() - 4);
  auto precond = std::string("BJILU0,") + argv[2] + ",";

  auto data = CSR<double, host>(path);
  data.scaling();
  auto rhs =
      vector<double, host>(data.nrows()).random(std::mt19937_64(0), 0, 1);

  using tag = host;

  auto A = CSR<double, tag>(data);
  auto b = vector<double, tag>(rhs);
  auto x = vector<double, tag>(A.nrows());
  auto r = vector<double, tag>(A.nrows());

  auto bd = data.duplicate_block(112);
  auto [l, u] = algorithm::ilup(bd, {0, acc});

  auto test = [&A, &x, &r, &suite_iter](
                  auto solver, auto b, auto &t, bool ff, int &itr_sum, int ww) {
    x.fill(0.0);
    auto reduce = reducer<tag>(A.nrows());
    scalar<double, host> nrm_b, nrm_r;
    nrm_b = reduce.norm(b);

#if defined(LOGGING)
    auto cond = converg::rrn_log(nrm_b[0], eps);
#else
    auto cond = converg::rrn_log(nrm_b[0], eps);
#endif

    t.tick();
    auto flag = solver(b, x, cond);
    t.tock();

    itr_sum += flag.res_iter;

    if (ff) {
      double sum = 0;
      for (const auto &d : t.durations)
        sum += d.count();
#if !defined(LOGGING)
      printf("%e,", sum / suite_iter);
      printf("%d,%e,", itr_sum * ww / suite_iter, flag.res_nrm2 / nrm_b[0]);
#endif
      A.apply(x, r);
      r = b - r;
      nrm_r = reduce.norm(r);
#if !defined(LOGGING)
      printf("%e\n", nrm_r[0] / nrm_b[0]);
#endif
    }
  };

  auto L = trsv::l::CSR<half, host, strategy::partition>(l);
  auto U = trsv::du::CSR<half, host, strategy::partition>(u);
  auto M = concat<half>(U, L);

#if defined(F4)
  auto A16 = CSR<half, tag>(A);
  auto A32 = CSR<float, tag>(A);
  auto FF = Solver<FGMRES, half>(A16, M, {2, true});
  auto F = Solver<FGMRES, float>(A16, FF, {4, true});
  auto inner = Solver<FGMRES, float>(A32, F, {8, true});
  auto solver_ = Solver<FGMRES, double>(A, inner, {100, false});
  auto solver = [&solver_, &restart](auto b, auto x, auto cnv) {
    auto res = solver_.solve(b, x, cnv);
    while (!res.is_solved && res.res_iter < 100 * restart)
      res += solver_.solve(b, x, cnv);
    return res;
  };
  const std::string solver_name = "F4";
#endif

#if defined(F3)
  auto A16 = CSR<half, tag>(A);
  auto A32 = CSR<float, tag>(A);
  auto F = Solver<FGMRES, float>(A16, M, {8, true});
  auto inner = Solver<FGMRES, float>(A32, F, {8, true});
  auto solver_ = Solver<FGMRES, double>(A, inner, {100, false});
  auto solver = [&solver_, &restart](auto b, auto x, auto cnv) {
    auto res = solver_.solve(b, x, cnv);
    while (!res.is_solved && res.res_iter < 100 * restart)
      res += solver_.solve(b, x, cnv);
    return res;
  };
  const std::string solver_name = "F3";
#endif

#if defined(F3H)
  auto A16 = CSR<half, tag>(A);
  auto A32 = CSR<float, tag>(A);
  auto F = Solver<FGMRES, half>(A16, M, {8, true});
  auto inner = Solver<FGMRES, float>(A32, F, {8, true});
  auto solver_ = Solver<FGMRES, double>(A, inner, {100, false});
  auto solver = [&solver_, &restart](auto b, auto x, auto cnv) {
    auto res = solver_.solve(b, x, cnv);
    while (!res.is_solved && res.res_iter < 100 * restart)
      res += solver_.solve(b, x, cnv);
    return res;
  };
  const std::string solver_name = "F3H";
#endif

#if defined(F2)
  auto A32 = CSR<float, tag>(A);
  auto inner = Solver<FGMRES, float>(A32, M, {64, true});
  auto solver_ = Solver<FGMRES, double>(A, inner, {100, false});
  auto solver = [&solver_, &restart](auto b, auto x, auto cnv) {
    auto res = solver_.solve(b, x, cnv);
    while (!res.is_solved && res.res_iter < 100 * restart)
      res += solver_.solve(b, x, cnv);
    return res;
  };
  const std::string solver_name = "F2";
#endif

#if defined(F2H)
  auto A16 = CSR<half, tag>(A);
  auto inner = Solver<FGMRES, half>(A16, M, {64, true});
  auto solver_ = Solver<FGMRES, double>(A, inner, {100, false});
  auto solver = [&solver_, &restart](auto b, auto x, auto cnv) {
    auto res = solver_.solve(b, x, cnv);
    while (!res.is_solved && res.res_iter < 100 * restart)
      res += solver_.solve(b, x, cnv);
    return res;
  };
  const std::string solver_name = "F2H";
#endif

  std::cout << name << "," << solver_name << "," << type_name << "," << m2
            << "," << m3 << "," << m4 << "," << c << "," << precond;

#if defined(LOGGING)
  std::cout << std::endl;
#endif

  auto t = timer();
  int itr_sum = 0;
  for (int i = 0; i < suite_iter; i++) {
    (i == suite_iter - 1) ? test(solver, b, t, true, itr_sum, 64)
                          : test(solver, b, t, false, itr_sum, 64);
  }

  return 0;
}
