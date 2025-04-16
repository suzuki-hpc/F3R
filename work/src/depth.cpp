#include "umineko-core/timer.hpp"

#include "umineko-sparse/models2.hpp"

#include "umineko-sparse/matrix/io.hpp"
#include "umineko-sparse/matrix/csr.hpp"

#include "umineko-sparse/solver/bicgstab.hpp"
#include "umineko-sparse/solver/cg.hpp"
// #include "umineko-sparse/solver/gmres.hpp"
#include "umineko-sparse/solver/richardson.hpp"
#include "umineko-sparse/solver/restart.hpp"

#include "umineko-sparse/preconditioner/ilu.hpp"
#include "umineko-sparse/preconditioner/ainv.hpp"
#include "umineko-sparse/trsv/bj.hpp"

template <typename T>
T abs(const T& val) {
  if constexpr(std::is_same_v<T, _Float16>)
    return _Float16(std::abs(float(val)));
  else
    return std::abs(val);
}

template <typename T>
T sqrt(const T& val) {
  if constexpr(std::is_same_v<T, _Float16>)
    return _Float16(std::sqrt(float(val)));
  else
    return std::sqrt(val);
}
#include "umineko-sparse/solver/gmres2.hpp"

using namespace kmm;

const double eps = 1.e-8;

#define _STR(x) #x
#define STR(x) _STR(x)

const std::string type_name = STR(TYPE);
const std::string precond_name = "BJILU0";
using precond_type = TYPE;

int main(int argc, char *argv[]) {
  std::string path = std::string("../matrix/") + argv[1];
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

  auto [coo, rhs] = io::mm::read_system(path, [](vector<double, host> v) { v.random<0, 1>(std::mt19937_64(0)); });
  auto data = CSR<double, host>(io::CSR(coo));

  using tag = host;

  algorithm::scaling(data);

  auto A = CSR<double, tag>(data);
  auto b = vector<double, tag>(rhs);
  auto x = vector<double, tag>(A.nrows());
  auto r = vector<double, tag>(A.nrows());

  impl::pool<double, tag>::init(A.nrows());

  auto bd = data.duplicate_block_diagonal(112, 1);
  auto [l, u] = algorithm::split(algorithm::ilup(bd, 0, acc, 112, 1));

  auto test = [&A, &x, &r, &suite_iter](
    auto solver, auto b, auto &t, bool ff, int &itr_sum, int ww)
  {
    x.fill(0.0);
    scalar<double, host> nrm_b, nrm_r;
    nrm_b = nrm(b);
    auto cond = [=]([[maybe_unused]]int i, double e) {
      return e / nrm_b[0] < eps;
    };
    t.tick();
    auto flag = solver.solve(b, x, cond);
    t.tock();

    itr_sum += flag.res_iter;

    if(ff) {
      double sum = 0;
      for (const auto &d : t.durations)
        sum += d.count();
      printf("%e,", sum/suite_iter);
      printf("%d,%e,", itr_sum*ww/suite_iter, flag.res_nrm2 / nrm_b[0]);
      A.operate(x, r);
      r = b - r;
      nrm_r = nrm(r);
      printf("%e\n", nrm_r[0] / nrm_b[0]);
    }
  };

  auto L = trsv::l::BJ<CSR<_Float16, host>>(l, {112});
  auto U = trsv::du::BJ<CSR<_Float16, host>>(u, {112});
  auto tmp = vector<_Float16, tag>(data.nrows());
  auto M = Operator<tag>(data.get_shape(), [&](auto in, auto out) {
    L.solve(in, tmp);
    U.solve(tmp, out);
  });

#if defined(F4)
  auto A16 = CSR<_Float16, tag>(A);
  auto A32 = CSR<float, tag>(A);
  auto FF = FGMRES<_Float16, tag>(A16, M, {2, false});
  auto F = FGMRES<float, tag>(A16, FF, {4, false});
  auto inner = FGMRES<double, tag>(A32, F, {8, false});
  auto solver_ = Solver<tag>(FGMRES<double, tag>(A, inner, {100, true}));
  auto solver = Restarted<double, tag>(A, solver_, {restart});
  const std::string solver_name = "F4";
#endif

#if defined(F3)
  auto A16 = CSR<_Float16, tag>(A);
  auto A32 = CSR<float, tag>(A);
  auto F = FGMRES<float, tag>(A16, M, {8, false});
  auto inner = FGMRES<double, tag>(A32, F, {8, false});
  auto solver_ = Solver<tag>(FGMRES<double, tag>(A, inner, {100, true}));
  auto solver = Restarted<double, tag>(A, solver_, {restart});
  const std::string solver_name = "F3";
#endif

#if defined(F3H)
  auto A16 = CSR<_Float16, tag>(A);
  auto A32 = CSR<float, tag>(A);
  auto F = FGMRES<_Float16, tag>(A16, M, {8, false});
  auto inner = FGMRES<double, tag>(A32, F, {8, false});
  auto solver_ = Solver<tag>(FGMRES<double, tag>(A, inner, {100, true}));
  auto solver = Restarted<double, tag>(A, solver_, {restart});
  const std::string solver_name = "F3H";
#endif

#if defined(F2)
  auto A32 = CSR<float, tag>(A);
  auto inner = FGMRES<double, tag>(A32, M, {16, false});
  auto solver_ = Solver<tag>(FGMRES<double, tag>(A, inner, {100, true}));
  auto solver = Restarted<double, tag>(A, solver_, {restart});
  const std::string solver_name = "F2";
#endif

#if defined(F2H)
  auto A16 = CSR<_Float16, tag>(A);
  auto inner = FGMRES<_Float16, tag>(A16, M, {8, false});
  auto solver_ = Solver<tag>(FGMRES<double, tag>(A, inner, {100, true}));
  auto solver = Restarted<double, tag>(A, solver_, {restart});
  const std::string solver_name = "F2H";
#endif

  std::cout << name << "," 
            << solver_name << ","
            << type_name << ","
            << m2 << "," << m3 << "," << m4 << "," << c << ","
            << precond;

  auto t = timer();
  int itr_sum = 0;
  for(int i=0; i<suite_iter; i++) {
    (i==suite_iter - 1)?
      test(solver, b, t, true, itr_sum, 64) :
      test(solver, b, t, false, itr_sum, 64);
  }

  return 0;
}
