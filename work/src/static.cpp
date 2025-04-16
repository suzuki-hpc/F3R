#include "umineko-core/timer.hpp"

#include "umineko-sparse/matrix/io.hpp"
#include "umineko-sparse/matrix/csr.hpp"

#include "umineko-sparse/solver/bicgstab.hpp"
#include "umineko-sparse/solver/cg.hpp"
#include "umineko-sparse/solver/gmres.hpp"
#include "umineko-sparse/solver/richardson.hpp"
#include "umineko-sparse/solver/restart.hpp"

#include "umineko-sparse/preconditioner/ilu.hpp"
#include "umineko-sparse/preconditioner/ainv.hpp"
#include "umineko-sparse/trsv/bj.hpp"

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
  double weight = atof(argv[9]);

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


  impl::pool<_Float16, tag>::init(A.nrows());  
  auto A16 = CSR<_Float16, tag>(A);
  auto L = trsv::l::BJ<CSR<_Float16, host>>(l, {112});
  auto U = trsv::du::BJ<CSR<_Float16, host>>(u, {112});
  auto tmp = vector<_Float16, tag>(data.nrows());
  auto tmp2 = vector<_Float16, tag>(data.nrows());
  auto hin = vector<_Float16, tag>(data.nrows());
  auto hout = vector<_Float16, tag>(data.nrows());
  auto hr = vector<_Float16, tag>(data.nrows());
  scalar<_Float16, tag> ome;
  ome.fill(_Float16(weight));

  auto R = Operator<tag>(data.get_shape(), [&](auto in, auto out) {
    hin.copy(in);
    L.solve(hin, tmp);
    U.solve(tmp, hout);
    hout *= ome;
    for (int k = 1; k < m4; k++) {
      A16.compute_residual(hin, hout, hr);
      L.solve(hr, tmp);
      U.solve(tmp, tmp2);
      hout += ome * tmp2;
    }
    out.copy(hout);
  });
  auto A32 = CSR<float, tag>(A);
  auto MM = FGMRES<float, tag>(A16, R, {m3, false});
  auto MMM = FGMRES<float, tag>(A32, MM, {m2, false});
  auto inner = Solver<tag>(FGMRES<double, tag>(A, MMM, {100, true}));
  auto solver = Restarted<double, tag>(A, inner, {restart});

  std::cout << name << "," 
            << "F3R-Static,"
            << type_name << ","
            << m2 << "," << m3 << "," << m4 << "," << weight << ","
            << precond;

  auto t = timer();
  int itr_sum = 0;
  for(int i=0; i<suite_iter; i++) {
    (i==suite_iter - 1)?
      test(solver, b, t, true, itr_sum, m2*m3*m4) :
      test(solver, b, t, false, itr_sum, m2*m3*m4);
  }

  return 0;
}
