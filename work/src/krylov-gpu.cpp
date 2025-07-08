#include "umineko-core/timer.hpp"

#include "umineko-sparse/matrix/io.hpp"
#include "umineko-sparse/matrix/csr.hpp"
#include "umineko-sparse/matrix/sell.hpp"
#include "umineko-sparse/matrix/algorithm.hpp"
#include "umineko-sparse/solver/cg.hpp"
#include "umineko-sparse/solver/bicgstab.hpp"
#include "umineko-sparse/solver/gmres.hpp"
#include "umineko-sparse/solver/restart.hpp"
#include "umineko-sparse/preconditioner/ainv.hpp"

#include <cuda_fp16.h>

using namespace kmm;

const double eps = 1.e-8;

#define _STR(x) #x
#define STR(x) _STR(x)

const std::string type_name = STR(TYPE);
const std::string precond_name = "SDAINV";
using precond_type = TYPE;

__host__ __device__ inline float operator*(const float &lh, const __half &rh) {
  return lh * static_cast<float>(rh);
};

__host__ __device__ inline double operator*(const double &lh, const __half &rh) {
  return lh * static_cast<double>(rh);
};

__host__ __device__ inline float operator*(const __half &lh, const float &rh) {
  return static_cast<float>(lh) * rh;
};

__host__ __device__ inline double operator*(const __half &lh, const double &rh) {
  return static_cast<double>(lh) * rh;
};

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

  auto [coo, rhs] = io::mm::read_system(path, [](vector<double, host> v) { v.random<0, 1>(std::mt19937_64(0)); });
  auto data = CSR<double, host>(io::CSR(coo));

  using tag = device;

  algorithm::scaling(data);

  auto A = SELL<32, double, tag>(data);
  auto b = vector<double, tag>(rhs);
  auto x = vector<double, tag>(A.nrows());
  auto r = vector<double, tag>(A.nrows());

  impl::pool<double, tag>::init(A.nrows());

  auto [z, w] = algorithm::sdainv(data, 0.1, acc);
  auto na = CSR<double, host>({1,1},1);

  auto test = [&A, &x, &r, &suite_iter](
    auto solver, auto b, auto &t, bool ff, int &itr_sum, int ww=1)
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

    if(ff | !flag.is_solved) {
      double sum = 0;
      for (const auto &d : t.durations)
        sum += d.count();
      if (!flag.is_solved) {
        printf("%e,", sum);
        printf("%d,%e,", itr_sum*ww, flag.res_nrm2 / nrm_b[0]);
      } else {
        printf("%e,", sum/suite_iter);
        printf("%d,%e,", itr_sum*ww/suite_iter, flag.res_nrm2 / nrm_b[0]);
      }
      A.operate(x, r);
      r = b - r;
      nrm_r = nrm(r);
      printf("%e\n", nrm_r[0] / nrm_b[0]);
    }

    return flag.is_solved;
  };

  auto W = SELL<32, precond_type, tag>(w);
  auto Z = SELL<32, precond_type, tag>(z);
  w = na;
  z = na;
  auto tmp = vector<double, tag>(A.nrows());
  auto M = Operator<tag>(A.get_shape(), [&W, &Z, &tmp](auto in, auto out) {
    W.operate(in, tmp);
    Z.operate(tmp, out);
  });

#if defined(SOLV_BiCG)
  const std::string solver_name = "BiCGStab";
  auto solver = Solver<tag>(BiCGSTAB<double, tag>(A, M, {19200}));
  int ww = 2;
#endif

#if defined(SOLV_CG)
  const std::string solver_name = "CG";
  auto solver = Solver<tag>(CG<double, tag>(A, M, {19200}));
  int ww = 1;
#endif

#if defined(SOLV_GM)
  const std::string solver_name = "GMRES";
  auto gmres = Solver<tag>(FGMRES<double, tag>(A, M, {64, true}));
  auto solver = Restarted<double, tag>(A, gmres, {300});
  int ww = 1;
#endif

  std::cout << name << "," 
            << solver_name << ","
            << type_name << ","
            << m2 << "," << m3 << "," << m4 << "," << c << ","
            << precond;

  auto t = timer();
  int itr_sum = 0;
  for(int i=0; i<suite_iter; i++) {
    bool is_solved = (i==suite_iter - 1)?
      test(solver, b, t, true, itr_sum, ww) :
      test(solver, b, t, false, itr_sum, ww);
    if (!is_solved)
      break;
  }

  return 0;
}
