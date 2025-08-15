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
#if defined(LOGGING)
      printf("%d %e\n", i, e);
#endif
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
#if !defined(LOGGING)
      printf("%e,", sum/suite_iter);
      printf("%d,%e,", itr_sum*ww/suite_iter, flag.res_nrm2 / nrm_b[0]);
#endif
      A.operate(x, r);
      r = b - r;
      nrm_r = nrm(r);
#if !defined(LOGGING)
      printf("%e\n", nrm_r[0] / nrm_b[0]);
#endif
    }
  };

  const int period = c;
  int cnt = c;
  scalar<int, tag> den;

#if defined(DOUBLE)
  scalar<double, tag> dot_ar, dot_r;
  auto L = trsv::l::BJ<CSR<double, host>>(l, {112});
  auto U = trsv::du::BJ<CSR<double, host>>(u, {112});
  auto tmp = vector<double, tag>(data.nrows());
  auto tmp2 = vector<double, tag>(data.nrows());
  auto hr = vector<double, tag>(data.nrows());
  std::vector<scalar<double, tag>> sum;
  std::vector<scalar<double, tag>> ome;
  sum.resize(m4);
  ome.resize(m4);
  auto R = Operator<tag>(data.get_shape(), [&](auto in, auto out) {
    ++cnt;
    L.solve(in, tmp);
    U.solve(tmp, out);
    if (cnt % period == 0) {
      A.operate(out, tmp);
      dot_ar = dot(tmp, tmp);
      dot_r = dot(tmp, in);
      out *= dot_r / dot_ar;
      sum[0] += dot_r / dot_ar;
      ome[0] = sum[0] / (den = cnt / period);
    } else {
      out *= ome[0];
    }
    for (int k = 1; k < m4; k++) {
      A.compute_residual(in, out, hr);
      L.solve(hr, tmp);
      U.solve(tmp, tmp2);
      if (cnt % period == 0) {
        A.operate(tmp2, tmp);
        dot_ar = dot(tmp, tmp);
        dot_r = dot(tmp, hr);
        out += dot_r / dot_ar * tmp2;
        sum[k] += dot_r / dot_ar;
        ome[k] = sum[k] / den;
      } else {
        out += ome[k] * tmp2;
      }
    }
  });
  auto MM = FGMRES<double, tag>(A, R, {m3, false});
  auto MMM = FGMRES<double, tag>(A, MM, {m2, false});
  auto inner = Solver<tag>(FGMRES<double, tag>(A, MMM, {100, true}));
  auto solver = Restarted<double, tag>(A, inner, {restart});
#endif

#if defined(FLOAT)
  impl::pool<float, tag>::init(A.nrows());
  auto A32 = CSR<float, tag>(A);
  scalar<float, tag> dot_ar, dot_r;
  auto L = trsv::l::BJ<CSR<float, host>>(l, {112});
  auto U = trsv::du::BJ<CSR<float, host>>(u, {112});
  auto tmp = vector<float, tag>(data.nrows());
  auto tmp2 = vector<float, tag>(data.nrows());
  auto hr = vector<float, tag>(data.nrows());
  std::vector<scalar<float, tag>> sum;
  std::vector<scalar<float, tag>> ome;
  sum.resize(m4);
  ome.resize(m4);
  auto R = Operator<tag>(data.get_shape(), [&](auto in, auto out) {
    ++cnt;
    L.solve(in, tmp);
    U.solve(tmp, out);
    if (cnt % period == 0) {
      A.operate(out, tmp);
      dot_ar = dot(tmp, tmp);
      dot_r = dot(tmp, in);
      out *= dot_r / dot_ar;
      sum[0] += dot_r / dot_ar;
      ome[0] = sum[0] / (den = cnt / period);
    } else {
      out *= ome[0];
    }
    for (int k = 1; k < m4; k++) {
      A.compute_residual(in, out, hr);
      L.solve(hr, tmp);
      U.solve(tmp, tmp2);
      if (cnt % period == 0) {
        A.operate(tmp2, tmp);
        dot_ar = dot(tmp, tmp);
        dot_r = dot(tmp, hr);
        out += dot_r / dot_ar * tmp2;
        sum[k] += dot_r / dot_ar;
        ome[k] = sum[k] / den;
      } else {
        out += ome[k] * tmp2;
      }
    }
  });
  auto MM = FGMRES<float, tag>(A32, R, {m3, false});
  auto MMM = FGMRES<float, tag>(A32, MM, {m2, false});
  auto inner = Solver<tag>(FGMRES<double, tag>(A, MMM, {100, true}));
  auto solver = Restarted<double, tag>(A, inner, {restart});
#endif

#if defined(HALF)
  impl::pool<_Float16, tag>::init(A.nrows());
  
  scalar<float, tag> dot_ar, dot_r;
  auto A16 = CSR<_Float16, tag>(A);
  auto L = trsv::l::BJ<CSR<_Float16, host>>(l, {112});
  auto U = trsv::du::BJ<CSR<_Float16, host>>(u, {112});
  auto tmp = vector<_Float16, tag>(data.nrows());
  auto tmp2 = vector<_Float16, tag>(data.nrows());
  auto hin = vector<_Float16, tag>(data.nrows());
  auto hout = vector<_Float16, tag>(data.nrows());
  auto hr = vector<_Float16, tag>(data.nrows());
  std::vector<scalar<float, tag>> sum;
  std::vector<scalar<_Float16, tag>> ome;
  sum.resize(m4);
  ome.resize(m4);
  auto tof32 = [](auto v) {return static_cast<float>(v);};
  auto R = Operator<tag>(data.get_shape(), [&](auto in, auto out) {
    ++cnt;
    hin.copy(in);
    L.solve(hin, tmp);
    U.solve(tmp, hout);
    if (cnt % period == 0) {
      A16.operate(hout, tmp);
      dot_ar = dot(wildcard(tmp,tof32), wildcard(tmp,tof32));
      dot_r = dot(wildcard(tmp,tof32), wildcard(hin,tof32));
      hout *= dot_r / dot_ar;
      sum[0] += dot_r / dot_ar;
      ome[0] = sum[0] / (den = cnt / period);
    } else {
      hout *= ome[0];
    }
    for (int k = 1; k < m4; k++) {
      A16.compute_residual(hin, hout, hr);
      L.solve(hr, tmp);
      U.solve(tmp, tmp2);
      if (cnt % period == 0) {
        A16.operate(tmp2, tmp);
        dot_ar = dot(wildcard(tmp,tof32), wildcard(tmp,tof32));
        dot_r = dot(wildcard(tmp,tof32), wildcard(hr,tof32));
        hout += dot_r / dot_ar * tmp2;
        sum[k] += dot_r / dot_ar;
        ome[k] = sum[k] / den;
      } else {
        hout += ome[k] * tmp2;
      }
    }
    out.copy(hout);
  });
  auto A32 = CSR<float, tag>(A);
  auto MM = FGMRES<float, tag>(A16, R, {m3, false});
  auto MMM = FGMRES<float, tag>(A32, MM, {m2, false});
  auto inner = Solver<tag>(FGMRES<double, tag>(A, MMM, {100, true}));
  auto solver = Restarted<double, tag>(A, inner, {restart});
#endif

  std::cout << name << "," 
            << "F3R,"
            << type_name << ","
            << m2 << "," << m3 << "," << m4 << "," << c << ","
            << precond;

#if defined(LOGGING)
  std::cout << std::endl;
#endif

  auto t = timer();
  int itr_sum = 0;
  for(int i=0; i<suite_iter; i++) {
    cnt = c;
    for(int k=0; k<m4; k++) {
      sum[k] = 1.0;
      ome[k] = 1.0;
    }
    (i==suite_iter - 1)?
      test(solver, b, t, true, itr_sum, m2*m3*m4) :
      test(solver, b, t, false, itr_sum, m2*m3*m4);
  }

  return 0;
}
