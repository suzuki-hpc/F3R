#include "doctest/doctest.h"

#include "senk/core/tensor.hpp"

namespace senk {

template <typename T, class RHS>
bool operator==(const scalar<T, device> &lhs, const RHS &rhs) {
  scalar<T, host> res(lhs);
  return double(res[0]) == double(rhs);
}

template <typename T>
bool operator==(const scalar<T, device> &lhs, const scalar<T, device> &rhs) {
  scalar<T, host> res1(lhs), res2(rhs);
  return double(res1[0]) == double(res2[0]);
}

template <typename T, class RHS>
bool operator!=(const scalar<T, device> &lhs, const RHS &rhs) {
  return !operator==(lhs, rhs);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const scalar<T, device> &h) {
  scalar<T, host> res(h);
  return os << double(res[0]);
}

} // namespace senk

TEST_CASE_TEMPLATE("scalar", T, double, float, int, senk::half) {
  using namespace senk;
  SUBCASE("assign") {
    scalar<T, device> a, b;
    a = 1.1, a += 1.1, a -= 1.1, a *= 1, a /= 1;
    CHECK(a == static_cast<T>(1.1));
    a = 1.0;
    b = 1.0;
    b += a, b -= a, b *= a, b /= a;
    CHECK(b == static_cast<T>(1.0));
    b = a + a, b += b + a, b -= a + a;
    CHECK(b == static_cast<T>(1.0) + static_cast<T>(1.0) + static_cast<T>(1.0));
  }

  scalar<T, device> a, b, c;
  a = 1.;

  b.fill(2.);
  c.copy(a);
  c = -c;
  c.abs();
  c += a + b - c * a;
  CHECK(c == 3.);
  c.inv().sqrt().neg();
  CHECK(c == T(-std::sqrt(1. / 3.)));

  scalar<double, device> da(a);
  CHECK(da == 1.);
  if constexpr (std::is_same_v<double, T>) {
    CHECK(da.raw() == a.raw());
  }

  b = a;
  CHECK(b.raw() == a.raw());
}

TEST_CASE("scalar2") {
  using namespace senk;
  scalar<double, device> a;
  scalar<float, device> b;
  scalar<int, device> c;
  a = 1.143;
  b.copy(a);
  c.copy(b);

  scalar<double, host> d;
  d.copy(c);
  CHECK(d[0] == int(float(1.143)));
}

TEST_CASE_TEMPLATE("vector", T, double, float) {
  using namespace senk;

  for (int N : {2048, 2 << 16, 1'000'000}) {
    CAPTURE(N);
    auto a = vector<T, device>(N).iota(0);
    auto b = vector<T, device>(N).random(std::mt19937_64(0), -1, 1);
    auto c = vector<T, device>(N);
    scalar<T, host> hc, hb;
    c = 1.;
    c += a + b;
    CHECK(hc.copy(c(0))[0] == hb.copy(b(0))[0] + T(1));
    CHECK(hc.copy(c(N - 1))[0] == hb.copy(b(N - 1))[0] + T(N));
    c.fill(-2.0);
    c.abs().inv();
    CHECK(c(0) == T(1 / 2.));
    CHECK(c(N - 1) == T(1 / 2.));

    c = a.apply([] SENK_LOC(T item) { return item * item; });
    CHECK(c(0) == T(0));
    CHECK(c(N - 1) == T(N - 1) * T(N - 1));

    auto reduce = reducer<device>(N);
    scalar<T, device> res;
    res = reduce.norm(c);
    c /= res;
    c = c * res;
    c = c / res;

    double eps = (std::is_same_v<T, float>) ? 1e-3 : 1e-12;

    scalar<T, host> hres;
    hres = reduce.dot(c, c);
    CHECK(hres[0] == doctest::Approx(1.).epsilon(eps));
    hres = reduce.dot(c, c + c);
    CHECK(hres[0] == doctest::Approx(2.).epsilon(eps));
    hres = reduce.dot(c + c, c);
    CHECK(hres[0] == doctest::Approx(2.).epsilon(eps));
    hres = reduce.dot(c + c, c + c);
    CHECK(hres[0] == doctest::Approx(4.).epsilon(eps));

    CHECK(c.raw() != a.raw());
    c = a;
    CHECK(c.raw() == a.raw());
  }
}

TEST_CASE("vector2") {
  using namespace senk;
  int N = 10000;
  auto a = vector<double, device>(N).random(std::mt19937_64(0), -1, 1);
  auto b1 = vector<float, host>(N).copy(a);
  auto _b2 = vector<double, host>(N).copy(a);
  auto b2 = vector<float, host>(N).copy(_b2);
  scalar<double, host> bb1, bb2;
  auto reduce = reducer<host>(N);

  bb1 = reduce.norm(b1.as<double>());
  bb2 = reduce.norm(b2.as<double>());

  CHECK(bb1[0] == bb2[0]);
}

TEST_CASE_TEMPLATE("matrix", T, double) {
  using namespace senk;
  auto N = 1'048'576;
  auto A = matrix<T, device>(N, 3);

  auto a1 = A(0).fill(1);
  auto a2 = A(1).fill(1);
  auto a3 = A(2).fill(1);
  a1 *= 1.;
  a2 *= 2.;
  a3 *= 3.;
  CHECK(a1(0) == 1.);
  CHECK(a2(0) == 2.);
  CHECK(a3(0) == 3.);
  a3 += a2 - a1;
  CHECK(a3(0) == 4.);

  auto reduce = reducer<device>(N);
  scalar<T, device> res;
  scalar<T, host> hres;

  hres = reduce.norm(A(0));
  CHECK(hres[0] == std::sqrt(N));
}

TEST_CASE("complex") {
  using namespace senk;
  scalar<complex<float>, device> a, b, c;
  scalar<complex<float>, host> hres;
  a = complex<float>(1, 1);
  b = complex<float>(2, 1);
  c = a + b;
  CHECK(hres.copy(c)[0] == complex<float>(3, 2));
  c = a * b;
  CHECK(hres.copy(c)[0] == complex<float>(1, 3));
  // c = a.conj() * b;
  c = a.apply([] SENK_LOC(const complex<float> &item) {
    return senk::conj(item);
  }) * b;
  CHECK(hres.copy(c)[0] == complex<float>(3, -1));

  c.inv().neg();
  CHECK(hres.copy(c)[0] == -complex<float>(3, 1) / complex<float>(10));
}