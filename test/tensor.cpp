#include "doctest/doctest.h"

#include "senk/core/tensor.hpp"

namespace senk {

template <typename T, class RHS>
bool operator==(const scalar<T, host> &lhs, const RHS &rhs) {
  return lhs[0] == rhs;
}

template <typename T>
bool operator==(const scalar<T, host> &lhs, const scalar<T, host> &rhs) {
  return lhs[0] == rhs[0];
}

template <typename T, class RHS>
bool operator!=(const scalar<T, host> &lhs, const RHS &rhs) {
  return lhs[0] != rhs;
}

template <typename T>
inline std::ostream &operator<<(
    std::ostream &os, const senk::scalar<T, host> &h) {
  return os << static_cast<double>(h[0]);
}

} // namespace senk

TEST_CASE_TEMPLATE("scalar", T, double, float, int, senk::half) {
  using namespace senk;
  SUBCASE("assign") {
    scalar<T, host> a, b;
    a = 1.1, a += 1.1, a -= 1.1, a *= 1, a /= 1;
    CHECK(a == static_cast<T>(1.1));
    a = 1.0;
    b = 1.0;
    b += a;
    b -= a;
    b *= a;
    b /= a;
    CHECK(b == static_cast<T>(1.0));
    b = a + a, b += b + a, b -= a + a;
    CHECK(b == static_cast<T>(1.0) + static_cast<T>(1.0) + static_cast<T>(1.0));
  }
  scalar<T, host> a, b, c;
  a = 1.;
  b.fill(2.);
  c.copy(a);
  c = -c;
  c.abs();
  c += a + b - c * a;
  CHECK(c == 3.);
  c.inv().sqrt().neg();
  CHECK(c == T(-senk::sqrt(1. / 3.)));

  scalar<double, host> da(a);
  CHECK(da == 1.);
  if constexpr (std::is_same_v<double, T>) {
    CHECK(da.raw() == a.raw());
  }
  b.copy(a);
  CHECK(b.raw() != a.raw());
  CHECK(b == a);
  b = a;
  CHECK(b.raw() == a.raw());
}

TEST_CASE_TEMPLATE("vector", T, double, float) {
  using namespace senk;

  for (int N : {2048, 2 << 16, 1'000'000}) {
    CAPTURE(N);
    auto a = vector<T, host>(N).iota(0);
    auto b = vector<T, host>(N).random(std::mt19937_64(0), -1, 1);
    auto c = vector<T, host>(N);
    c = 1.;
    a = a * c;
    c += a + b;
    CHECK(c(0) == b(0)[0] + T(1));
    CHECK(c(N - 1) == b(N - 1)[0] + T(N));
    c.fill(-2.0);
    c.abs().inv();
    CHECK(c(0) == T(1 / 2.));
    CHECK(c(N - 1) == T(1 / 2.));

    c = a.apply([](auto item) { return item * item; });
    CHECK(c(0) == T(0));
    CHECK(c(N - 1) == T(N - 1) * T(N - 1));

    auto reduce = reducer<host>(N);
    scalar<T, host> res;
    res = reduce.norm(c);
    c /= res;

    double eps = (std::is_same_v<T, float>) ? 1e-3 : 1e-12;

    res = reduce.dot(c, c);
    CHECK(res[0] == doctest::Approx(1.).epsilon(eps));
    res = reduce.dot(c, c + c);
    CHECK(res[0] == doctest::Approx(2.).epsilon(eps));
    res = reduce.dot(c + c, c);
    CHECK(res[0] == doctest::Approx(2.).epsilon(eps));
    res = reduce.dot(c + c, c + c);
    CHECK(res[0] == doctest::Approx(4.).epsilon(eps));

    CHECK(c.raw() != a.raw());
    c = a;
    CHECK(c.raw() == a.raw());
  }
}

TEST_CASE_TEMPLATE("matrix", T, double) {
  using namespace senk;
  auto N = 1'048'576;
  auto A = matrix<T, host>(N, 3);

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
}

TEST_CASE("complex") {
  using namespace senk;
  scalar<complex<float>, host> a, b, c;
  a = complex<float>(1, 1);
  b = complex<float>(2, 1);
  c = a + b;
  CHECK(c[0] == complex<float>(3, 2));
  c = a * b;
  CHECK(c[0] == complex<float>(1, 3));
  // c = a.conj() * b;
  c = a.apply([](auto item) { return conj(item); }) * b;
  CHECK(c[0] == complex<float>(3, -1));

  // c.inv().sqrt().neg();
  // CHECK(c == T(-std::sqrt(1. / 3.)));
}