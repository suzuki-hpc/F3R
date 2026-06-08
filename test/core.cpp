#include "doctest/doctest.h"

#include "senk/core/io.hpp"
#include "senk/core/kernel.hpp"
#include "senk/core/memory.hpp"
#include "senk/core/timer.hpp"

#include <thread>

TEST_CASE("Core") {
  SUBCASE("timer") {
    auto t = senk::timer();
    t.tick();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    t.tock();
    CHECK(t.durations[0].count() >= 0.99);
    CHECK(t.durations[0].count() <= 1.01);
  }
  SUBCASE("memory") {
    using tag = senk::host;
    auto a = senk::memory<tag>::alloc<double>(10);
    auto b = senk::memory<tag>::alloc<double>(10);
    double l = 1.0, r = -1.;
    a[0] = l;
    a[9] = r;
    CHECK(a[0] == l);
    CHECK(a[9] == r);
    // senk::communicator<tag, tag>::to(b, a, 10);
    senk::memcpy<tag, tag>(b, a, 10);
    senk::memory<tag>::free(a);
    CHECK(b[0] == l);
    CHECK(b[9] == r);
    senk::memory<tag>::free(b);
  }
  SUBCASE("kernel") {
    using tag = senk::host;
    size_t n = 1000;
    auto a = senk::memory<tag>::alloc<double>(n);
    auto b = senk::memory<tag>::alloc<double>(n);

    senk::kernel<tag>::single([=]() {
      for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = 1.;
      }
    });
    senk::kernel<tag>::parallel(n, [=](size_t i) { a[i] += b[i]; });
    // senk::kernel<tag>::parallel<1>({n}, [=](size_t i) { a[i] += b[i]; });

    CHECK(a[0] == 1.);
    CHECK(a[n - 1] == n);
    double res;
    senk::kernel<tag>::reduce_add(n, &res, [=](size_t i) { return a[i]; });
    CHECK(res == (1 + n) * n / 2);

    senk::memory<tag>::free(a);
    senk::memory<tag>::free(b);
  }
}

TEST_CASE("IO") {
  auto path = std::string("../data/cage5.mtx");
  auto A = senk::io::readmm_as_csr<double>(path);

  auto M = senk::io::readmm_as_dense<double>("../data/test.mtx");
  double res;
  senk::kernel<senk::host>::reduce_add(
      M.nrows * M.ncols, &res, [=](size_t i) { return M.val[i]; });
  CHECK(res == 45 + 55);

  delete[] A.val;
  delete[] A.rptr;
  delete[] A.col;
  delete[] M.val;
}