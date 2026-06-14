#include "doctest/doctest.h"

#include "senk/core/io.hpp"
#include "senk/core/kernel.hpp"
#include "senk/core/memory.hpp"
#include "senk/core/timer.hpp"

#include <thread>

TEST_CASE("Core") {
  SUBCASE("kernel") {
    using namespace senk;
    size_t n = 1000;
    auto a = senk::memory<device>::alloc<double>(n);
    auto b = senk::memory<device>::alloc<double>(n);

    kernel<device>::parallel(n, [=] SENK_LOC(size_t i) {
      a[i] = i;
      b[i] = 1.;
    });
    kernel<device>::parallel<1>({n}, [=] SENK_LOC(size_t i) { a[i] += b[i]; });

    kernel<device>::parallel<2>({100, 10},
        [=] SENK_LOC(size_t i, size_t j) { a[i + j * 100] = i + j; });

    auto ha = senk::memory<host>::alloc<double>(n);
    // senk::communicator<device, host>::to(ha, a, n);
    senk::copy<host, device>(ha, a, n);

    // senk::kernel<tag>::parallel<1>({n}, [=](size_t i) { a[i] += b[i]; });

    // CHECK(a[0] == 1.);
    // CHECK(a[n - 1] == n);
    // double res;
    // kernel<device>::reduce_add(n, &res, [=](size_t i) { return a[i]; });
    // CHECK(res == (1 + n) * n / 2);

    // senk::memory<device>::free(a);
    // senk::memory<device>::free(b);
  }
  SUBCASE("copy") {
    using namespace senk;
    size_t n = 1000;
    auto a = senk::memory<device>::alloc<double>(n);
    auto b = senk::memory<device>::alloc<int>(n);
    auto c = senk::memory<host>::alloc<double>(n);
    auto d = senk::memory<host>::alloc<int>(n);

    auto ha = senk::memory<host>::alloc<double>(n);
    auto ha2 = senk::memory<host>::alloc<double>(n);

    kernel<device>::parallel(n, [=] SENK_LOC(size_t i) { a[i] = i / 100.; });
    senk::copy<host, device>(ha2, a, n);
    copy<device, device>(b, a, n);
    copy<host, device>(c, b, n);
    copy<host, host>(d, c, n);
    copy<device, host>(a, d, n);

    senk::copy<host, device>(ha, a, n);
    // for (int i = 0; i < n; i++)
    // printf("%d %e %e\n", i, ha[i], ha2[i]);
  }
}
