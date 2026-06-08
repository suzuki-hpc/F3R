#ifndef SENK_TRSV_BASE_HPP
#define SENK_TRSV_BASE_HPP

#include "senk/core/tools.hpp"

namespace senk {

namespace strategy {

struct direct {};
struct level {};
struct partition {};
struct jacobi {
  int32_t max_iter = 3;
};
struct baj {
  int32_t max_iter = 3;
};

} // namespace strategy

} // namespace senk

#endif