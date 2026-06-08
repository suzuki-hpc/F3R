#ifndef SENK_MATRIX_BASE_HPP
#define SENK_MATRIX_BASE_HPP

#include <cstdint>

#include "senk/core/tensor.hpp"

namespace senk {

using index_t = int32_t;
// using serial_t = int64_t;
using serial_t = int32_t;

template <typename I>
struct has_idx_t {
  using idx_t = I;
};

template <typename I>
struct has_srl_t {
  using srl_t = I;
};

enum class algo {
  standard,
  col_reduce,
};

namespace impl {

enum class form { lower1, lower, upper1, upper };

enum flags : unsigned {
  has_complex_value = 1 << 0,
  is_symmetric = 1 << 1,
  is_square = 1 << 2,
  is_lower1 = 1 << 3,
  is_lower = 1 << 4,
  is_stlower = 1 << 5,
  is_upper1 = 1 << 6,
  is_upper = 1 << 7,
  is_stupper = 1 << 8,
  is_diagonal = 1 << 9,
  is_block_diagonal = 1 << 10,
  is_colored = 1 << 11,
  is_partitioned = 1 << 12,
};

} // namespace impl

struct attribute {
  uint32_t flags;
  vector<index_t, host> segm;

  attribute() : flags(0), segm(1) {}
  attribute(uint32_t flags, const vector<index_t, host> &segm)
      : flags(flags), segm(segm) {}

  attribute duplicate() const {
    auto _segm = vector<index_t, host>(segm.shape(0));
    auto attr = attribute(flags, _segm);
    for (size_t i = 0; i < segm.shape(0); i++)
      attr.segm[i] = segm[i];
    return attr;
  }

  attribute duplicate_rotate180() const {
    auto _segm = vector<index_t, host>(segm.shape(0));
    auto attr = attribute(flags, _segm);
    auto end = segm[segm.shape(0) - 1];
    for (size_t i = 0; i < segm.shape(0); i++)
      attr.segm[i] = end - segm[segm.shape(0) - 1 - i];
    return attr;
  }

  attribute &set_flag(impl::flags flag) {
    flags |= flag;
    return *this;
  }
  attribute &reset_flag(impl::flags flag) {
    flags &= ~flag;
    return *this;
  }
  attribute &set_ld(impl::flags ld) {
    flags &= ~impl::flags::is_lower1;
    flags &= ~impl::flags::is_lower;
    flags &= ~impl::flags::is_upper1;
    flags &= ~impl::flags::is_upper;
    flags |= ld;
    return *this;
  }

  [[nodiscard]] bool is_symmetric() const {
    return flags & impl::flags::is_symmetric;
  }
  [[nodiscard]] bool is_colored() const {
    return flags & impl::flags::is_colored;
  }
  [[nodiscard]] bool is_block_diagonal() const {
    return flags & impl::flags::is_block_diagonal;
  }
  [[nodiscard]] bool is_partitioned() const {
    return flags & impl::flags::is_partitioned;
  }
};

} // namespace senk

#endif // SENK_MATRIX_BASE_HPP