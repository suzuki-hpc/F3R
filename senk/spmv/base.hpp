#ifndef SENK_SPMV_BASE_HPP
#define SENK_SPMV_BASE_HPP

#include "senk/core/tools.hpp"

namespace senk {

namespace spmv {

struct algo_default {};
template <uint8_t _p>
struct algo_reduce {
  static_assert(
      (_p == 1 || _p == 2 || _p == 4 || _p == 8 || _p == 16 || _p == 32));
  static constexpr uint8_t p = _p;
};
struct algo_reduce_opt {
  const double score = 2.;
};

template <class>
struct is_algo_default : std::false_type {};
template <class T>
inline constexpr bool is_algo_default_v = is_algo_default<T>::value;
template <>
struct is_algo_default<algo_default> : std::true_type {};

template <class>
struct is_algo_reduce : std::false_type {};
template <class T>
inline constexpr bool is_algo_reduce_v = is_algo_reduce<T>::value;
template <uint8_t p>
struct is_algo_reduce<algo_reduce<p>> : std::true_type {};

template <class>
struct is_algo_reduce_opt : std::false_type {};
template <class T>
inline constexpr bool is_algo_reduce_opt_v = is_algo_reduce_opt<T>::value;
template <>
struct is_algo_reduce_opt<algo_reduce_opt> : std::true_type {};

} // namespace spmv

} // namespace senk

#endif