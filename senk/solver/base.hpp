#ifndef SENK_SOLVER_BASE_HPP
#define SENK_SOLVER_BASE_HPP

#include <cstdint>
#include <cstdio>

namespace senk {

namespace param {

struct kls {
  int32_t max_iter;
};

struct kls_normalize {
  int32_t max_iter;
  bool normalized;
};

struct kls_aug_normalize {
  int32_t max_iter, aug;
  bool normalized;
};

struct ss {
  int32_t max_iter;
  double weight;
};

struct ss_cycle {
  int32_t max_iter;
  double weight;
  int32_t cycle;
};

struct r {
  int32_t max_iter;
};

} // namespace param

template <template <typename, typename, typename> class S, typename T, class Op,
    class Pre>
auto Solver(
    const Op &op, const Pre &pre, const typename S<T, Op, Pre>::Params &prm) {
  return S<T, Op, Pre>(op, pre, prm);
}

template <template <typename, typename, typename, typename> class S, typename T,
    typename T2, class Op, class Pre>
auto Solver(const Op &op, const Pre &pre,
    const typename S<T, T2, Op, Pre>::Params &prm) {
  return S<T, T2, Op, Pre>(op, pre, prm);
}

template <template <typename, typename, typename, typename> class S, typename T,
    class Op, class Pre>
auto Solver(const Op &op, const Pre &pre,
    const typename S<T, T, Op, Pre>::Params &prm) {
  return S<T, T, Op, Pre>(op, pre, prm);
}

namespace converg {

inline auto rrn(double norm_b, double tol) {
  return [=]([[maybe_unused]] int i, double e) { return (e < norm_b * tol); };
}

inline auto rrn_log(double norm_b, double tol) {
  return [=]([[maybe_unused]] int i, double e) {
    printf("%d, %e\n", i, e / norm_b);
    return (e < norm_b * tol);
  };
}

} // namespace converg

} // namespace senk

#endif