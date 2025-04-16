#ifndef UMINEKO_CORE_SORT_HPP
#define UMINEKO_CORE_SORT_HPP

#include <cstddef>
#include <utility>

namespace kmm::sort {

enum class order { asc, desc };

static void pack_swap(size_t, size_t) {}
template <typename Head, typename... Tail>
static void pack_swap(size_t left, size_t right, Head list, Tail... tail) {
  std::swap(list[left], list[right]);
  pack_swap(left, right, std::forward<Tail>(tail)...);
}

template <order order, typename T, typename... Args>
static void quick(size_t left, size_t right, T *key, Args... args) {
  if (left >= right)
    return;
  size_t L = left;
  size_t R = right - 1;
  T pivot = key[(L + R) / 2];
  while (true) {
    if (order == order::asc) {
      while (key[L] < pivot)
        L++;
      while (pivot < key[R])
        R--;
    } else {
      while (key[L] > pivot)
        L++;
      while (pivot > key[R])
        R--;
    }
    if (L >= R)
      break;
    pack_swap(L++, R--, key, args...);
  }
  if (left + 1 < L)
    quick<order>(left, L, key, args...);
  if (R + 2 < right)
    quick<order>(R + 1, right, key, args...);
}

} // namespace kmm::sort

#endif
