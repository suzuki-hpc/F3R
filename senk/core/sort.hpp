#ifndef SENK_CORE_SORT_HPP
#define SENK_CORE_SORT_HPP

#include <cstddef>
#include <utility>

namespace senk::sort {

enum class order { asc, desc };

inline void pack_swap(size_t, size_t) {}
template <typename Head, typename... Tail>
void pack_swap(size_t left, size_t right, Head list, Tail... tail) {
  std::swap(list[left], list[right]);
  pack_swap(left, right, std::forward<Tail>(tail)...);
}

template <order order, typename T, typename... Args>
void pack_sort(size_t left, size_t right, T *key, Args... args) {
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
    pack_sort<order>(left, L, key, args...);
  if (R + 2 < right)
    pack_sort<order>(R + 1, right, key, args...);
}

} // namespace senk::sort

#endif // SENK_CORE_SORT_HPP