#ifndef SENK_CORE_CSV_HPP
#define SENK_CORE_CSV_HPP

#include <iostream>

namespace senk {

template <typename T>
void print_csv(std::ostream &os, const T &value) {
  os << value << std::endl;
}

template <typename T, typename... Args>
void print_csv(std::ostream &os, const T &first, const Args &...rest) {
  os << first << ",";
  print_csv(os, rest...);
}

} // namespace senk

#endif
