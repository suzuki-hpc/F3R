#ifndef SENK_CORE_TIMER_HPP
#define SENK_CORE_TIMER_HPP

#include <chrono>
#include <iostream>
#include <vector>

namespace senk {

struct timer {
  using point = std::chrono::time_point<std::chrono::steady_clock>;
  std::vector<std::chrono::duration<double>> durations;

  timer() = default;
  void tick() { start = std::chrono::steady_clock::now(); }
  void tock() {
    end = std::chrono::steady_clock::now();
    durations.push_back(std::chrono::duration<double>(end - start));
  }
  void print(const std::string &mod = "", const std::string &fin = "\n") {
    for (const auto &d : durations)
      std::cout << mod << d.count() << fin;
  }

private:
  point start, end;
};

} // namespace senk

#endif // SENK_CORE_TIMER_HPP