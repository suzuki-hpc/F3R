#ifndef UMINEKO_CORE_TIMER_HPP
#define UMINEKO_CORE_TIMER_HPP

#include <chrono>
#include <iostream>
#include <vector>

namespace kmm {

struct timer {
  using point = std::chrono::time_point<std::chrono::steady_clock>;

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
#if 0
  timer &operator+=(timer in) {
    duration += in.duration;
    return *this;
  }
#endif

// private:
  point start, end;
  std::vector<std::chrono::duration<double>> durations;
};

} // namespace kmm

#endif // UMINEKO_CORE_TIMER_HPP