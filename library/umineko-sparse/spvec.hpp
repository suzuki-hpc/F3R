#ifndef UMINEKO_SPARSE_SPVEC_HPP
#define UMINEKO_SPARSE_SPVEC_HPP

#include <cmath>
#include <limits>
#include <vector>

#include "umineko-core/interface.hpp"

namespace kmm {

template <typename T> struct SpVec {
  std::vector<std::pair<T, idx_t>> elems;
  explicit SpVec(idx_t _size) { elems.reserve(_size); }
  void append(const std::pair<T, idx_t> &_elem) {
    if (_elem.first != 0)
      elems.push_back(_elem);
  }
  [[nodiscard]] idx_t size() const { return idx_t(elems.size()); }

  T dot(const SpVec<T> &in) {
    T res = 0;
    idx_t pos = 0, in_pos = 0;
    while (pos < size() && in_pos < in.size()) {
      idx_t temp = elems[pos].second;
      idx_t in_temp = in.elems[in_pos].second;
      if (temp < in_temp) {
        pos++;
      } else if (temp == in_temp) {
        res += elems[pos].first * in.elems[in_pos].first;
        pos++;
        in_pos++;
      } else {
        in_pos++;
      }
    }
    return res;
  }

  T dot(T const *in_val, idx_t const *in_idx, idx_t in_length) {
    T res = 0;
    idx_t pos = 0, in_pos = 0;
    while (pos < this->size() && in_pos < in_length) {
      idx_t temp = elems[pos].second;
      idx_t in_temp = in_idx[in_pos];
      if (temp < in_temp) {
        pos++;
      } else if (temp == in_temp) {
        res += elems[pos].first * in_val[in_pos];
        pos++;
        in_pos++;
      } else {
        in_pos++;
      }
    }
    return res;
  }

  void set(const SpVec<T> &out) {
    elems.reserve(out.elems.size());
    elems.resize(out.elems.size());
    for (idx_t i = 0; i < this->size(); i++)
      elems[i] = out.elems[i];
  }

  SpVec<T> &set_axpy(T alpha, const SpVec<T> &x, const SpVec<T> &y, T tol) {
    idx_t x_pos = 0, y_pos = 0;
    elems.clear();
    while (x_pos < x.size() || y_pos < y.size()) {
      T x_temp = (x_pos < x.size()) ? x.elems[x_pos].second
                                    : std::numeric_limits<idx_t>::max();
      T y_temp = (y_pos < y.size()) ? y.elems[y_pos].second
                                    : std::numeric_limits<idx_t>::max();
      std::pair<T, idx_t> elem;
      if (x_temp < y_temp) {
        elem = x.elems[x_pos];
        elem.first *= alpha;
        x_pos++;
      } else if (x_temp == y_temp) {
        elem = y.elems[y_pos];
        elem.first += alpha * x.elems[x_pos].first;
        x_pos++;
        y_pos++;
      } else {
        elem = y.elems[y_pos];
        y_pos++;
      }
      if (std::abs(elem.first) >= tol)
        append(elem);
    }
    return *this;
  }

  SpVec<T> &set_axpy_ainv(T alpha, const SpVec<T> &x, const SpVec<T> &y, T tol,
                          SpVec<T> *h, int idx) {
    idx_t x_pos = 0, y_pos = 0;
    elems.clear();
    bool flag;
    while (x_pos < x.size() || y_pos < y.size()) {
      flag = false;
      T x_temp = (x_pos < x.size()) ? x.elems[x_pos].second
                                    : std::numeric_limits<idx_t>::max();
      T y_temp = (y_pos < y.size()) ? y.elems[y_pos].second
                                    : std::numeric_limits<idx_t>::max();
      std::pair<T, idx_t> elem;
      if (x_temp < y_temp) {
        elem = x.elems[x_pos];
        elem.first *= alpha;
        x_pos++;
        flag = true;
      } else if (x_temp == y_temp) {
        elem = y.elems[y_pos];
        elem.first += alpha * x.elems[x_pos].first;
        x_pos++;
        y_pos++;
      } else {
        elem = y.elems[y_pos];
        y_pos++;
      }
      if (std::abs(elem.first) >= tol) {
        append(elem);
        if (flag)
          h[elem.second].append(std::pair<T, idx_t>{1.0, idx});
      }
    }
    return *this;
  }
};

} // namespace kmm

#endif
