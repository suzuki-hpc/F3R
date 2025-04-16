#ifndef UMINEKO_SPARSE_MATRIX_BASE_HPP
#define UMINEKO_SPARSE_MATRIX_BASE_HPP

#include <array>

#include "umineko-core/interface.hpp"
#include "umineko-core/tensor.hpp"

namespace kmm {

namespace impl {

using shape = std::array<idx_t, 2>;

template <class L> struct pattern_set {
  idx_t unit_size;
  idx_t idx_size;
  vector<idx_t, L> idx_ptr;
  pattern_set() : unit_size(0), idx_size(0), idx_ptr(0) {}
  pattern_set(
      const idx_t &unit_size, const idx_t &idx_size, const vector<idx_t, L> &idx_ptr)
      : unit_size(unit_size), idx_size(idx_size), idx_ptr(idx_ptr) {}
  [[nodiscard]] pattern_set clone() const {
    auto res_ptr = vector<idx_t, L>(idx_ptr.size(0)).copy(idx_ptr);
    return pattern_set(unit_size, idx_size, res_ptr);
  }
  pattern_set &reverse() {
    exec<L>::para_for((idx_size + 1 + 1) / 2,
        [=, n = idx_ptr[idx_size], s = idx_size, *this](idx_t i) mutable {
          auto tmp = idx_ptr[i];
          idx_ptr[i] = n - idx_ptr[s - i];
          idx_ptr[s - i] = n - tmp;
        });
    return *this;
  }
  // pattern_set &copy(const pattern_set &other) {
  //   unit_size = other.unit_size;
  //   idx_size = other.idx_size;
  //   idx_ptr.copy(other.idx_ptr);
  //   pi.copy(other.pi);
  //   return *this;
  // }
};

} // namespace impl

struct spmat {
  impl::shape shp;
  idx_t nz;
  enum class Type { real, integer, complex } type;
  enum class Sym { general, symmetric, other } sym;
  enum class Form { square, lower, lower_d, d_upper, upper } form;
  enum class Pattern {
    general,
    diagonal,
    block_diagonal,
    colored,
    colored2,
    partitioned
  } pattern;
  impl::pattern_set<host> pattern_set;

  spmat()
      : shp({1, 1}), nz(1), type(Type::real), sym(Sym::general), form(Form::square),
        pattern(Pattern::general) {}
  explicit spmat(const impl::shape &s, const idx_t nz, const Type type = Type::real,
      const Sym sym = Sym::general, const Form form = Form::square,
      const enum Pattern pattern = Pattern::general)
      : shp(s), nz(nz), type(type), sym(sym), form(form), pattern(pattern) {}

  [[nodiscard]] spmat duplicate() const { return spmat().copy(*this); }

  spmat &copy(const spmat &in) {
    shp = in.shp;
    nz = in.nz;
    type = in.type;
    sym = in.sym;
    form = in.form;
    pattern = in.pattern;
    return *this;
  };

  void copy_attrs(const spmat &in) {
    type = in.type;
    sym = in.sym;
    form = in.form;
    pattern = in.pattern;
    pattern_set = in.pattern_set;
  }

  [[nodiscard]] idx_t nrows() const { return shp[0]; }
  [[nodiscard]] idx_t ncols() const { return shp[1]; }
  [[nodiscard]] idx_t nnz() const { return nz; }
  [[nodiscard]] impl::shape get_shape() const { return shp; }
  [[nodiscard]] bool is_symmetric() const { return sym == Sym::symmetric; }
  [[nodiscard]] bool is_colored() const { return pattern == Pattern::colored; }

  spmat &change_sym(const Sym _sym) {
    sym = _sym;
    return *this;
  }
  spmat &change_form(const Form _form) {
    form = _form;
    return *this;
  }
  spmat &change_pattern(const Pattern _pattern) {
    pattern = _pattern;
    return *this;
  }
};

} // namespace kmm

#endif // UMINEKO_SPARSE_MATRIX_BASE_HPP
