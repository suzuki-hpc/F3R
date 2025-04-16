#ifndef UMINEKO_CORE_TENSOR2_HPP
#define UMINEKO_CORE_TENSOR2_HPP

#include <complex>
#include <random>
#include <type_traits>

#include "umineko-core/interface.hpp"
#include "umineko-core/loop.hpp"
#include "umineko-core/memory.hpp"
#include "umineko-core/share.hpp"

#if defined(KMM_WITH_CUDA)
#define KMM_LOC __host__ __device__
#else
#define KMM_LOC
#endif

#define KMM_ENABULER(cond) std::enable_if_t<cond, std::nullptr_t> = nullptr
#define KMM_IS_SAME_V(cond1, cond2) std::is_same_v<cond1, cond2>

namespace kmm {

namespace impl {

#define KMM_DEFINE_IS_EXPR(name)                                               \
  template <typename> struct is_##name##_expr {                                \
    static constexpr bool value = false;                                       \
  };                                                                           \
  template <typename T>                                                        \
  inline constexpr bool is_##name##_expr_v = is_##name##_expr<T>::value;
KMM_DEFINE_IS_EXPR(scalar)
KMM_DEFINE_IS_EXPR(vector)
KMM_DEFINE_IS_EXPR(matrix)
#undef KMM_DEFINE_IS_EXPR

template <typename... Args>
constexpr bool all_integral_v = (std::is_integral_v<Args> && ...);

template <typename> struct is_complex {
  static constexpr bool value = false;
};
template <class T> inline constexpr bool is_complex_v = is_complex<T>::value;
template <typename T> struct is_complex<std::complex<T>> {
  static constexpr bool value = true;
};

template <uint16_t dim> struct _shape {
  idx_t arr[dim];
  template <typename... I>
  explicit _shape(I... args) : arr{static_cast<idx_t>(args)...} {
    for (int i = sizeof...(I); i < dim; ++i)
      arr[i] = 1;
  }
  _shape(const _shape &in) = default;
  explicit _shape(const _shape<dim + 1> &in) : arr{} {
    for (auto i = 0; i < dim; i++)
      arr[i] = in.arr[i];
  }
  [[nodiscard]] KMM_LOC idx_t total() const {
    auto res = arr[0];
    for (auto i = 1; i < dim; i++)
      res *= arr[i];
    return res;
  };
#if 0
  [[nodiscard]] KMM_LOC idx_t offset() const {
    auto res = 1;
    for (auto i = 0; i < dim - 1; i++) {
      res *= num[i];
    }
    return res;
  }
#endif
};

template <> struct _shape<0> {
  _shape() = default;
  explicit _shape(const _shape<1> &) {}
  _shape(const _shape &in) = default;
  [[nodiscard]] KMM_LOC idx_t total() const { return 1; }
  // [[nodiscard]] KMM_LOC idx_t offset() const { return 0; }
};

template <typename T, uint16_t dim, class L, KMM_ENABULER(is_locator_v<L>)>
struct _tensor {
  constexpr static auto get_dim() { return dim; }
  using val_t = T;
  using loc_t = L;
  _shape<dim> shape;
  hd_ptr<T> data;

  template <typename... I, KMM_ENABULER(all_integral_v<I...>)>
  explicit _tensor(I... args)
      : shape(args...),
        data(memory<L>::template alloc<T>(shape.total()), memory<L>::free) {}
  explicit _tensor(const _shape<dim> &sizes)
      : shape(sizes),
        data(memory<L>::template alloc<T>(shape.total()), memory<L>::free) {}
  _tensor(const _tensor &in) : shape(in.shape), data(in.data) {}
  template <typename T2, class L2>
  explicit _tensor(const _tensor<T2, dim, L2> &in)
      : shape(in.shape),
        data(memory<L>::template alloc<T>(shape.total()), memory<L>::free) {}
  _tensor(T *ptr, const _shape<dim> &sizes)
      : shape(sizes), data(ptr, [](void *) {}) {}
  _tensor &operator=(const _tensor &in) = default;
  [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return shape.arr[i]; }
  [[nodiscard]] KMM_LOC T *raw() const { return data.ptr; }
};

} // namespace impl

template <typename T, class L, KMM_ENABULER(is_locator_v<L>)>
struct scalar : impl::_tensor<T, 0, L> {
#define KMM_DEFINE_MEM_FN(name, kernel)                                        \
  scalar &name {                                                               \
    exec<L>::seq([ =, p = raw() ] kernel);                                     \
    return *this;                                                              \
  }
#define KMM_DEFINE_OP_ARITHMETIC(name, kernel)                                 \
  template <typename _T,                                                       \
            KMM_ENABULER(std::is_arithmetic_v<_T> || impl::is_complex_v<_T>)>  \
  KMM_DEFINE_MEM_FN(name, kernel)
#define KMM_DEFINE_OP_SCALAR_EXPR(name, kernel)                                \
  template <typename _T, KMM_ENABULER(impl::is_scalar_expr_v<_T>)>             \
  KMM_DEFINE_MEM_FN(name, kernel)

  using impl::_tensor<T, 0, L>::data;
  using impl::_tensor<T, 0, L>::raw;
  explicit scalar() : impl::_tensor<T, 0, L>() {}
  scalar(const scalar &in) = default;
  template <typename T2, class L2>
  explicit scalar(const scalar<T2, L2> &in) : impl::_tensor<T, 0, L>(in) {
    copy(in);
  };
  explicit scalar(T *ptr, impl::_shape<0>) : impl::_tensor<T, 0, L>(ptr, {}) {};

  KMM_LOC T operator[]([[maybe_unused]] idx_t i) const { return data.ptr[0]; }
  KMM_LOC T &operator[]([[maybe_unused]] idx_t i) { return data.ptr[0]; }
  KMM_LOC T eval([[maybe_unused]] idx_t i = 0,
                 [[maybe_unused]] idx_t j = 0) const {
    return data.ptr[0];
  }

  KMM_DEFINE_MEM_FN(fill(T val), { p[0] = val; })
  KMM_DEFINE_MEM_FN(inv(), { p[0] = 1. / p[0]; })
  KMM_DEFINE_MEM_FN(neg(), { p[0] = -p[0]; })
  KMM_DEFINE_MEM_FN(abs(), { p[0] = std::abs(p[0]); })
  KMM_DEFINE_MEM_FN(sqrt(), { p[0] = std::sqrt(p[0]); })
  KMM_DEFINE_MEM_FN(copy(const scalar &in), { p[0] = in.eval(0); })
  template <typename T2, typename L2, KMM_ENABULER(KMM_IS_SAME_V(T, T2))>
  scalar &copy(const scalar<T2, L2> &in) {
    memory<L2>::template copy<L>(in.raw(), raw(), 1);
    return *this;
  }
  template <typename T2, typename L2, KMM_ENABULER(KMM_IS_SAME_V(L, L2))>
  scalar &copy(const scalar<T2, L2> &in) {
    exec<L>::seq([=, p = raw()] { p[0] = in.eval(0); });
    return *this;
  }
  template <typename T2, typename L2, KMM_ENABULER(!KMM_IS_SAME_V(T, T2)),
            KMM_ENABULER(!KMM_IS_SAME_V(L, L2))>
  scalar &copy(const scalar<T2, L2> &in) {
    if constexpr (std::is_same_v<L, host>) {
      T2 t;
      memory<L2>::template copy<L>(in.raw(), &t, 1);
      raw()[0] = T(t);
    } else {
      T t = in.raw()[0];
      memory<L2>::template copy<L>(&t, raw(), 1);
    }
    return *this;
  }
  KMM_DEFINE_OP_ARITHMETIC(copy(_T *in), { p[0] = T(in[0]); })
  KMM_DEFINE_OP_ARITHMETIC(operator=(_T val), { p[0] = T(val); })
  KMM_DEFINE_OP_ARITHMETIC(operator+=(_T val), { p[0] += val; })
  KMM_DEFINE_OP_ARITHMETIC(operator-=(_T val), { p[0] -= val; })
  KMM_DEFINE_OP_ARITHMETIC(operator*=(_T val), { p[0] *= val; })
  KMM_DEFINE_OP_ARITHMETIC(operator/=(_T val), { p[0] /= val; })
  // KMM_DEFINE_OP_SCALAR_EXPR(operator=(_T right), { p[0] = right.eval(); })
  template <typename _T, KMM_ENABULER(impl::is_scalar_expr_v<_T>)>
  scalar &operator=(_T right) {
    if constexpr (KMM_IS_SAME_V(L, typename _T::loc_t)) {
      exec<L>::seq([=, p = raw()] { p[0] = T(right.eval()); });
    } else {
      scalar<T, typename _T::loc_t> tmp;
      exec<typename _T::loc_t>::seq(
          [=, t = tmp.raw()] { t[0] = right.eval(); });
      copy(tmp);
    }
    return *this;
  }
  KMM_DEFINE_OP_SCALAR_EXPR(operator+=(_T right), { p[0] += right.eval(); })
  KMM_DEFINE_OP_SCALAR_EXPR(operator-=(_T right), { p[0] -= right.eval(); })
  KMM_DEFINE_OP_SCALAR_EXPR(operator*=(_T right), { p[0] *= right.eval(); })
  KMM_DEFINE_OP_SCALAR_EXPR(operator/=(_T right), { p[0] /= right.eval(); })

#undef KMM_DEFINE_OP_SCALAR_EXPR
#undef KMM_DEFINE_OP_ARITHMETIC
#undef KMM_DEFINE_MEM_FN
};

template <typename T, class L, KMM_ENABULER(is_locator_v<L>)>
struct vector : impl::_tensor<T, 1, L> {
#define KMM_DEFINE_MEM_FN(name, kernel)                                        \
  vector &name {                                                               \
    exec<L>::para_for(shape.arr[0], [ =, p = raw() ](idx_t i) kernel);         \
    return *this;                                                              \
  }
#define KMM_DEFINE_OP_ARITHMETIC(name, kernel)                                 \
  template <typename _T,                                                       \
            KMM_ENABULER(std::is_arithmetic_v<_T> || impl::is_complex_v<_T>)>  \
  KMM_DEFINE_MEM_FN(name, kernel)
#define KMM_DEFINE_OP_VECTOR_EXPR(name, kernel)                                \
  template <typename _T, KMM_ENABULER(impl::is_scalar_expr_v<_T> ||            \
                                      impl::is_vector_expr_v<_T>)>             \
  KMM_DEFINE_MEM_FN(name, kernel)

  using impl::_tensor<T, 1, L>::data;
  using impl::_tensor<T, 1, L>::shape;
  using impl::_tensor<T, 1, L>::raw;
  explicit vector(idx_t n) : impl::_tensor<T, 1, L>(n) {}
  vector(const vector &in) = default;
  template <typename T2, class L2>
  explicit vector(const vector<T2, L2> &in) : impl::_tensor<T, 1, L>(in) {
    copy(in);
  };
  explicit vector(T *ptr, impl::_shape<1> size)
      : impl::_tensor<T, 1, L>(ptr, size) {};

  auto operator()(idx_t row) const {
    return scalar<T, L>(raw() + row, impl::_shape<0>(shape));
  }
  KMM_LOC T operator[](idx_t i) const { return data.ptr[i]; }
  KMM_LOC T &operator[](idx_t i) { return data.ptr[i]; }
  KMM_LOC T eval(idx_t i, [[maybe_unused]] idx_t j = 0) const {
    return data.ptr[i];
  }

  vector slice(int len) {
    auto t = shape;
    t.arr[0] = len;
    return vector(raw(), t);
  }

  KMM_DEFINE_MEM_FN(fill(T val), { p[i] = val; })
  KMM_DEFINE_MEM_FN(iota(const int offset), { p[i] = offset + i; })
  KMM_DEFINE_MEM_FN(inv(), { p[i] = 1. / p[i]; })
  KMM_DEFINE_MEM_FN(neg(), { p[i] = -p[i]; })
  KMM_DEFINE_MEM_FN(abs(), { p[i] = std::abs(p[i]); })
  KMM_DEFINE_MEM_FN(sqrt(), { p[i] = std::sqrt(p[i]); })
  template <int l, int r, typename RG> vector &random(RG engine) {
    std::uniform_real_distribution<double> dist1(l, r);
    auto size = shape.total();
    const auto s = 1024;
    T buff[s];
    for (idx_t c = 0; c < size; c += s) {
      auto t_s = (c + s < size) ? s : size - c;
      for (idx_t i = 0; i < t_s; i++)
        buff[i] = dist1(engine);
      memory<host>::copy<L>(buff, raw() + c, t_s);
    }
    return *this;
  }
  KMM_DEFINE_MEM_FN(copy(const vector &in), { p[i] = in.eval(i); })
  template <typename T2, typename L2, KMM_ENABULER(KMM_IS_SAME_V(T, T2))>
  vector &copy(const vector<T2, L2> &in) {
    memory<L2>::template copy<L>(in.raw(), raw(), shape.arr[0]);
    return *this;
  }
  template <typename T2, typename L2, KMM_ENABULER(KMM_IS_SAME_V(L, L2))>
  vector &copy(const vector<T2, L2> &in) {
    exec<L>::para_for(shape.arr[0],
                      [=, p = raw()](idx_t i) { p[i] = T(in.eval(i)); });
    return *this;
  }
  template <typename T2, typename L2, KMM_ENABULER(!KMM_IS_SAME_V(T, T2)),
            KMM_ENABULER(!KMM_IS_SAME_V(L, L2))>
  vector &copy(const vector<T2, L2> &in) {
    const auto s = 65536;
    auto len = shape.arr[0];
    if constexpr (std::is_same_v<L, host>) {
      T2 t[s];
      for (idx_t off = 0; off < len; off += s) {
        auto ss = (off + s < len) ? s : len - off;
        memory<L2>::template copy<L>(in.raw() + off, t, ss);
        exec<L>::para_for(ss, [=, p = raw() + off](idx_t i) { p[i] = t[i]; });
      }
    } else {
      T t[s];
      for (idx_t off = 0; off < len; off += s) {
        auto ss = (off + s < len) ? s : len - off;
        exec<L2>::para_for(
            ss, [=, t = t, p = in.raw() + off](idx_t i) { t[i] = p[i]; });
        memory<L2>::template copy<L>(t, raw() + off, ss);
      }
    }
    return *this;
  }
  KMM_DEFINE_OP_ARITHMETIC(copy(_T *in), { p[i] = T(in[i]); })
  KMM_DEFINE_OP_ARITHMETIC(operator=(_T val), { p[i] = val; })
  KMM_DEFINE_OP_ARITHMETIC(operator+=(_T val), { p[i] += val; })
  KMM_DEFINE_OP_ARITHMETIC(operator-=(_T val), { p[i] -= val; })
  KMM_DEFINE_OP_ARITHMETIC(operator*=(_T val), { p[i] *= val; })
  KMM_DEFINE_OP_ARITHMETIC(operator/=(_T val), { p[i] /= val; })
  KMM_DEFINE_OP_VECTOR_EXPR(operator=(_T right), { p[i] = T(right.eval(i)); })
  KMM_DEFINE_OP_VECTOR_EXPR(operator+=(_T right), { p[i] += T(right.eval(i)); })
  KMM_DEFINE_OP_VECTOR_EXPR(operator-=(_T right), { p[i] -= T(right.eval(i)); })
  KMM_DEFINE_OP_VECTOR_EXPR(operator*=(_T right), { p[i] *= T(right.eval(i)); })
  KMM_DEFINE_OP_VECTOR_EXPR(operator/=(_T right), { p[i] /= T(right.eval(i)); })

#undef KMM_DEFINE_OP_VECTOR_EXPR
#undef KMM_DEFINE_OP_ARITHMETIC
#undef KMM_DEFINE_MEM_FN
};

template <typename T, class L, KMM_ENABULER(is_locator_v<L>)>
struct matrix : impl::_tensor<T, 2, L> {
#define KMM_DEFINE_MEM_FN(name, kernel)                                        \
  matrix &name {                                                               \
    exec<L>::para_for_2d(shape.arr[0], shape.arr[1],                           \
                         [ =, n = shape.arr[0], p = raw() ](idx_t i, idx_t j)  \
                             kernel);                                          \
    return *this;                                                              \
  }
#define KMM_DEFINE_OP_ARITHMETIC(name, kernel)                                 \
  template <typename _T,                                                       \
            KMM_ENABULER(std::is_arithmetic_v<_T> || impl::is_complex_v<_T>)>  \
  KMM_DEFINE_MEM_FN(name, kernel)
#define KMM_DEFINE_OP_MATRIX_EXPR(name, kernel)                                \
  template <typename _T, KMM_ENABULER(impl::is_scalar_expr_v<_T> ||            \
                                      impl::is_vector_expr_v<_T> ||            \
                                      impl::is_matrix_expr_v<_T>)>             \
  KMM_DEFINE_MEM_FN(name, kernel)

  using impl::_tensor<T, 2, L>::data;
  using impl::_tensor<T, 2, L>::shape;
  using impl::_tensor<T, 2, L>::raw;
  explicit matrix(idx_t n, idx_t m) : impl::_tensor<T, 2, L>(n, m) {}
  matrix(const matrix &in) = default;
  template <typename T2, class L2>
  explicit matrix(const matrix<T2, L2> &in) : impl::_tensor<T, 2, L>(in) {
    copy(in);
  };

  auto operator()(idx_t col) const {
    return vector<T, L>(raw() + shape.arr[0] * col, impl::_shape<1>(shape));
  }
  auto operator()(idx_t col, idx_t row) const {
    return scalar<T, L>(raw() + shape.arr[0] * col + row, impl::_shape<0>());
  }
  KMM_LOC T operator[](idx_t i) const { return data.ptr[i]; }
  KMM_LOC T &operator[](idx_t i) { return data.ptr[i]; }
  KMM_LOC T eval(idx_t i = 0, idx_t j = 0) const {
    return data.ptr[j * shape.arr[0] + i];
  }

  KMM_DEFINE_MEM_FN(fill(T val), { p[j * n + i] = val; })
  // KMM_DEFINE_MEM_FN(iota(const int offset), { p[j * n + i] = offset + i; })
  KMM_DEFINE_MEM_FN(inv(), { p[j * n + i] = 1. / p[j * n + i]; })
  KMM_DEFINE_MEM_FN(neg(), { p[j * n + i] = -p[j * n + i]; })
  KMM_DEFINE_MEM_FN(abs(), { p[j * n + i] = std::abs(p[j * n + i]); })
  KMM_DEFINE_MEM_FN(sqrt(), { p[j * n + i] = std::sqrt(p[j * n + i]); })
  template <int l, int r, typename RG> matrix &random(RG engine) {
    std::uniform_real_distribution<double> dist1(l, r);
    auto size = shape.total();
    const auto s = 1024;
    T buff[s];
    for (idx_t c = 0; c < size; c += s) {
      auto t_s = (c + s < size) ? s : size - c;
      for (idx_t i = 0; i < t_s; i++)
        buff[i] = dist1(engine);
      memory<host>::copy<L>(buff, raw() + c, t_s);
    }
    return *this;
  }
  KMM_DEFINE_MEM_FN(copy(const matrix &in), { p[j * n + i] = in.eval(i, j); })
  template <typename T2, typename L2, KMM_ENABULER(KMM_IS_SAME_V(T, T2))>
  matrix &copy(const matrix<T2, L2> &in) {
    memory<L2>::template copy<L>(in.raw(), raw(), shape.total());
    return *this;
  }
  template <typename T2, typename L2, KMM_ENABULER(KMM_IS_SAME_V(L, L2))>
  matrix &copy(const matrix<T2, L2> &in) {
    exec<L>::para_for(shape.total(),
                      [=, p = raw()](idx_t i) { p[i] = in.eval(i); });
    return *this;
  }
  template <typename T2, typename L2, KMM_ENABULER(!KMM_IS_SAME_V(T, T2)),
            KMM_ENABULER(!KMM_IS_SAME_V(L, L2))>
  matrix &copy(const matrix<T2, L2> &in) {
    const auto s = 65536;
    auto len = shape.total();
    if constexpr (std::is_same_v<L, host>) {
      T2 t[s];
      for (idx_t off = 0; off < len; off += s) {
        auto ss = (off + s < len) ? s : len - off;
        memory<L2>::template copy<L>(in.raw() + off, t, ss);
        exec<L>::para_for(ss, [=, p = raw() + off](idx_t i) { p[i] = t[i]; });
      }
    } else {
      T t[s];
      for (idx_t off = 0; off < len; off += s) {
        auto ss = (off + s < len) ? s : len - off;
        exec<L2>::para_for(
            ss, [=, t = t, p = in.raw() + off](idx_t i) { t[i] = p[i]; });
        memory<L2>::template copy<L>(t, raw() + off, ss);
      }
    }
    return *this;
  }
  KMM_DEFINE_OP_ARITHMETIC(copy(_T *in), { p[j * n + i] = T(in[j * n + i]); })
  KMM_DEFINE_OP_ARITHMETIC(operator=(_T val), { p[j * n + i] = val; })
  KMM_DEFINE_OP_ARITHMETIC(operator+=(_T val), { p[j * n + i] += val; })
  KMM_DEFINE_OP_ARITHMETIC(operator-=(_T val), { p[j * n + i] -= val; })
  KMM_DEFINE_OP_ARITHMETIC(operator*=(_T val), { p[j * n + i] *= val; })
  KMM_DEFINE_OP_ARITHMETIC(operator/=(_T val), { p[j * n + i] /= val; })
  KMM_DEFINE_OP_MATRIX_EXPR(operator=(_T right),
                            { p[j * n + i] = right.eval(i, j); })
  KMM_DEFINE_OP_MATRIX_EXPR(operator+=(_T right),
                            { p[j * n + i] += right.eval(i, j); })
  KMM_DEFINE_OP_MATRIX_EXPR(operator-=(_T right),
                            { p[j * n + i] -= right.eval(i, j); })
  KMM_DEFINE_OP_MATRIX_EXPR(operator*=(_T right),
                            { p[j * n + i] *= right.eval(i, j); })
  KMM_DEFINE_OP_MATRIX_EXPR(operator/=(_T right),
                            { p[j * n + i] /= right.eval(i, j); })

#undef KMM_DEFINE_OP_MATRIX_EXPR
#undef KMM_DEFINE_OP_ARITHMETIC
#undef KMM_DEFINE_MEM_FN
};

namespace impl {

template <typename T, class L> struct is_scalar_expr<scalar<T, L>> {
  static constexpr bool value = true;
};
template <typename T, class L> struct is_vector_expr<vector<T, L>> {
  static constexpr bool value = true;
};
template <typename T, class L> struct is_matrix_expr<matrix<T, L>> {
  static constexpr bool value = true;
};

#define KMM_DEFINE_EXPR(name, op, type, type_l, type_r)                        \
  template <typename L, typename R, KMM_ENABULER(is_##type_l##_expr_v<L>),     \
            KMM_ENABULER(is_##type_r##_expr_v<R>)>                             \
  struct name {                                                                \
    using loc_t = typename L::loc_t;                                           \
    using val_t = decltype(std::declval<typename L::val_t &>() *               \
                           std::declval<typename R::val_t &>());               \
    L l;                                                                       \
    R r;                                                                       \
    name(const L &l, const R &r) : l(l), r(r) {}                               \
    KMM_LOC auto eval(idx_t i = 0, idx_t j = 0) const {                        \
      return l.eval(i, j) op r.eval(i, j);                                     \
    }                                                                          \
    [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return r.size(i); }      \
  };                                                                           \
  template <typename T1, typename T2> struct is_##type##_expr<name<T1, T2>> {  \
    static constexpr bool value = true;                                        \
  }
KMM_DEFINE_EXPR(ADDSS, +, scalar, scalar, scalar);
KMM_DEFINE_EXPR(SUBSS, -, scalar, scalar, scalar);
KMM_DEFINE_EXPR(MULSS, *, scalar, scalar, scalar);
KMM_DEFINE_EXPR(DIVSS, /, scalar, scalar, scalar);

KMM_DEFINE_EXPR(ADDVV, +, vector, vector, vector);
KMM_DEFINE_EXPR(SUBVV, -, vector, vector, vector);
KMM_DEFINE_EXPR(MULVV, *, vector, vector, vector);
KMM_DEFINE_EXPR(DIVVV, /, vector, vector, vector);
KMM_DEFINE_EXPR(MULSV, *, vector, scalar, vector);
// KMM_DEFINE_EXPR(MULVS, *, vector, vector, scalar);
KMM_DEFINE_EXPR(DIVVS, /, vector, vector, scalar);

template <typename L, typename R, KMM_ENABULER(is_vector_expr_v<L>)>
struct WIDL {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<R>()(std::declval<typename L::val_t>()));
  L l;
  R r;
  WIDL(const L &l, const R &r) : l(l), r(r) {}
  KMM_LOC val_t eval(idx_t i = 0, idx_t j = 0) const { return r(l.eval(i, j)); }
  [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return l.size(i); }
};
template <typename T1, typename T2> struct is_vector_expr<WIDL<T1, T2>> {
  static constexpr bool value = true;
};

template <typename T, class L> struct pool {
  inline static vector<T, L> *buffer = nullptr;
  inline static scalar<T, L> *receiver = nullptr;
  static void init(idx_t size) {
    if (buffer == nullptr) {
      buffer = new vector<T, L>(size);
      receiver = new scalar<T, L>();
    }
  }
  static void clean() {
    delete buffer;
    buffer = nullptr;
    delete receiver;
    receiver = nullptr;
  }
};

template <typename L, typename R, KMM_ENABULER(is_vector_expr_v<L>),
          KMM_ENABULER(is_vector_expr_v<R>)>
struct DOT : pool<decltype(std::declval<typename L::val_t &>() *
                           std::declval<typename R::val_t &>()),
                  typename L::loc_t> {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<typename L::val_t &>() *
                         std::declval<typename R::val_t &>());
  val_t *ptr = this->receiver->raw();
  L l;
  R r;
  DOT(const L &l, const R &r) : l(l), r(r) {
    if constexpr (is_complex_v<val_t>) {
      exec<loc_t>::para_reduce(
          r.size(0), ptr, 0,
          [=](auto k) { return std::conj(l.eval(k)) * r.eval(k); },
          this->buffer->raw());
    } else {
      exec<loc_t>::para_reduce(
          r.size(0), ptr, 0, [=](auto k) { return l.eval(k) * r.eval(k); },
          this->buffer->raw());
    }
  }
  KMM_LOC auto eval([[maybe_unused]] idx_t i = 0,
                    [[maybe_unused]] idx_t j = 0) const {
    return ptr[0];
  }
  [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return r.size(i); }
};
template <typename T1, typename T2> struct is_scalar_expr<DOT<T1, T2>> {
  static constexpr bool value = true;
};

template <typename L, KMM_ENABULER(is_vector_expr_v<L>)>
struct NRM : pool<decltype(std::declval<typename L::val_t &>() *
                           std::declval<typename L::val_t &>()),
                  typename L::loc_t> {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<typename L::val_t &>() *
                         std::declval<typename L::val_t &>());
  val_t *ptr = this->receiver->raw();
  L l;
  explicit NRM(const L &l) : l(l) {
    if constexpr (is_complex_v<val_t>) {
      exec<loc_t>::para_reduce(
          l.size(0), ptr, 0,
          [=](auto k) { return std::conj(l.eval(k)) * l.eval(k); },
          this->buffer->raw());
    } else {
      exec<loc_t>::para_reduce(
          l.size(0), ptr, 0, [=](auto k) { return l.eval(k) * l.eval(k); },
          this->buffer->raw());
    }
  }
  KMM_LOC auto eval([[maybe_unused]] idx_t i = 0,
                    [[maybe_unused]] idx_t j = 0) const {
    return sqrt(ptr[0]);
  }
  [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return l.size(i); }
};
template <typename T1> struct is_scalar_expr<NRM<T1>> {
  static constexpr bool value = true;
};

template <typename L, KMM_ENABULER(is_vector_expr_v<L>)>
struct REDUCE : pool<decltype(std::declval<typename L::val_t &>() *
                              std::declval<typename L::val_t &>()),
                     typename L::loc_t> {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<typename L::val_t &>() *
                         std::declval<typename L::val_t &>());
  val_t *ptr = this->receiver->raw();
  L l;
  explicit REDUCE(const L &l) : l(l) {
    exec<loc_t>::para_reduce(
        l.size(0), ptr, 0, [=](auto k) { return l.eval(k); },
        this->buffer->raw());
  }
  KMM_LOC auto eval([[maybe_unused]] idx_t i = 0,
                    [[maybe_unused]] idx_t j = 0) const {
    return ptr[0];
  }
  [[nodiscard]] KMM_LOC idx_t size(idx_t i) const { return l.size(i); }
};
template <typename T1> struct is_scalar_expr<REDUCE<T1>> {
  static constexpr bool value = true;
};

KMM_DEFINE_EXPR(ADDMM, +, matrix, matrix, matrix);
KMM_DEFINE_EXPR(SUBMM, -, matrix, matrix, matrix);
KMM_DEFINE_EXPR(MULMM, *, matrix, matrix, matrix);
KMM_DEFINE_EXPR(DIVMM, /, matrix, matrix, matrix);
KMM_DEFINE_EXPR(MULSM, *, matrix, scalar, matrix);
KMM_DEFINE_EXPR(MULMS, *, matrix, matrix, scalar);
KMM_DEFINE_EXPR(DIVMS, /, matrix, matrix, scalar);

template <typename L, typename R, KMM_ENABULER(is_matrix_expr_v<L>),
          KMM_ENABULER(is_vector_expr_v<R>)>
struct MULMV {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<typename L::val_t &>() *
                         std::declval<typename R::val_t &>());
  L l;
  R r;
  MULMV(const L &l, const R &r) : l(l), r(r) {}
  KMM_LOC auto eval(idx_t i = 0, [[maybe_unused]] idx_t j = 0) const {
    val_t res = l.eval(i, 0) * r.eval(0);
    for (idx_t k = 1; k < r.size(0); k++)
      res += l.eval(i, k) * r.eval(k);
    return res;
  }
};
template <typename T1, typename T2> struct is_vector_expr<MULMV<T1, T2>> {
  static constexpr bool value = true;
};

template <typename R, KMM_ENABULER(is_scalar_expr_v<R>)> struct NEGS {
  using loc_t = typename R::loc_t;
  using val_t = typename R::val_t;
  R r;
  NEGS(const R &_r) : r(_r) {}
  KMM_LOC auto eval([[maybe_unused]] idx_t i = 0,
                    [[maybe_unused]] idx_t j = 0) const {
    return -r.eval(0);
  }
  int size() const { return r.size(); }
};
template <typename T1> struct is_scalar_expr<NEGS<T1>> {
  static constexpr bool value = true;
};

template <typename R, KMM_ENABULER(is_vector_expr_v<R>)> struct NEGV {
  using loc_t = typename R::loc_t;
  using val_t = typename R::val_t;
  R r;
  NEGV(const R &_r) : r(_r) {}
  KMM_LOC auto eval(idx_t i, [[maybe_unused]] idx_t j = 0) const {
    return -r.eval(i);
  }
  int size() const { return r.size(); }
};
template <typename T1> struct is_vector_expr<NEGV<T1>> {
  static constexpr bool value = true;
};

#undef KMM_DEFINE_EXPR

} // namespace impl

} // namespace kmm

template <typename R, KMM_ENABULER(kmm::impl::is_scalar_expr_v<R>)>
kmm::impl::NEGS<R> operator-(const R &v2) {
  return kmm::impl::NEGS<R>(R(v2));
};

template <typename R, KMM_ENABULER(kmm::impl::is_vector_expr_v<R>)>
kmm::impl::NEGV<R> operator-(const R &v2) {
  return kmm::impl::NEGV<R>(R(v2));
};

#define KMM_DEFINE_BINARY_OP(name, op)                                         \
  template <typename L, typename R> name<L, R> operator op(L v1, R v2) {       \
    return name<L, R>(L(v1), R(v2));                                           \
  }
KMM_DEFINE_BINARY_OP(kmm::impl::ADDSS, +);
KMM_DEFINE_BINARY_OP(kmm::impl::SUBSS, -);
KMM_DEFINE_BINARY_OP(kmm::impl::MULSS, *);
KMM_DEFINE_BINARY_OP(kmm::impl::DIVSS, /);

KMM_DEFINE_BINARY_OP(kmm::impl::ADDVV, +);
KMM_DEFINE_BINARY_OP(kmm::impl::SUBVV, -);
KMM_DEFINE_BINARY_OP(kmm::impl::MULVV, *);
KMM_DEFINE_BINARY_OP(kmm::impl::DIVVV, /);
KMM_DEFINE_BINARY_OP(kmm::impl::MULSV, *);
// KMM_DEFINE_BINARY_OP(kmm::impl::MULVS, *);
KMM_DEFINE_BINARY_OP(kmm::impl::DIVVS, /);
template <typename L, typename R> kmm::impl::DOT<L, R> dot(L v1, R v2) {
  return kmm::impl::DOT<L, R>(L(v1), R(v2));
}
template <typename L> kmm::impl::NRM<L> nrm(L v1) {
  return kmm::impl::NRM<L>(L(v1));
}
template <typename L> kmm::impl::REDUCE<L> reduce(L v1) {
  return kmm::impl::REDUCE<L>(L(v1));
}
template <typename L, typename R> kmm::impl::WIDL<L, R> wildcard(L v1, R v2) {
  return kmm::impl::WIDL<L, R>(L(v1), R(v2));
}

KMM_DEFINE_BINARY_OP(kmm::impl::ADDMM, +);
KMM_DEFINE_BINARY_OP(kmm::impl::SUBMM, -);
KMM_DEFINE_BINARY_OP(kmm::impl::MULMM, *);
KMM_DEFINE_BINARY_OP(kmm::impl::DIVMM, /);
KMM_DEFINE_BINARY_OP(kmm::impl::MULSM, *);
KMM_DEFINE_BINARY_OP(kmm::impl::MULMS, *);
KMM_DEFINE_BINARY_OP(kmm::impl::DIVMS, /);

KMM_DEFINE_BINARY_OP(kmm::impl::MULMV, *);
#undef KMM_DEFINE_BINARY_OP

#undef KMM_LOC
#undef KMM_IS_SAME_V
#undef KMM_ENABULER

#if !defined(__NVCOMPILER)
#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in)
#endif

#endif
