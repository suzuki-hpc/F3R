#ifndef UNMK_CORE_TENSOR_HPP
#define UNMK_CORE_TENSOR_HPP

#include <cmath>
#include <memory>
#include <random>

#include "senk/core/io.hpp"
#include "senk/core/kernel.hpp"
#include "senk/core/math.hpp"
#include "senk/core/memory.hpp"
#include "senk/core/tools.hpp"

namespace senk {

template <class L>
struct reducer;
template <typename T, class L>
struct scalar;
template <typename T, class L>
struct vector;
template <typename T, class L>
struct matrix;

namespace impl {

template <class>
struct is_complex : std::false_type {};
template <class T>
inline constexpr bool is_complex_v = is_complex<T>::value;
template <typename T>
struct is_complex<complex<T>> : std::true_type {};

template <class T>
struct is_arith : std::disjunction<std::is_arithmetic<T>, is_complex<T>> {};
template <class T>
inline constexpr bool is_arith_v = is_arith<T>::value;

#define _UMNK_DEFINE_IS_X(type)                                                \
  template <class>                                                             \
  struct is_##type : std::false_type {};                                       \
  template <class T>                                                           \
  inline constexpr bool is_##type##_v = is_##type<T>::value;                   \
  template <class>                                                             \
  struct is_##type##_expr : std::false_type {};                                \
  template <class T>                                                           \
  inline constexpr bool is_##type##_expr##_v = is_##type##_expr<T>::value;
_UMNK_DEFINE_IS_X(scalar)
_UMNK_DEFINE_IS_X(vector)
_UMNK_DEFINE_IS_X(matrix)
#undef _UMNK_DEFINE_IS_X

template <typename T, class L>
struct is_scalar<scalar<T, L>> : std::true_type {};
template <typename T, class L>
struct is_vector<vector<T, L>> : std::true_type {};
template <typename T, class L>
struct is_matrix<matrix<T, L>> : std::true_type {};

template <class T>
struct is_tensor : std::disjunction<is_scalar<T>, is_vector<T>, is_matrix<T>> {
};
template <class T>
inline constexpr bool is_tensor_v = is_tensor<T>::value;
template <class T>
struct is_tensor_expr : std::disjunction<is_scalar_expr<T>, is_vector_expr<T>,
                            is_matrix_expr<T>> {};
template <class T>
inline constexpr bool is_tensor_expr_v = is_tensor_expr<T>::value;

#define UMNK_DEFINE_BIN_EXPR(name, op, type, type_l, type_r)                   \
  template <typename L, typename R,                                            \
      SENK_ENABULER(is_##type_l##_expr_v<L> &&is_##type_r##_expr_v<R>),        \
      SENK_ENABULER(SENK_IS_SAME_V(typename L::loc_t, typename R::loc_t))>     \
  struct name {                                                                \
    using loc_t = typename L::loc_t;                                           \
    using val_t = decltype(std::declval<typename L::val_t &>()                 \
            op std::declval<typename R::val_t &>());                           \
    L l;                                                                       \
    R r;                                                                       \
    template <typename LL, typename RR,                                        \
        SENK_ENABULER(SENK_IS_CONV_V(LL &&, L) && SENK_IS_CONV_V(RR &&, R))>   \
    name(LL &&l, RR &&r) : l(std::forward<LL>(l)), r(std::forward<RR>(r)) {}   \
    SENK_LOC auto eval(size_t i = 0, size_t j = 0) const {                     \
      return l.eval(i, j) op r.eval(i, j);                                     \
    }                                                                          \
    [[nodiscard]] size_t shape(size_t i) const { return r.shape(i); }          \
  };                                                                           \
  template <typename T1, typename T2>                                          \
  struct is_##type##_expr<name<T1, T2>> : std::true_type {}
UMNK_DEFINE_BIN_EXPR(ADDSS, +, scalar, scalar, scalar);
UMNK_DEFINE_BIN_EXPR(SUBSS, -, scalar, scalar, scalar);
UMNK_DEFINE_BIN_EXPR(MULSS, *, scalar, scalar, scalar);
UMNK_DEFINE_BIN_EXPR(DIVSS, /, scalar, scalar, scalar);

UMNK_DEFINE_BIN_EXPR(ADDVV, +, vector, vector, vector);
UMNK_DEFINE_BIN_EXPR(SUBVV, -, vector, vector, vector);
UMNK_DEFINE_BIN_EXPR(MULVV, *, vector, vector, vector);
UMNK_DEFINE_BIN_EXPR(DIVVV, /, vector, vector, vector);

UMNK_DEFINE_BIN_EXPR(MULVS, *, vector, vector, scalar);
UMNK_DEFINE_BIN_EXPR(MULSV, *, vector, scalar, vector);
UMNK_DEFINE_BIN_EXPR(DIVVS, /, vector, vector, scalar);
#undef UMNK_DEFINE_BIN_EXPR

#define PST_DEFINE_NEG_EXPR(name, cond, expr)                                  \
  template <typename L, SENK_ENABULER(cond)>                                   \
  struct name {                                                                \
    using val_t = typename L::val_t;                                           \
    using loc_t = typename L::loc_t;                                           \
    L l;                                                                       \
    template <typename LL, SENK_ENABULER(SENK_IS_CONV_V(LL &&, L))>            \
    name(LL &&l) : l(std::forward<LL>(l)) {}                                   \
    SENK_LOC auto eval(size_t i = 0, size_t j = 0) const { return expr; }      \
    [[nodiscard]] size_t shape(size_t i) const { return l.shape(i); }          \
  };
PST_DEFINE_NEG_EXPR(NEGS, is_scalar_expr_v<L>, -l.eval(i, j));
PST_DEFINE_NEG_EXPR(NEGV, is_vector_expr_v<L>, -l.eval(i, j));
PST_DEFINE_NEG_EXPR(NEGM, is_matrix_expr_v<L>, -l.eval(i, j));
template <typename T1>
struct is_scalar_expr<NEGS<T1>> : std::true_type {};
template <typename T1>
struct is_vector_expr<NEGV<T1>> : std::true_type {};
template <typename T1>
struct is_matrix_expr<NEGM<T1>> : std::true_type {};
#undef PST_DEFINE_NEG_EXPR

#define PST_DEFINE_APPLY_EXPR(name, cond)                                      \
  template <typename L, class F, SENK_ENABULER(cond)>                          \
  struct name {                                                                \
    using val_t = std::invoke_result_t<F, typename L::val_t>;                  \
    using loc_t = typename L::loc_t;                                           \
    L l;                                                                       \
    F unary;                                                                   \
    template <typename LL, SENK_ENABULER(SENK_IS_CONV_V(LL &&, L))>            \
    name(LL &&l, F unary) : l(std::forward<LL>(l)), unary(unary) {}            \
    SENK_LOC auto eval(size_t i = 0, size_t j = 0) const {                     \
      return unary(l.eval(i, j));                                              \
    }                                                                          \
    [[nodiscard]] size_t shape(size_t i) const { return l.shape(i); }          \
  };
PST_DEFINE_APPLY_EXPR(APPS, is_scalar_expr_v<L>);
PST_DEFINE_APPLY_EXPR(APPV, is_vector_expr_v<L>);
PST_DEFINE_APPLY_EXPR(APPM, is_matrix_expr_v<L>);
template <typename T, typename F>
struct is_scalar_expr<APPS<T, F>> : std::true_type {};
template <typename T, typename F>
struct is_vector_expr<APPV<T, F>> : std::true_type {};
template <typename T, typename F>
struct is_matrix_expr<APPM<T, F>> : std::true_type {};
#undef PST_DEFINE_APPLY_EXPR

#define PST_DEFINE_AS_EXPR(name, cond)                                         \
  template <typename L, typename F, SENK_ENABULER(cond)>                       \
  struct name##_wrapper {                                                      \
    using val_t = F;                                                           \
    using loc_t = typename L::loc_t;                                           \
    L l;                                                                       \
    template <typename LL, SENK_ENABULER(SENK_IS_CONV_V(LL &&, L))>            \
    name##_wrapper(LL &&l) : l(std::forward<LL>(l)) {}                         \
    SENK_LOC auto eval(size_t i = 0, size_t j = 0) const {                     \
      return static_cast<F>(l.eval(i, j));                                     \
    }                                                                          \
    [[nodiscard]] size_t shape(size_t i) const { return l.shape(i); }          \
  };                                                                           \
  template <typename L, typename F, SENK_ENABULER(cond)>                       \
  using name = std::conditional_t<std::is_same_v<typename L::val_t, F>, L,     \
      name##_wrapper<L, F>>;
PST_DEFINE_AS_EXPR(ASS, is_scalar_expr_v<L>);
PST_DEFINE_AS_EXPR(ASV, is_vector_expr_v<L>);
PST_DEFINE_AS_EXPR(ASM, is_matrix_expr_v<L>);
template <typename T, typename F>
struct is_scalar_expr<ASS_wrapper<T, F>> : std::true_type {};
template <typename T, typename F>
struct is_vector_expr<ASV_wrapper<T, F>> : std::true_type {};
template <typename T, typename F>
struct is_matrix_expr<ASM_wrapper<T, F>> : std::true_type {};
#undef PST_DEFINE_AS_EXPR

template <typename L, typename R,
    SENK_ENABULER(is_matrix_expr_v<L> &&is_vector_expr_v<R>),
    SENK_ENABULER(SENK_IS_SAME_V(typename L::loc_t, typename R::loc_t))>
struct MULMV {
  using loc_t = typename L::loc_t;
  using val_t = decltype(std::declval<typename L::val_t &>() *
                         std::declval<typename R::val_t &>());
  L l;
  R r;
  MULMV(const L &l, const R &r) : l(l), r(r) {}
  SENK_LOC auto eval(size_t i = 0, [[maybe_unused]] size_t j = 0) const {
    val_t res = l.eval(i, 0) * r.eval(0);
    for (size_t k = 1; k < r.shape(0); k++)
      res += l.eval(i, k) * r.eval(k);
    return res;
  }
};
template <typename T1, typename T2>
struct is_vector_expr<MULMV<T1, T2>> : std::true_type {};

template <uint16_t dim>
struct _shape {
  size_t arr[dim];
  template <typename... I,
      SENK_ENABULER(all_integral_v<I...> && sizeof...(I) == dim)>
  _shape(I... args) : arr{static_cast<size_t>(args)...} {}
  _shape(const _shape &in) = default;
  _shape &operator=(const _shape &other) = default;
  explicit _shape(const _shape<dim + 1> &in) : arr{} {
    for (auto i = 0; i < dim; i++)
      arr[i] = in.arr[i];
  }
  size_t total() const {
    auto res = arr[0];
    for (auto i = 1; i < dim; i++)
      res *= arr[i];
    return res;
  }
};

template <>
struct _shape<0> {
  _shape() = default;
  explicit _shape(const _shape<1> &) {}
  _shape(const _shape &in) = default;
  _shape &operator=(const _shape &other) = default;
  size_t total() const { return 1; }
};

template <typename T, uint16_t dim, class L>
struct tensor : public _shape<dim>, public has_val_t<T>, public has_loc_t<L> {
  constexpr static auto get_dim() { return dim; }
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  std::shared_ptr<val_t> data;

  template <typename... I, SENK_ENABULER(all_integral_v<I...>)>
  explicit tensor(I... args)
      : _shape<dim>(args...),
        data(memory<L>::template alloc<T>(_shape<dim>::total()),
            memory<L>::free) {}
  explicit tensor(const _shape<dim> &sizes)
      : _shape<dim>(sizes),
        data(memory<L>::template alloc<T>(_shape<dim>::total()),
            memory<L>::free) {}
  tensor(const tensor &in) = default;
  template <typename T2, class L2>
  explicit tensor(const tensor<T2, dim, L2> &in)
      : _shape<dim>(in),
        data(memory<L>::template alloc<T>(_shape<dim>::total()),
            memory<L>::free) {}
  tensor(T *ptr, const _shape<dim> &sizes)
      : _shape<dim>(sizes), data(ptr, [](void *) {}) {}
  tensor &operator=(const tensor &in) = default;
  size_t shape(size_t i) const {
    if constexpr (dim == 0)
      return 1;
    else
      return _shape<dim>::arr[i];
  }
  T *raw() const { return data.get(); }

  template <typename T2, class L2>
  void copy(const tensor<T2, dim, L2> &in) {
    senk::copy<L, L2>(raw(), in.raw(), _shape<dim>::total());
  }
};

template <typename T, uint16_t dim, class L>
struct view : public _shape<dim>, public has_val_t<T>, public has_loc_t<L> {
  constexpr static auto get_dim() { return dim; }
  using typename has_val_t<T>::val_t;
  using typename has_loc_t<L>::loc_t;
  val_t *raw;
  view() = delete;
  template <typename U, SENK_ENABULER(is_tensor_v<U>)>
  view(const U &in) : _shape<dim>(in), raw(in.raw()) {}
  SENK_LOC val_t eval(
      [[maybe_unused]] size_t i = 0, [[maybe_unused]] size_t j = 0) const {
    if constexpr (dim == 0)
      return raw[0];
    else if constexpr (dim == 1)
      return raw[i];
    else // (dim == 2)
      return raw[j * _shape<2>::arr[0] + i];
  }
  SENK_LOC size_t shape(size_t i) const {
    if constexpr (dim == 0)
      return 1;
    else
      return _shape<dim>::arr[i];
  }

private:
  view(val_t *ptr) : raw(ptr) {}
  friend struct reducer<loc_t>;
};

template <typename T, class L>
using scalar_view = view<T, 0, L>;
template <typename T, class L>
using vector_view = view<T, 1, L>;
template <typename T, class L>
using matrix_view = view<T, 2, L>;

template <typename T, class L>
struct is_scalar_expr<scalar_view<T, L>> : std::true_type {};
template <typename T, class L>
struct is_vector_expr<vector_view<T, L>> : std::true_type {};
template <typename T, class L>
struct is_matrix_expr<matrix_view<T, L>> : std::true_type {};

} // namespace impl

template <typename T, class L>
struct scalar : impl::tensor<T, 0, L> {
  using view_t = impl::scalar_view<T, L>;
  using typename impl::tensor<T, 0, L>::val_t;
  using typename impl::tensor<T, 0, L>::loc_t;
  using impl::tensor<T, 0, L>::shape;
  using impl::tensor<T, 0, L>::raw;
  using impl::tensor<T, 0, L>::data;

  scalar() : impl::tensor<T, 0, L>() {}
  scalar(const scalar &in) = default;
  template <typename T2, class L2>
  explicit scalar(const scalar<T2, L2> &in) : impl::tensor<T, 0, L>(in) {
    copy(in);
  }
  explicit scalar(T *ptr, impl::_shape<0>) : impl::tensor<T, 0, L>(ptr, {}){};
  scalar &operator=(const scalar &rhs) = default;
  val_t &operator[]([[maybe_unused]] size_t i) const { return data.get()[0]; }
  view_t view() const { return *this; }

#define KMM_SCAL_ASSIGN(COND, OP, RHS, INIT)                                   \
  template <class E>                                                           \
  SENK_RET(COND, scalar &)                                                     \
  operator OP(E && e) {                                                        \
    auto p = raw();                                                            \
    INIT;                                                                      \
    kernel<loc_t>::single(                                                     \
        [=] SENK_LOC() { p[0] OP static_cast<val_t>(RHS); });                  \
    return *this;                                                              \
  }
#define KMM_DEF_SCAL_ASSIGN_ARITH(COND, RHS, INIT)                             \
  KMM_SCAL_ASSIGN(COND, +=, RHS, INIT)                                         \
  KMM_SCAL_ASSIGN(COND, -=, RHS, INIT)                                         \
  KMM_SCAL_ASSIGN(COND, *=, RHS, INIT)                                         \
  KMM_SCAL_ASSIGN(COND, /=, RHS, INIT)
#define KMM_DEF_SCAL_ASSIGN(COND, RHS, INIT)                                   \
  KMM_DEF_SCAL_ASSIGN_ARITH(COND, RHS, INIT)                                   \
  KMM_SCAL_ASSIGN(COND, =, RHS, INIT)
#define KMM_SAME_LOC SENK_IS_SAME_V(typename std::decay_t<E>::loc_t, loc_t)
#define KMM_ARITH impl::is_arith_v<std::decay_t<E>>
#define KMM_SCAL (impl::is_scalar_v<std::decay_t<E>> && KMM_SAME_LOC)
#define KMM_SCAL_EXPR (impl::is_scalar_expr_v<std::decay_t<E>> && KMM_SAME_LOC)
  KMM_DEF_SCAL_ASSIGN(KMM_ARITH, e, )
  KMM_DEF_SCAL_ASSIGN(KMM_SCAL_EXPR, e.eval(0), )
  KMM_DEF_SCAL_ASSIGN_ARITH(KMM_SCAL, in_p[0], auto in_p = e.raw())
  KMM_SCAL_ASSIGN((KMM_SCAL && !SENK_IS_SAME_V(scalar, std::decay_t<E>)), =,
      in_p[0], auto in_p = e.raw())
#undef KMM_SCAL_EXPR
#undef KMM_SCAL
#undef KMM_ARITH
#undef KMM_SAME_LOC
#undef KMM_DEF_SCAL_ASSIGN
#undef KMM_DEF_SCAL_ASSIGN_ARITH
#undef KMM_SCAL_ASSIGN

  template <typename T2, class L2>
  SENK_RET(!SENK_IS_SAME_V(L2, loc_t), scalar &)
  operator=(const scalar<T2, L2> &rhs) {
    senk::copy<L, L2>(raw(), rhs.raw(), 1);
    return *this;
  }
  template <typename T2, typename L2>
  SENK_RET(!SENK_IS_SAME_V(L2, loc_t), scalar &)
  operator=(const impl::scalar_view<T2, L2> &rhs) {
    senk::copy<L, L2>(raw(), rhs.raw, 1);
    // communicator<L2, L>::to(raw(), rhs.raw, 1);
    return *this;
  }

#define PST_SELF_FUNCTION(FUNC, PROC)                                          \
  scalar &FUNC() {                                                             \
    auto p = raw();                                                            \
    kernel<loc_t>::single([=] SENK_LOC() { PROC; });                           \
    return *this;                                                              \
  }
  PST_SELF_FUNCTION(inv, p[0] = static_cast<val_t>(1.) / p[0])
  PST_SELF_FUNCTION(neg, p[0] = -p[0])
  PST_SELF_FUNCTION(abs, p[0] = senk::abs(p[0]))
  PST_SELF_FUNCTION(sqrt, p[0] = senk::sqrt(p[0]))
  PST_SELF_FUNCTION(conj, p[0] = senk::conj(p[0]))
#undef PST_SELF_FUNCTION

  template <typename T2>
  scalar &fill(T2 val) {
    return operator=(val);
  }

  template <class F>
  auto apply(F unary) {
    return impl::APPS<view_t, F>(this->view(), unary);
  }

  template <typename T2, class L2>
  scalar &copy(const scalar<T2, L2> &in) {
    impl::tensor<T, 0, L>::copy(in);
    return *this;
  }
};

template <typename T, class L>
struct vector : impl::tensor<T, 1, L> {
  using view_t = impl::vector_view<T, L>;
  using typename impl::tensor<T, 1, L>::val_t;
  using typename impl::tensor<T, 1, L>::loc_t;
  using impl::tensor<T, 1, L>::shape;
  using impl::tensor<T, 1, L>::raw;
  using impl::tensor<T, 1, L>::data;

  explicit vector(size_t size) : impl::tensor<T, 1, L>(size) {}
  vector(const vector &in) = default;
  template <typename T2, class L2>
  explicit vector(const vector<T2, L2> &in) : impl::tensor<T, 1, L>(in) {
    copy(in);
  }
  vector(T *ptr, impl::_shape<1> size) : impl::tensor<T, 1, L>(ptr, size){};
  vector &operator=(const vector &rhs) = default;
  auto operator()(size_t i) const {
    return scalar<T, L>(raw() + i, impl::_shape<0>(*this));
  }
  val_t &operator[](size_t i) const { return data.get()[i]; }
  view_t view() const { return *this; }
  vector slice(int len) {
    auto t = impl::_shape<1>(len);
    return vector(raw(), t);
  }

#define KMM_VECTOR_ASSIGN(COND, OP, RHS, INIT)                                 \
  template <class E>                                                           \
  SENK_RET(COND, vector &)                                                     \
  operator OP(E && e) {                                                        \
    auto p = raw();                                                            \
    INIT;                                                                      \
    kernel<loc_t>::parallel(shape(0),                                          \
        [=] SENK_LOC(size_t i) { p[i] OP static_cast<val_t>(RHS); });          \
    return *this;                                                              \
  }
#define KMM_DEF_VECTOR_ASSIGN_ARITH(COND, RHS, INIT)                           \
  KMM_VECTOR_ASSIGN(COND, +=, RHS, INIT)                                       \
  KMM_VECTOR_ASSIGN(COND, -=, RHS, INIT)                                       \
  KMM_VECTOR_ASSIGN(COND, *=, RHS, INIT)                                       \
  KMM_VECTOR_ASSIGN(COND, /=, RHS, INIT)
#define KMM_DEF_VECTOR_ASSIGN(COND, RHS, INIT)                                 \
  KMM_DEF_VECTOR_ASSIGN_ARITH(COND, RHS, INIT)                                 \
  KMM_VECTOR_ASSIGN(COND, =, RHS, INIT)
#define KMM_SAME_LOC SENK_IS_SAME_V(typename std::decay_t<E>::loc_t, loc_t)
#define KMM_ARITH impl::is_arith_v<std::decay_t<E>>
#define KMM_SCAL (impl::is_scalar_v<std::decay_t<E>> && KMM_SAME_LOC)
#define KMM_SCAL_EXPR (impl::is_scalar_expr_v<std::decay_t<E>> && KMM_SAME_LOC)
#define KMM_VEC (impl::is_vector_v<std::decay_t<E>> && KMM_SAME_LOC)
#define KMM_VEC_EXPR (impl::is_vector_expr_v<std::decay_t<E>> && KMM_SAME_LOC)
  KMM_DEF_VECTOR_ASSIGN(KMM_ARITH, e, )
  KMM_DEF_VECTOR_ASSIGN(KMM_SCAL_EXPR, e.eval(), )
  KMM_DEF_VECTOR_ASSIGN(KMM_SCAL, in_p[0], auto in_p = e.raw())
  KMM_DEF_VECTOR_ASSIGN(KMM_VEC_EXPR, e.eval(i), )
  KMM_DEF_VECTOR_ASSIGN_ARITH(KMM_VEC, in_p[i], auto in_p = e.raw())
  KMM_VECTOR_ASSIGN((KMM_VEC && !SENK_IS_SAME_V(vector, std::decay_t<E>)), =,
      in_p[i], auto in_p = e.raw())
#undef KMM_VEC_EXPR
#undef KMM_VEC
#undef KMM_SCAL_EXPR
#undef KMM_SCAL
#undef KMM_ARITH
#undef KMM_SAME_LOC
#undef KMM_DEF_VECTOR_ASSIGN
#undef KMM_DEF_VECTOR_ASSIGN_ARITH
#undef KMM_VECTOR_ASSIGN

#define PST_SELF_FUNCTION(FUNC, PROC)                                          \
  vector &FUNC() {                                                             \
    auto p = raw();                                                            \
    kernel<loc_t>::parallel(shape(0), [=] SENK_LOC(size_t i) { PROC; });       \
    return *this;                                                              \
  }
  PST_SELF_FUNCTION(inv, p[i] = static_cast<val_t>(1.) / p[i])
  PST_SELF_FUNCTION(neg, p[i] = -p[i])
  PST_SELF_FUNCTION(abs, p[i] = senk::abs(p[i]))
  PST_SELF_FUNCTION(sqrt, p[i] = senk::sqrt(p[i]))
  PST_SELF_FUNCTION(conj, p[i] = senk::conj(p[i]))
#undef PST_SELF_FUNCTION

  template <typename T2>
  vector &fill(T2 val) {
    return operator=(val);
  }

  template <class F>
  auto apply(F unary) {
    return impl::APPV<view_t, F>(this->view(), unary);
  }

  template <typename T2>
  auto as() const {
    return impl::ASV<view_t, T2>(this->view());
  }

  vector &iota(int s) {
    auto p = raw();
    kernel<loc_t>::parallel(shape(0), [=] SENK_LOC(size_t i) { p[i] = s + i; });
    return *this;
  }

  template <typename RG>
  vector &random(RG engine, int l, int r) {
    std::uniform_real_distribution<double> dist1(l, r);
    auto size = this->total();
    const auto s = 4096;
    T buff[s];
    for (size_t c = 0; c < size; c += s) {
      auto t_s = (c + s < size) ? s : size - c;
      for (size_t i = 0; i < t_s; i++)
        buff[i] = dist1(engine);
      communicator<host, L>::to(raw() + c, buff, t_s);
    }
    return *this;
  }

  template <typename T2, class L2>
  vector &copy(const vector<T2, L2> &in) {
    impl::tensor<T, 1, L>::copy(in);
    return *this;
  }

  vector &copy(T *in) {
    communicator<L, L>::to(raw(), in, shape(0));
    return *this;
  }
};

template <typename T, class L>
struct matrix : impl::tensor<T, 2, L> {
  using view_t = impl::matrix_view<T, L>;
  using typename impl::tensor<T, 2, L>::val_t;
  using typename impl::tensor<T, 2, L>::loc_t;
  using impl::tensor<T, 2, L>::shape;
  using impl::tensor<T, 2, L>::raw;
  using impl::tensor<T, 2, L>::data;

  matrix(const std::string &filename)
      : matrix(io::readmm_as_dense<double>(filename)) {}
  explicit matrix(size_t n, size_t m) : impl::tensor<T, 2, L>(n, m) {}
  matrix(const matrix &in) = default;
  template <typename T2, class L2>
  explicit matrix(const matrix<T2, L2> &in) : impl::tensor<T, 2, L>(in) {
    copy(in);
  }
  matrix(T *ptr, impl::_shape<2> size) : impl::tensor<T, 2, L>(ptr, size){};
  auto operator()(size_t col) const {
    return vector<T, L>(raw() + this->arr[0] * col, impl::_shape<1>(*this));
  }
  auto operator()(size_t row, size_t col) const {
    return scalar<T, L>(raw() + this->arr[0] * col + row, impl::_shape<0>());
  }
  val_t &operator[](size_t i) const { return data.get()[i]; }
  view_t view() const { return *this; }

private:
  template <typename T2>
  matrix(const io::Dense<T2> &d) : impl::tensor<T, 2, L>(d.nrows, d.ncols) {
    senk::copy<L, host>(raw(), d.val, d.nrows * d.ncols);
  }
};

} // namespace senk

#define PST_DEFINE_UNARY(name, op)                                             \
  template <typename L,                                                        \
      SENK_ENABULER(senk::impl::is_tensor_v<std::decay_t<L>>)>                 \
  name<typename std::decay_t<L>::view_t> operator op(L && v1) {                \
    return name<typename std::decay_t<L>::view_t>(                             \
        std::forward<typename std::decay_t<L>::view_t>(v1.view()));            \
  }                                                                            \
  template <typename L,                                                        \
      SENK_ENABULER(!senk::impl::is_tensor_v<std::decay_t<L>>)>                \
  name<std::decay_t<L>> operator op(L && v1) {                                 \
    return name<std::decay_t<L>>(std::forward<std::decay_t<L>>(v1));           \
  }
#define PST_DEFINE_BINARY_CASE(                                                \
    name, op, lcond, rcond, ltype, rtype, lexpr, rexpr)                        \
  template <typename L, typename R,                                            \
      SENK_ENABULER(lcond senk::impl::is_tensor_v<std::decay_t<L>>),           \
      SENK_ENABULER(rcond senk::impl::is_tensor_v<std::decay_t<R>>)>           \
  name<ltype, rtype> operator op(L && v1, R && v2) {                           \
    return name<ltype, rtype>(                                                 \
        std::forward<ltype>(lexpr), std::forward<rtype>(rexpr));               \
  }
#define PST_DEFINE_BINARY(name, op)                                            \
  PST_DEFINE_BINARY_CASE(name, op, , , typename std::decay_t<L>::view_t,       \
      typename std::decay_t<R>::view_t, v1.view(), v2.view())                  \
  PST_DEFINE_BINARY_CASE(name, op, !, , std::decay_t<L>,                       \
      typename std::decay_t<R>::view_t, v1, v2.view())                         \
  PST_DEFINE_BINARY_CASE(name, op, , !, typename std::decay_t<L>::view_t,      \
      std::decay_t<R>, v1.view(), v2)                                          \
  PST_DEFINE_BINARY_CASE(                                                      \
      name, op, !, !, std::decay_t<L>, std::decay_t<R>, v1, v2)

PST_DEFINE_UNARY(senk::impl::NEGS, -);
PST_DEFINE_UNARY(senk::impl::NEGV, -);
PST_DEFINE_UNARY(senk::impl::NEGM, -);

PST_DEFINE_BINARY(senk::impl::ADDSS, +);
PST_DEFINE_BINARY(senk::impl::SUBSS, -);
PST_DEFINE_BINARY(senk::impl::MULSS, *);
PST_DEFINE_BINARY(senk::impl::DIVSS, /);

PST_DEFINE_BINARY(senk::impl::ADDVV, +);
PST_DEFINE_BINARY(senk::impl::SUBVV, -);
PST_DEFINE_BINARY(senk::impl::MULVV, *);
PST_DEFINE_BINARY(senk::impl::DIVVV, /);

PST_DEFINE_BINARY(senk::impl::MULVS, *);
PST_DEFINE_BINARY(senk::impl::MULSV, *);
PST_DEFINE_BINARY(senk::impl::DIVVS, /);

PST_DEFINE_BINARY(senk::impl::MULMV, *);

#undef PST_DEFINE_BINARY
#undef PST_DEFINE_BINARY_CASE
#undef PST_DEFINE_UNARY

namespace senk {

template <class LL>
struct reducer : has_loc_t<LL> {
  using typename has_loc_t<LL>::loc_t;

  std::shared_ptr<uint64_t> receiver = nullptr;
  std::shared_ptr<uint64_t> buffer = nullptr;

  reducer(size_t size = 0)
      : receiver(
            memory<loc_t>::template alloc<uint64_t>(2), memory<loc_t>::free) {
    if constexpr (!std::is_same_v<loc_t, host>)
      buffer = std::shared_ptr<uint64_t>(
          memory<loc_t>::template alloc<uint64_t>(size), memory<loc_t>::free);
  }

#define _PST_REDUCE_DOT_RET                                                    \
  impl::scalar_view<                                                           \
      decltype(std::declval<typename std::decay_t<L>::val_t &>() *             \
               std::declval<typename std::decay_t<R>::val_t &>()),             \
      loc_t>
#define PST_REDUCE_DOT_RET(cond)                                               \
  template <class L, class R>                                                  \
  SENK_RET(cond, _PST_REDUCE_DOT_RET)
#define PST_DEFINE_DOT_BODY(ltype, rtype, lhs, rhs, linit, rinit)              \
  PST_REDUCE_DOT_RET((ltype<std::decay_t<L>> && rtype<std::decay_t<R>> &&      \
                      SENK_IS_SAME_V(typename std::decay_t<L>::loc_t,          \
                          typename std::decay_t<R>::loc_t)))                   \
  dot(L &&l, R &&r) const {                                                    \
    using val_t = decltype(std::declval<typename std::decay_t<L>::val_t &>() * \
                           std::declval<typename std::decay_t<R>::val_t &>()); \
    val_t *res = reinterpret_cast<val_t *>(receiver.get());                    \
    val_t *buf = (buffer) ? reinterpret_cast<val_t *>(buffer.get()) : nullptr; \
    linit;                                                                     \
    rinit;                                                                     \
    if constexpr (impl::is_complex_v<val_t>)                                   \
      kernel<loc_t>::reduce_add(                                               \
          l.shape(0), res, [=] SENK_LOC(size_t i) { return conj(lhs) * rhs; }, \
          buf);                                                                \
    else                                                                       \
      kernel<loc_t>::reduce_add(                                               \
          l.shape(0), res, [=] SENK_LOC(size_t i) { return lhs * rhs; }, buf); \
    return impl::scalar_view<val_t, loc_t>(res);                               \
  }

  PST_DEFINE_DOT_BODY(impl::is_vector_v, impl::is_vector_v, lp[i], rp[i],
      auto lp = l.raw(), auto rp = r.raw())
  PST_DEFINE_DOT_BODY(impl::is_vector_expr_v, impl::is_vector_v, l.eval(i),
      rp[i], , auto rp = r.raw())
  PST_DEFINE_DOT_BODY(impl::is_vector_v, impl::is_vector_expr_v, lp[i],
      r.eval(i), auto lp = l.raw(), )
  PST_DEFINE_DOT_BODY(
      impl::is_vector_expr_v, impl::is_vector_expr_v, l.eval(i), r.eval(i), , )

#undef PST_DEFINE_DOT_BODY
#undef PST_REDUCE_DOT_RET
#undef _PST_REDUCE_DOT_RET

#define _PST_REDUCE_NORM_RET                                                   \
  impl::scalar_view<                                                           \
      decltype(std::declval<typename std::decay_t<L>::val_t &>() *             \
               std::declval<typename std::decay_t<L>::val_t &>()),             \
      loc_t>
#define PST_REDUCE_NORM_RET(cond)                                              \
  template <class L>                                                           \
  SENK_RET(cond, _PST_REDUCE_NORM_RET)
#define PST_DEFINE_NORM_BODY(type, term, init)                                 \
  PST_REDUCE_NORM_RET(type<std::decay_t<L>>)                                   \
  norm(L &&l) const {                                                          \
    using val_t = decltype(std::declval<typename std::decay_t<L>::val_t &>() * \
                           std::declval<typename std::decay_t<L>::val_t &>()); \
    val_t *res = reinterpret_cast<val_t *>(receiver.get());                    \
    val_t *buf = (buffer) ? reinterpret_cast<val_t *>(buffer.get()) : nullptr; \
    init;                                                                      \
    if constexpr (impl::is_complex_v<val_t>)                                   \
      kernel<loc_t>::reduce_add(                                               \
          l.shape(0), res,                                                     \
          [=] SENK_LOC(size_t i) { return conj(term) * term; }, buf);          \
    else                                                                       \
      kernel<loc_t>::reduce_add(                                               \
          l.shape(0), res, [=] SENK_LOC(size_t i) { return term * term; },     \
          buf);                                                                \
    kernel<loc_t>::single([=] SENK_LOC() { res[0] = sqrt(res[0]); });          \
    return impl::scalar_view<val_t, loc_t>(res);                               \
  }
  PST_DEFINE_NORM_BODY(impl::is_vector_v, lp[i], auto lp = l.raw())
  PST_DEFINE_NORM_BODY(impl::is_vector_expr_v, l.eval(i), )

#undef PST_DEFINE_NORM_BODY
#undef PST_REDUCE_NORM_RET
#undef _PST_REDUCE_NORM_RET

#define _PST_REDUCE_ADD_RET                                                    \
  impl::scalar_view<                                                           \
      decltype(std::declval<typename std::decay_t<L>::val_t &>() *             \
               std::declval<typename std::decay_t<L>::val_t &>()),             \
      loc_t>
#define PST_REDUCE_ADD_RET(cond)                                               \
  template <class L>                                                           \
  SENK_RET(cond, _PST_REDUCE_ADD_RET)
#define PST_DEFINE_ADD_BODY(type, term, init)                                  \
  PST_REDUCE_ADD_RET(type<std::decay_t<L>>)                                    \
  add(L &&l) const {                                                           \
    using val_t = typename std::decay_t<L>::val_t;                             \
    val_t *res = reinterpret_cast<val_t *>(receiver.get());                    \
    val_t *buf = (buffer) ? reinterpret_cast<val_t *>(buffer.get()) : nullptr; \
    init;                                                                      \
    kernel<loc_t>::reduce_add(                                                 \
        l.shape(0), res, [=] SENK_LOC(size_t i) { return term; }, buf);        \
    return impl::scalar_view<val_t, loc_t>(res);                               \
  }
  PST_DEFINE_ADD_BODY(impl::is_vector_v, lp[i], auto lp = l.raw())
  PST_DEFINE_ADD_BODY(impl::is_vector_expr_v, l.eval(i), )

#undef PST_DEFINE_ADD_BODY
#undef PST_REDUCE_ADD_RET
#undef _PST_REDUCE_ADD_RET
};

} // namespace senk

#endif