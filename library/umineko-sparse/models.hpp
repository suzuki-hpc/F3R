#ifndef UMINEKO_SPARSE_MODELS_HPP
#define UMINEKO_SPARSE_MODELS_HPP

#include <array>
#include <functional>
#include <memory>

#include "umineko-core/tensor.hpp"

#define KMM_ENABULER(cond) std::enable_if_t<cond, std::nullptr_t> = nullptr
#define KMM_DEFINE_ALL_PAIRS_OF_2MACROS(macro)                                      \
  macro(double, double) macro(double, float) macro(float, float) macro(float, double)
#define KMM_DEFINE_ALL_PAIRS_OF_3MACROS(macro)                                      \
  macro(double, double, double) macro(double, double, float)                        \
      macro(double, float, double) macro(float, double, double)                     \
          macro(double, float, float) macro(float, double, float)                   \
              macro(float, float, double) macro(float, float, float)
#define KMM_DEFINE_OPERATE(in_t, out_t)                                             \
  void operate(const vec_t<in_t> &in, vec_t<out_t> out) const override {            \
    return object.operate(in, out);                                                 \
  }
#define KMM_DEFINE_COMP_RES(rhs_t, in_t, res_t)                                     \
  void compute_residual(const vec_t<rhs_t> &rhs, const vec_t<in_t> &in,             \
      vec_t<res_t> res) const override {                                            \
    return object.compute_residual(rhs, in, res);                                   \
  }
#define KMM_DEFINE_SOLVE(in_t, out_t)                                               \
  solve_res_t solve(const vec_t<in_t> &in, vec_t<out_t> out,                        \
      const std::function<bool(int, double)> &cond) const override {                \
    return object.solve(in, out, cond);                                             \
  }
#define KMM_DEFINE_SOLVE_VOID(in_t, out_t)                                          \
  void solve(const vec_t<in_t> &in, vec_t<out_t> out) const override {              \
    object.solve(in, out, nullptr);                                                 \
  }

namespace kmm {

template <typename, class> struct CSR;

template <typename> struct is_trsv {
  static constexpr bool value = false;
};
template <class T> inline constexpr bool is_trsv_v = is_trsv<T>::value;

template <class L> struct Operator {
  template <typename val_t> using vec_t = vector<val_t, L>;
  template <typename Obj, KMM_ENABULER(!is_trsv_v<Obj>)>
  Operator(const Obj obj) : op(std::make_shared<Model<Obj>>(obj)) {}
  template <typename Obj, KMM_ENABULER(is_trsv_v<Obj>)>
  Operator(const Obj obj) : op(std::make_shared<Model_from_TRSV<Obj>>(obj)) {}
  template <class F>
  Operator(const std::array<idx_t, 2> &shape, F func)
      : Operator(wrapper<F>(shape, func)) {}

  template <typename val1_t, typename val2_t>
  void operate(const vec_t<val1_t> &in, vec_t<val2_t> out) const {
    return op->operate(in, out);
  }
  [[nodiscard]] idx_t nrows() const { return op->nrows(); }
  [[nodiscard]] idx_t ncols() const { return op->ncols(); }

  struct Concept {
    virtual ~Concept() = default;
    virtual void operate(const vec_t<double> &in, vec_t<double> out) const = 0;
    virtual void operate(const vec_t<double> &in, vec_t<float> out) const = 0;
    virtual void operate(const vec_t<float> &in, vec_t<double> out) const = 0;
    virtual void operate(const vec_t<float> &in, vec_t<float> out) const = 0;
    [[nodiscard]] virtual idx_t nrows() const = 0;
    [[nodiscard]] virtual idx_t ncols() const = 0;
  };

  template <typename T> struct Model final : Concept {
    explicit Model(const T &t) : object(t) {}
    KMM_DEFINE_ALL_PAIRS_OF_2MACROS(KMM_DEFINE_OPERATE)
    [[nodiscard]] idx_t nrows() const override { return object.nrows(); }
    [[nodiscard]] idx_t ncols() const override { return object.ncols(); }

  private:
    T object;
  };

  template <typename T> struct Model_from_TRSV : Concept {
    explicit Model_from_TRSV(const T &t) : object(t) {}
#define DEF_OPERATE(in_t, out_t)                                                    \
  void operate(const vec_t<in_t> &in, vec_t<out_t> out) const override {            \
    return object.solve(in, out);                                                   \
  }
    KMM_DEFINE_ALL_PAIRS_OF_2MACROS(DEF_OPERATE)
#undef DEF_OPERATE
    [[nodiscard]] idx_t nrows() const override { return object.nrows(); }
    [[nodiscard]] idx_t ncols() const override { return object.ncols(); }

  private:
    T object;
  };

  template <class C>
  static std::function<Operator(const CSR<double, host> &)> factory() {
    return [](const CSR<double, host> &in) { return Operator(C(in)); };
  }
  template <class C>
  static std::function<Operator(const CSR<double, host> &)> factory(
      const typename C::params &p) {
    return [p](const CSR<double, host> &in) { return Operator(C(in, p)); };
  }
  template <class C>
  static std::function<Operator(const CSR<double, host> &)> factory(
      const typename C::params &p, const typename C::factories &f) {
    return [p, f](const CSR<double, host> &in) { return Operator(C(in, p, f)); };
  }

private:
  std::shared_ptr<const Concept> op;
  template <class F> struct wrapper {
    F func;
    std::array<idx_t, 2> shape;
    wrapper(const std::array<idx_t, 2> &shape, F func) : func(func), shape(shape) {}
    [[nodiscard]] idx_t nrows() const { return shape[0]; };
    [[nodiscard]] idx_t ncols() const { return shape[1]; };
    template <typename in_t, typename out_t>
    void operate(const in_t &in, out_t out) const {
      func(in, out);
    }
  };
};

template <class L> struct CoefficientMatrix {
  template <typename val_t> using vec_t = vector<val_t, L>;
  template <typename Obj>
  CoefficientMatrix(const Obj obj) : op(std::make_shared<Model<Obj>>(obj)) {}

  template <typename val1_t, typename val2_t>
  void operate(const vec_t<val1_t> &in, vec_t<val2_t> out) const {
    return op->operate(in, out);
  }
  template <typename rhs_t, typename in_t, typename res_t>
  void compute_residual(
      const vec_t<rhs_t> &rhs, const vec_t<in_t> &in, vec_t<res_t> res) const {
    return op->compute_residual(rhs, in, res);
  }
  [[nodiscard]] idx_t nrows() const { return op->nrows(); }
  [[nodiscard]] idx_t ncols() const { return op->ncols(); }

  struct Concept {
    virtual ~Concept() = default;
    virtual void operate(const vec_t<double> &, vec_t<double>) const = 0;
    virtual void operate(const vec_t<double> &, vec_t<float>) const = 0;
    virtual void operate(const vec_t<float> &, vec_t<double>) const = 0;
    virtual void operate(const vec_t<float> &, vec_t<float>) const = 0;
    virtual void compute_residual(
        const vec_t<double> &, const vec_t<double> &, vec_t<double>) const = 0;
    virtual void compute_residual(
        const vec_t<double> &, const vec_t<double> &, vec_t<float>) const = 0;
    virtual void compute_residual(
        const vec_t<double> &, const vec_t<float> &, vec_t<double>) const = 0;
    virtual void compute_residual(
        const vec_t<double> &, const vec_t<float> &, vec_t<float>) const = 0;
    virtual void compute_residual(
        const vec_t<float> &, const vec_t<double> &, vec_t<double>) const = 0;
    virtual void compute_residual(
        const vec_t<float> &, const vec_t<double> &, vec_t<float>) const = 0;
    virtual void compute_residual(
        const vec_t<float> &, const vec_t<float> &, vec_t<double>) const = 0;
    virtual void compute_residual(
        const vec_t<float> &, const vec_t<float> &, vec_t<float>) const = 0;
    [[nodiscard]] virtual idx_t nrows() const = 0;
    [[nodiscard]] virtual idx_t ncols() const = 0;
  };

  template <typename T> struct Model : Concept {
    explicit Model(const T &t) : object(t) {}
    KMM_DEFINE_ALL_PAIRS_OF_2MACROS(KMM_DEFINE_OPERATE)
    KMM_DEFINE_ALL_PAIRS_OF_3MACROS(KMM_DEFINE_COMP_RES)
    [[nodiscard]] idx_t nrows() const override { return object.nrows(); }
    [[nodiscard]] idx_t ncols() const override { return object.ncols(); }

  private:
    T object;
  };

  template <class C>
  static std::function<CoefficientMatrix(const CSR<double, host> &)> factory() {
    return [](const CSR<double, host> &in) { return CoefficientMatrix(C(in)); };
  }

private:
  std::shared_ptr<const Concept> op;
};

struct solve_res_t {
  bool is_solved;
  int32_t res_iter;
  double res_nrm2;
  void operator+=(const solve_res_t &in) {
    is_solved = is_solved || in.is_solved;
    res_iter = res_iter + in.res_iter;
    res_nrm2 = in.res_nrm2;
  }
};

template <class L> struct Solver {
  template <typename val_t> using vec_t = vector<val_t, L>;
  template <typename Obj>
  explicit Solver(const Obj obj) : op(std::make_shared<Model<Obj>>(obj)) {}

  template <typename val1_t, typename val2_t>
  solve_res_t solve(const vec_t<val1_t> &in, vec_t<val2_t> out,
      const std::function<bool(int, double)> &cond) const {
    return op->solve(in, out, cond);
  }
  template <typename val1_t, typename val2_t>
  void solve(const vec_t<val1_t> &in, vec_t<val2_t> out) const {
    op->solve(in, out, nullptr);
  }
  template <typename val1_t, typename val2_t>
  void operate(const vec_t<val1_t> &in, vec_t<val2_t> out) const {
    op->operate(in, out);
  }

  [[nodiscard]] idx_t nrows() const { return op->nrows(); }
  [[nodiscard]] idx_t ncols() const { return op->ncols(); }

  struct Concept {
    virtual ~Concept() = default;
    virtual solve_res_t solve(const vec_t<double> &, vec_t<double>,
        const std::function<bool(int, double)> &) const = 0;
    virtual solve_res_t solve(const vec_t<double> &, vec_t<float>,
        const std::function<bool(int, double)> &) const = 0;
    virtual solve_res_t solve(const vec_t<float> &, vec_t<double>,
        const std::function<bool(int, double)> &) const = 0;
    virtual solve_res_t solve(const vec_t<float> &, vec_t<float>,
        const std::function<bool(int, double)> &) const = 0;
    virtual void solve(const vec_t<double> &, vec_t<double>) const = 0;
    virtual void solve(const vec_t<double> &, vec_t<float>) const = 0;
    virtual void solve(const vec_t<float> &, vec_t<double>) const = 0;
    virtual void solve(const vec_t<float> &, vec_t<float>) const = 0;
    virtual void operate(const vec_t<double> &in, vec_t<double> out) const = 0;
    virtual void operate(const vec_t<double> &in, vec_t<float> out) const = 0;
    virtual void operate(const vec_t<float> &in, vec_t<double> out) const = 0;
    virtual void operate(const vec_t<float> &in, vec_t<float> out) const = 0;
    [[nodiscard]] virtual idx_t nrows() const = 0;
    [[nodiscard]] virtual idx_t ncols() const = 0;
  };

  template <typename T> struct Model : Concept {
    explicit Model(const T &t) : object(t) {}
    KMM_DEFINE_ALL_PAIRS_OF_2MACROS(KMM_DEFINE_SOLVE)
    KMM_DEFINE_ALL_PAIRS_OF_2MACROS(KMM_DEFINE_SOLVE_VOID)
    KMM_DEFINE_ALL_PAIRS_OF_2MACROS(KMM_DEFINE_OPERATE)
    [[nodiscard]] idx_t nrows() const override { return object.nrows(); }
    [[nodiscard]] idx_t ncols() const override { return object.ncols(); }

  private:
    T object;
  };

  template <class C, class AF, class MF>
  static std::function<Solver(const CSR<double, host> &)> factory(
      const typename C::params &p, AF af, MF mf) {
    return [p, af, mf](
               const CSR<double, host> &in) { return Solver(C(af(in), mf(in), p)); };
  }

private:
  std::shared_ptr<const Concept> op;
};

template <class L> class Smoother {
public:
  template <typename val_t> using vec_t = vector<val_t, L>;
  template <template <typename, class> typename Obj, typename T>
  explicit Smoother(Obj<T, L> &&obj) : op(std::make_shared<Model<Obj<T, L>>>(obj)) {}

  template <typename Obj>
  explicit Smoother(const Obj obj) : op(std::make_shared<Model<Obj>>(obj)) {}

  template <typename val1_t, typename val2_t>
  void smooth(const vec_t<val1_t> &in, vec_t<val2_t> out) const {
    return op->smooth(in, out);
  }
  template <typename val1_t, typename val2_t>
  void operate(const vec_t<val1_t> &in, vec_t<val2_t> out) const {
    return op->operate(in, out);
  }

  [[nodiscard]] idx_t nrows() const { return op->nrows(); }
  [[nodiscard]] idx_t ncols() const { return op->ncols(); }

private:
  struct Concept {
    virtual ~Concept() = default;
    virtual void smooth(const vec_t<double> &, vec_t<double>) const = 0;
    virtual void smooth(const vec_t<double> &, vec_t<float>) const = 0;
    virtual void smooth(const vec_t<float> &, vec_t<double>) const = 0;
    virtual void smooth(const vec_t<float> &, vec_t<float>) const = 0;

    virtual void operate(const vec_t<double> &in, vec_t<double> out) const = 0;
    virtual void operate(const vec_t<double> &in, vec_t<float> out) const = 0;
    virtual void operate(const vec_t<float> &in, vec_t<double> out) const = 0;
    virtual void operate(const vec_t<float> &in, vec_t<float> out) const = 0;

    [[nodiscard]] virtual idx_t nrows() const = 0;
    [[nodiscard]] virtual idx_t ncols() const = 0;
  };

  template <typename T> struct Model : Concept {
    explicit Model(const T &t) : object(t) {}
#define DEF_SMOOTH(in_t, out_t)                                                     \
  void smooth(const vec_t<in_t> &in, vec_t<out_t> out) const override {             \
    return object.smooth(in, out);                                                  \
  }                                                                                 \
  void operate(const vec_t<in_t> &in, vec_t<out_t> out) const override {            \
    return object.operate(in, out);                                                 \
  }
    DEF_SMOOTH(double, double)
    DEF_SMOOTH(double, float)
    DEF_SMOOTH(float, double)
    DEF_SMOOTH(float, float)
#undef DEF_SOLVE

    [[nodiscard]] idx_t nrows() const override { return object.nrows(); }
    [[nodiscard]] idx_t ncols() const override { return object.ncols(); }

  private:
    T object;
  };

  std::shared_ptr<const Concept> op;
};

} // namespace kmm

#undef KMM_DEFINE_SOLVE_VOID
#undef KMM_DEFINE_SOLVE
#undef KMM_DEFINE_COMP_RES
#undef KMM_DEFINE_OPERATE
#undef KMM_DEFINE_ALL_PAIRS_OF_3MACROS
#undef KMM_DEFINE_ALL_PAIRS_OF_2MACROS
#undef KMM_ENABULER

#endif