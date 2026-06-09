#ifndef SENK_TRSV_CSR_HPP
#define SENK_TRSV_CSR_HPP

#include "senk/models.hpp"

#include "senk/matrix/base.hpp"
#include "senk/matrix/csr.hpp"
#include "senk/trsv/base.hpp"

#if __has_include(<cusparse.h>)
#define DISABLE_CUSPARSE_DEPRECATED
#include <cusparse.h>
#endif

namespace senk {

namespace impl {

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_direct(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out);

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_level(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const idx_t nlevels, const idx_t *lptr, const in_t *in,
    out_t *out);

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_partition(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const idx_t npart, const idx_t *pptr,
    const in_t *in, out_t *out);

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_baj(const int32_t iter, const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out);

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_jacobi(const int32_t iter, const idx_t nrows,
    const val_t *val, const idx_t *col, const srl_t *rptr, const in_t *in,
    out_t *out, out_t *tmp);

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_inverse(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const in_t *in, out_t *out);

} // namespace impl

namespace trsv {

template <typename T, class L, typename I, typename S, impl::form form,
    class strat>
struct CSR : private senk::CSR<T, L, I, S>,
             public has_params<strat>,
             public model::is_operator<CSR<T, L, I, S, form, strat>>,
             public model::is_invertible<CSR<T, L, I, S, form, strat>> {
  using Base = senk::CSR<T, L, I, S>;
  using Base::attr;
  using Base::col;
  using Base::rptr;
  using Base::val;
  using typename Base::idx_t;
  using typename Base::loc_t;
  using typename Base::val_t;
  using model::is_operator<CSR>::apply;
  using model::is_operator<Base>::nrows;
  using model::is_operator<Base>::ncols;
  using typename has_params<strat>::Params;

  Params prm;

  template <typename T2>
  explicit CSR(const senk::CSR<T2, host, I, S> &in, Params prm = Params{})
      : Base([&] {
          if constexpr (form == impl::form::upper)
            return in.duplicate_rotate180().inverse_last();
          else if constexpr (form == impl::form::lower)
            return in.duplicate_val().inverse_last();
          else
            return in;
        }()),
        prm(prm),
        jacobi_buffer(
            (std::is_same_v<strat, strategy::jacobi>) ? in.nrows() : 1) {}

private:
  vector<uint64_t, loc_t> jacobi_buffer;

  template <typename in_t, typename out_t>
  void apply_impl(const vector<in_t, L> &in, vector<out_t, L> out) const {
    if constexpr (std::is_same_v<strat, strategy::direct>) {
      impl::trsv_csr_apply_direct<loc_t, form>(
          nrows(), val.raw(), col.raw(), rptr.raw(), in.raw(), out.raw());
    } else if constexpr (std::is_same_v<strat, strategy::level>) {
      impl::trsv_csr_apply_level<loc_t, form>(nrows(), val.raw(), col.raw(),
          rptr.raw(), (idx_t)attr.segm.shape(0) - 1, attr.segm.raw(), in.raw(),
          out.raw());
    } else if constexpr (std::is_same_v<strat, strategy::partition>) {
      impl::trsv_csr_apply_partition<loc_t, form>(nrows(), val.raw(), col.raw(),
          rptr.raw(), (idx_t)attr.segm.shape(0) - 1, attr.segm.raw(), in.raw(),
          out.raw());
    } else if constexpr (std::is_same_v<strat, strategy::jacobi>) {
      auto tmp = reinterpret_cast<out_t *>(jacobi_buffer.raw());
      impl::trsv_csr_apply_jacobi<loc_t, form>(prm.max_iter, nrows(), val.raw(),
          col.raw(), rptr.raw(), in.raw(), out.raw(), tmp);
    } else if constexpr (std::is_same_v<strat, strategy::baj>) {
      impl::trsv_csr_apply_baj<loc_t, form>(prm.max_iter, nrows(), val.raw(),
          col.raw(), rptr.raw(), in.raw(), out.raw());
    }
  }

  template <typename in_t, typename out_t>
  void inverse_impl(const vector<in_t, L> &in, vector<out_t, L> out) const {
    impl::trsv_csr_inverse<loc_t, form>(
        nrows(), val.raw(), col.raw(), rptr.raw(), in.raw(), out.raw());
  }

  friend struct model::is_operator<CSR>;
  friend struct model::is_invertible<CSR>;
};

#if __has_include(<cusparse.h>)

template <typename T, typename I, typename S, impl::form form>
struct CSR<T, device, I, S, form, strategy::direct>
    : private senk::CSR<T, device>,
      public has_params<strategy::direct>,
      public model::is_operator<CSR<T, device, I, S, form, strategy::direct>>,
      public model::is_invertible<
          CSR<T, device, I, S, form, strategy::direct>> {
  using L = device;
  using strat = strategy::direct;
  using typename senk::CSR<T, L, I, S>::val_t;
  using typename senk::CSR<T, L, I, S>::loc_t;
  using typename senk::CSR<T, L, I, S>::idx_t;
  using senk::CSR<T, L, I, S>::val;
  using senk::CSR<T, L, I, S>::col;
  using senk::CSR<T, L, I, S>::rptr;
  using model::is_operator<CSR<T, L, I, S, form, strat>>::apply;
  using model::is_operator<senk::CSR<T, L, I, S>>::nrows;
  using model::is_operator<senk::CSR<T, L, I, S>>::ncols;
  using typename has_params<strat>::Params;

  Params prm;
  std::shared_ptr<cusparseHandle_t> handle;
  cusparseSpMatDescr_t matA;
  cusparseSpSVDescr_t spsvDescr;
  mutable cusparseDnVecDescr_t vecX, vecY;
  std::shared_ptr<void> dBuffer = nullptr;
  size_t bufferSize = 0;
  const T alpha = 1.0;

  template <typename TT>
  explicit CSR(const senk::CSR<TT, host, I, S> &in, Params prm = Params{})
      : senk::CSR<T, L, I, S>(in), prm(prm), jacobi_buffer(1) {
    cusparseCreateCsr(&matA, nrows(), ncols(), in.rptr[nrows()], rptr.raw(),
        col.raw(), val.raw(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    handle = std::shared_ptr<cusparseHandle_t>(new cusparseHandle_t);
    cusparseCreate(handle.get());
    cusparseSpSV_createDescr(&spsvDescr);
    if constexpr (form == impl::form::lower1 || form == impl::form::lower) {
      cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
      cusparseSpMatSetAttribute(
          matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
      cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
      cusparseSpMatSetAttribute(
          matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
    } else if constexpr (form == impl::form::upper) {
      cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_UPPER;
      cusparseSpMatSetAttribute(
          matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
      cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
      cusparseSpMatSetAttribute(
          matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
    }
    auto dummyX = vector<T, device>(ncols());
    auto dummyY = vector<T, device>(ncols());
    cusparseCreateDnVec(&vecX, ncols(), dummyX.raw(), CUDA_R_64F);
    cusparseCreateDnVec(&vecY, ncols(), dummyY.raw(), CUDA_R_64F);
    cusparseSpSV_bufferSize(handle.get()[0], CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
        spsvDescr, &bufferSize);
    auto ne = (bufferSize + sizeof(T) - 1) / sizeof(T);
    dBuffer = std::shared_ptr<void>(
        static_cast<void *>(memory<loc_t>::template alloc<T>(ne)),
        memory<loc_t>::free);
    cusparseSpSV_analysis(handle.get()[0], CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
        spsvDescr, dBuffer.get());
  }

private:
  vector<uint64_t, loc_t> jacobi_buffer;

  template <typename in_t, typename out_t>
  void apply_impl(const vector<in_t, L> &in, vector<out_t, L> out) const {
    void *p_in = in.raw();
    void *p_out = out.raw();
    cusparseDnVecSetValues(vecX, p_in);
    cusparseDnVecSetValues(vecY, p_out);
    cusparseSpSV_solve(handle.get()[0], CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
        spsvDescr);
  }

  template <typename in_t, typename out_t>
  void inverse_impl(const vector<in_t, L> &in, vector<out_t, L> out) const {
    impl::trsv_csr_inverse<loc_t, form>(
        nrows(), val.raw(), col.raw(), rptr.raw(), in.raw(), out.raw());
  }

  friend struct model::is_operator<CSR<T, L, I, S, form, strat>>;
  friend struct model::is_invertible<CSR<T, L, I, S, form, strat>>;
};

#endif

namespace l {
template <typename T, class L, class strat, typename I = index_t,
    typename S = serial_t>
using CSR = trsv::CSR<T, L, I, S, impl::form::lower1, strat>;
}
namespace ld {
template <typename T, class L, class strat, typename I = index_t,
    typename S = serial_t>
using CSR = trsv::CSR<T, L, I, S, impl::form::lower, strat>;
}
namespace du {
template <typename T, class L, class strat, typename I = index_t,
    typename S = serial_t>
using CSR = trsv::CSR<T, L, I, S, impl::form::upper, strat>;
}

} // namespace trsv

namespace impl {

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_direct(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<val_t>() * std::declval<out_t>());
  if constexpr (form == impl::form::lower1) {
    kernel<loc_t>::single([=]() mutable {
      for (idx_t i = 0; i < nrows; ++i) {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[i] = t;
      }
    });
  } else if constexpr (form == impl::form::lower) {
    kernel<loc_t>::single([=]() mutable {
      for (idx_t i = 0; i < nrows; ++i) {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[i] = t * val[j];
      }
    });
  } else if constexpr (form == impl::form::upper) {
    kernel<loc_t>::single([=]() mutable {
      for (idx_t i = 0; i < nrows; ++i) {
        auto t = static_cast<tmp_t>(in[nrows - 1 - i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[nrows - 1 - col[j]];
        out[nrows - 1 - i] = t * val[j];
      }
    });
  }
}

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_level(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const idx_t nlevels, const idx_t *lptr, const in_t *in,
    out_t *out) {
  using tmp_t = decltype(std::declval<val_t>() * std::declval<out_t>());
  for (int lid = 0; lid < nlevels; lid++) {
    const auto strt = lptr[lid];
    const auto size = lptr[lid + 1] - strt;
    if constexpr (form == impl::form::lower1) {
      kernel<loc_t>::parallel(size, [=] SENK_LOC(idx_t i) mutable {
        auto row = strt + i;
        auto t = static_cast<tmp_t>(in[row]);
        auto j = rptr[row];
        for (; j < rptr[row + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[row] = t;
      });
    } else if constexpr (form == impl::form::lower) {
      kernel<loc_t>::parallel(size, [=] SENK_LOC(idx_t i) mutable {
        auto row = strt + i;
        auto t = static_cast<tmp_t>(in[row]);
        auto j = rptr[row];
        for (; j < rptr[row + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[row] = t * val[j];
      });
    } else if constexpr (form == impl::form::upper) {
      kernel<loc_t>::parallel(size, [=] SENK_LOC(idx_t i) mutable {
        auto row = strt + i;
        auto t = static_cast<tmp_t>(in[nrows - 1 - row]);
        auto j = rptr[row];
        for (; j < rptr[row + 1] - 1; ++j)
          t -= val[j] * out[nrows - 1 - col[j]];
        out[nrows - 1 - row] = t * val[j];
      });
    }
  }
}

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_partition(const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const idx_t npart, const idx_t *pptr,
    const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<val_t>() * std::declval<out_t>());
  if constexpr (form == impl::form::lower1) {
    kernel<loc_t>::parallel(npart, [=] SENK_LOC(idx_t id) mutable {
      auto s = pptr[id];
      auto e = pptr[id + 1];
      for (idx_t i = s; i < e; ++i) {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[i] = t;
      }
    });
  } else if constexpr (form == impl::form::lower) {
    kernel<loc_t>::parallel(npart, [=] SENK_LOC(idx_t id) mutable {
      auto s = pptr[id];
      auto e = pptr[id + 1];
      for (idx_t i = s; i < e; ++i) {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[i] = t * val[j];
      }
    });
  } else if constexpr (form == impl::form::upper) {
    kernel<loc_t>::parallel(npart, [=] SENK_LOC(idx_t id) mutable {
      auto s = pptr[id];
      auto e = pptr[id + 1];
      for (idx_t i = s; i < e; ++i) {
        auto t = static_cast<tmp_t>(in[nrows - 1 - i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[nrows - 1 - col[j]];
        out[nrows - 1 - i] = t * val[j];
      }
    });
  }
}

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_baj(const int32_t iter, const idx_t nrows, const val_t *val,
    const idx_t *col, const srl_t *rptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<val_t>() * std::declval<out_t>());
  if constexpr (form == impl::form::lower1) {
    for (int32_t k = 0; k < iter; k++) {
      kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[i] = t;
      });
    }
  } else if constexpr (form == impl::form::lower) {
    for (int32_t k = 0; k < iter; k++) {
      kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        out[i] = t * val[j];
      });
    }
  } else if constexpr (form == impl::form::upper) {
    for (int32_t k = 0; k < iter; k++) {
      kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
        auto t = static_cast<tmp_t>(in[nrows - 1 - i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[nrows - 1 - col[j]];
        out[nrows - 1 - i] = t * val[j];
      });
    }
  }
}

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_apply_jacobi(const int32_t iter, const idx_t nrows,
    const val_t *val, const idx_t *col, const srl_t *rptr, const in_t *in,
    out_t *out, out_t *tmp) {
  using tmp_t = decltype(std::declval<val_t>() * std::declval<out_t>());
  if (iter & 1)
    std::swap(out, tmp);
  if constexpr (form == impl::form::lower1) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
      auto t = static_cast<tmp_t>(in[i]);
      auto j = rptr[i];
      for (; j < rptr[i + 1] - 1; ++j)
        t -= val[j] * static_cast<tmp_t>(in[col[j]]);
      tmp[i] = t;
    });
    for (int32_t k = 1; k < iter; k++) {
      kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * tmp[col[j]];
        out[i] = t;
      });
      std::swap(out, tmp);
    }
  } else if constexpr (form == impl::form::lower) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
      out[i] = val[rptr[i + 1] - 1] * in[i];
    });
    for (int32_t k = 0; k < iter; k++) {
      kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
        auto t = static_cast<tmp_t>(in[i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[col[j]];
        tmp[i] = t * val[j];
      });
      std::swap(out, tmp);
    }
  } else if constexpr (form == impl::form::upper) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
      out[nrows - 1 - i] = val[rptr[i + 1] - 1] * in[nrows - 1 - i];
    });
    for (int32_t k = 0; k < iter; k++) {
      kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
        auto t = static_cast<tmp_t>(in[nrows - 1 - i]);
        auto j = rptr[i];
        for (; j < rptr[i + 1] - 1; ++j)
          t -= val[j] * out[nrows - 1 - col[j]];
        tmp[nrows - 1 - i] = t * val[j];
      });
      std::swap(out, tmp);
    }
  }
}

template <typename loc_t, impl::form form, typename val_t, typename idx_t,
    typename srl_t, typename in_t, typename out_t>
void trsv_csr_inverse(const idx_t nrows, const val_t *val, const idx_t *col,
    const srl_t *rptr, const in_t *in, out_t *out) {
  using tmp_t = decltype(std::declval<in_t>() * std::declval<val_t>());
  if constexpr (form == impl::form::lower1) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
      auto t = static_cast<tmp_t>(0.);
      auto j = rptr[i];
      for (; j < rptr[i + 1]; ++j)
        t += val[j] * in[col[j]];
      out[i] = t;
    });
  } else if constexpr (form == impl::form::lower) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
      auto t = static_cast<tmp_t>(0.);
      auto j = rptr[i];
      for (; j < rptr[i + 1] - 1; ++j)
        t += val[j] * in[col[j]];
      out[i] = t + in[col[j]] / val[j];
    });
  } else if constexpr (form == impl::form::upper) {
    kernel<loc_t>::parallel(nrows, [=] SENK_LOC(idx_t i) mutable {
      auto t = static_cast<tmp_t>(0.);
      auto j = rptr[i];
      for (; j < rptr[i + 1] - 1; ++j)
        t += val[j] * in[nrows - 1 - col[j]];
      out[nrows - 1 - i] = t + in[nrows - 1 - col[j]] / val[j];
    });
  }
}

} // namespace impl

} // namespace senk

#endif