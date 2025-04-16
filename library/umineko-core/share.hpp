#ifndef UMINEKO_CORE_SHARE_HPP
#define UMINEKO_CORE_SHARE_HPP

namespace kmm {

namespace impl {

class counter {
  void (*deleter)(void *);
  uint64_t cnt;

public:
  counter() = delete;
  template <class D> explicit counter(D d) : deleter(d), cnt(1) {}
  void inc() { ++cnt; }
  void clean(void *ptr) {
    --cnt;
    if (cnt == 0)
      if (ptr != nullptr)
        deleter(ptr);
  }
};

} // namespace impl

template <typename T> struct hd_ptr {
  T *ptr = nullptr;
  hd_ptr() = delete;
  template <typename D>
  hd_ptr(T *_ptr, D func) : ptr(_ptr), cntr(new impl::counter(func)) {}
  hd_ptr(const hd_ptr &in) : ptr(in.ptr), cntr(in.cntr) {
    if (in.ptr != nullptr)
      cntr->inc();
  }
  hd_ptr &operator=(const hd_ptr &in) {
    if (this == &in)
      return *this;
    cntr->clean((void*)ptr);
    ptr = in.ptr;
    cntr = in.cntr;
    if (ptr != nullptr)
      cntr->inc();
    return *this;
  }
  hd_ptr(hd_ptr &&in) noexcept : ptr(in.ptr), cntr(in.cntr) {
    in.ptr = nullptr;
    in.cntr = nullptr;
  }
  T *operator->() const noexcept { return ptr; }
  ~hd_ptr() { cntr->clean((void*)ptr); }

private:
  impl::counter *cntr;
};

} // namespace kmm

#endif // UMINEKO_CORE_SHARE_HPP
