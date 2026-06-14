#ifndef SENK_CORE_IO_HPP
#define SENK_CORE_IO_HPP

#include <fstream>
#include <iostream>
#include <tuple>

#include "senk/core/memory.hpp"
#include "senk/core/sort.hpp"
// #include "tensor.hpp"

namespace senk::io {

template <typename T>
struct COO {
  int64_t *row;
  int64_t *col;
  T *val;
  int64_t nrows, ncols, nnzs;
  bool is_symmetric;
};

template <typename T>
struct CSR {
  int64_t *rptr;
  int64_t *col;
  T *val;
  int64_t nrows, ncols, nnzs;
  bool is_symmetric;
};

template <typename T>
struct Dense {
  T *val;
  int64_t nrows, ncols;
};

namespace impl {

inline std::tuple<bool, bool, bool> read_header(std::ifstream &file);
inline std::tuple<bool, bool, bool> read_header(char *&ptr);
inline char *file_to_chars(std::string path);
template <typename T>
inline T read_and_move(char *&ptr) {
  char *end;
  T res{};
  res = static_cast<T>(std::strtod(ptr, &end));
  ptr = end + 1;
  return res;
}

} // namespace impl

template <typename T>
COO<T> readmm_as_coo(const std::string &filename) {
  auto file = std::ifstream(filename, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "File Not Found" << std::endl;
    exit(1);
  }

  auto [_, is_binary, is_symmetric] = impl::read_header(file);
  int64_t nrows, ncols, nlines;
  file >> nrows >> ncols >> nlines;

  auto val = (is_symmetric) ? new T[nlines * 2] : new T[nlines];
  auto row = (is_symmetric) ? new int64_t[nlines * 2] : new int64_t[nlines];
  auto col = (is_symmetric) ? new int64_t[nlines * 2] : new int64_t[nlines];

  int64_t nnz = 0;
  for (int64_t i = 0; i < nlines; i++, nnz++) {
    double t_val;
    if (is_binary) {
      file >> row[nnz] >> col[nnz];
      t_val = 1.;
    } else
      file >> row[nnz] >> col[nnz] >> t_val;
    if (t_val == 0.) {
      nnz--;
      continue;
    }
    val[nnz] = static_cast<T>(t_val);
    row[nnz]--;
    col[nnz]--;
    if (is_symmetric && (row[nnz] != col[nnz])) {
      row[nnz + 1] = col[nnz];
      col[nnz + 1] = row[nnz];
      val[nnz + 1] = val[nnz];
      nnz++;
    }
  }

  return COO<T>{row, col, val, nrows, ncols, nnz, is_symmetric};
}

template <typename T>
COO<T> fast_readmm_as_coo(const std::string &filename) {
  auto contents = impl::file_to_chars(filename);
  if (!contents) {
    std::cerr << "File Not Found" << std::endl;
    exit(1);
  }
  auto ptr = contents;
  auto [_, is_binary, is_symmetric] = impl::read_header(ptr);
  auto nrows = impl::read_and_move<int64_t>(ptr);
  auto ncols = impl::read_and_move<int64_t>(ptr);
  auto nlines = impl::read_and_move<int64_t>(ptr);

  auto val = (is_symmetric) ? new T[nlines * 2] : new T[nlines];
  auto row = (is_symmetric) ? new int64_t[nlines * 2] : new int64_t[nlines];
  auto col = (is_symmetric) ? new int64_t[nlines * 2] : new int64_t[nlines];

  int64_t nnz = 0;
  for (int64_t i = 0; i < nlines; i++, nnz++) {
    double t_val;
    row[nnz] = impl::read_and_move<int64_t>(ptr);
    col[nnz] = impl::read_and_move<int64_t>(ptr);
    if (is_binary) {
      t_val = 1.;
    } else {
      t_val = impl::read_and_move<double>(ptr);
    }
    if (t_val == 0.) {
      nnz--;
      continue;
    }
    val[nnz] = static_cast<T>(t_val);
    row[nnz]--;
    col[nnz]--;
    if (is_symmetric && (row[nnz] != col[nnz])) {
      row[nnz + 1] = col[nnz];
      col[nnz + 1] = row[nnz];
      val[nnz + 1] = val[nnz];
      nnz++;
    }
  }

  delete[] contents;
  return COO<T>{row, col, val, nrows, ncols, nnz, is_symmetric};
}

template <typename T>
CSR<T> readmm_as_csr(const std::string &filename) {
  // auto coo = readmm_as_coo<T>(filename);
  auto coo = fast_readmm_as_coo<T>(filename);

  auto val = new T[coo.nnzs];
  auto cidx = new int64_t[coo.nnzs];
  auto rptr = new int64_t[coo.nrows + 1]();
  auto _offset = new int64_t[coo.nrows]();

  for (int64_t i = 0; i < coo.nnzs; i++) {
    rptr[coo.row[i] + 1]++;
  }
  for (int64_t i = 0; i < coo.nrows; i++) {
    rptr[i + 1] += rptr[i];
  }
  for (int64_t i = 0; i < coo.nnzs; i++) {
    auto offset = _offset[coo.row[i]]++;
    val[rptr[coo.row[i]] + offset] = coo.val[i];
    cidx[rptr[coo.row[i]] + offset] = coo.col[i];
  }

#pragma omp parallel for
  for (int64_t i = 0; i < coo.nrows; i++) {
    sort::pack_sort<sort::order::asc>(rptr[i], rptr[i + 1], cidx, val);
  }

  delete[] coo.val;
  delete[] coo.row;
  delete[] coo.col;
  delete[] _offset;

  return CSR<T>{
      rptr, cidx, val, coo.nrows, coo.ncols, coo.nnzs, coo.is_symmetric};
}

template <typename T>
Dense<T> readmm_as_dense(const std::string &filename) {
  auto contents = impl::file_to_chars(filename);
  if (!contents) {
    std::cerr << "File Not Found" << std::endl;
    exit(1);
  }
  auto ptr = contents;
  [[maybe_unused]] auto [is_coordinate, is_binary, is_symmetric] =
      impl::read_header(ptr);
  auto nrows = impl::read_and_move<int64_t>(ptr);
  auto ncols = impl::read_and_move<int64_t>(ptr);

  auto res = Dense<T>{new T[nrows * ncols], nrows, ncols};

  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = 0; j < ncols; j++)
      res.val[j * nrows + i] = impl::read_and_move<T>(ptr);
  }

  // delete[] contents;
  memory<host>::free(contents);
  return res;
}

namespace impl {

inline std::tuple<bool, bool, bool> read_header(std::ifstream &file) {
  std::string dummy, array, type, sym;
  file >> dummy >> dummy >> array >> type >> sym;
  file.ignore(1, '\n');

  size_t pos;
  std::string line;
  while (pos = file.tellg(), std::getline(file, line), line[0] == '%') {
  }
  file.seekg(pos);
  return {std::string(array) == "coordinate", std::string(type) == "pattern",
      std::string(sym) == "symmetric"};
}

inline std::tuple<bool, bool, bool> read_header(char *&ptr) {
  char array[16], type[16], sym[16];
  if (char dum[20];
      sscanf(ptr, "%s %s %s %s %s", dum, dum, array, type, sym) != 5) {
    fprintf(stderr, "Unsupported File\n");
    exit(1);
  }
  bool is_coordinate = (std::string(array) == "symmetric");
  bool is_pattern = (std::string(type) == "pattern");
  bool is_symmetric = (std::string(sym) == "symmetric");

  while (*(ptr++) == '%') {
    while (*(ptr++) != '\n')
      ;
  }
  ptr--;
  return {is_coordinate, is_pattern, is_symmetric};
}

inline char *file_to_chars(std::string path) {
  auto file =
      std::ifstream(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open())
    return nullptr;
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  // auto *contents = new char[size];
  auto contents = memory<host>::alloc<char>(size);
  file.seekg(0, std::ios::beg);
  file.read(contents, size);
  file.close();
  return contents;
}

} // namespace impl

} // namespace senk::io

#endif // SENK_CORE_IO_HPP