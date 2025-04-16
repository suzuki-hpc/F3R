#ifndef UMINEKO_SPARSE_MATRIX_IO_HPP
#define UMINEKO_SPARSE_MATRIX_IO_HPP

#include <charconv>
#include <fstream>
#include <functional>
#include <string>

#include "umineko-core/sort.hpp"
#include "umineko-core/tensor.hpp"

#include "umineko-sparse/matrix/base.hpp"

#include <iostream>

namespace kmm {

struct io {
  io() = delete;
  // using value_t = double;
  using index_t = int32_t;
  template <typename value_t> struct COO : spmat {
    vector<index_t, host> row, col;
    vector<value_t, host> val;

    explicit COO(const impl::shape &size, const idx_t nz,
        const vector<index_t, host> &_row, const vector<index_t, host> &_col,
        const vector<value_t, host> &_val)
        : spmat(size, nz), row(nnz()), col(nnz()), val(nnz()) {
      row.copy(_row);
      col.copy(_col);
      val.copy(_val);
    }
    explicit COO(const spmat &in, const vector<index_t, host> &_row,
        const vector<index_t, host> &_col, const vector<value_t, host> &_val)
        : spmat(in), row(nnz()), col(nnz()), val(nnz()) {
      row.copy(_row);
      col.copy(_col);
      val.copy(_val);
    }
  };
  template <typename value_t> struct CSR : spmat {
    vector<index_t, host> idx, rptr;
    vector<value_t, host> val;
    CSR(const impl::shape &size, const idx_t nz)
        : spmat(size, nz), idx(nz), rptr(size[0] + 1), val(nz) {}
    explicit CSR(const COO<value_t> &coo)
        : spmat(coo), idx(nnz()), rptr(nrows() + 1), val(nnz()) {
      const auto off = new index_t[nrows()]();
      for (index_t i = 0; i < nnz(); i++)
        rptr[coo.row[i] + 1]++;
      for (index_t i = 0; i < nrows(); i++)
        rptr[i + 1] += rptr[i];
      for (index_t i = 0; i < nnz(); i++) {
        const auto offset = off[coo.row[i]]++;
        val[rptr[coo.row[i]] + offset] = coo.val[i];
        idx[rptr[coo.row[i]] + offset] = coo.col[i];
      }
#pragma omp parallel for
      for (index_t i = 0; i < nrows(); i++)
        sort::quick<sort::order::asc>(rptr[i], rptr[i + 1], idx.raw(), val.raw());
      delete[] off;
    }
  };

  template <typename value_t> struct CSC : spmat {
    vector<index_t, host> idx, cptr;
    vector<value_t, host> val;

    explicit CSC(const COO<value_t> &coo)
        : spmat(coo), idx(nnz()), cptr(ncols() + 1), val(nnz()) {
      const auto off = new index_t[ncols()]();
      for (index_t i = 0; i < nnz(); i++)
        cptr[coo.col[i] + 1]++;
      for (index_t i = 0; i < ncols(); i++)
        cptr[i + 1] += cptr[i];
      for (index_t i = 0; i < nnz(); i++) {
        const auto offset = off[coo.col[i]]++;
        val[cptr[coo.col[i]] + offset] = coo.val[i];
        idx[cptr[coo.col[i]] + offset] = coo.row[i];
      }
#pragma omp parallel for
      for (index_t i = 0; i < ncols(); i++)
        sort::quick<sort::order::asc>(cptr[i], cptr[i + 1], idx.raw(), val.raw());
      delete[] off;
    }
  };

  struct mm {
    mm() = delete;
    static COO<double> read_matrix(const std::string &filename) {
      auto contents = _file_to_chars(filename);
      auto mat = _read_matrix_market<double>(contents);
      delete[] contents;
      return mat;
    }
    static COO<std::complex<double>> read_complex_matrix(
        const std::string &filename) {
      auto contents = _file_to_chars(filename);
      auto mat = _read_matrix_market<std::complex<double>>(contents);
      delete[] contents;
      return mat;
    }
    static std::pair<COO<double>, vector<double, host>> read_system(
        const std::string &filename,
        std::function<void(vector<double, host>)> init = nullptr) {
      auto contents = _file_to_chars(filename);
      auto mat = _read_matrix_market<double>(contents);
      if (contents)
        delete[] contents;
      std::string path_b = filename;
      path_b.insert(filename.length() - 4, "_b");
      auto contents_b = _file_to_chars(path_b);
      auto vec = [&] {
        if (contents_b != nullptr)
          return _read_matrix_market_b(contents_b);
        auto res = vector<double, host>(mat.nrows());
        init(res);
        return res;
      }();
      if (contents_b)
        delete[] contents_b;
      if (mat.nrows() != vec.size(0))
        exit(1);
      return {mat, vec};
    }

  private:
    template <typename value_t>
    static COO<value_t> _read_matrix_market(char *contents) {
      // check the format with the header line
      // e.g., %%MatrixMarket matrix coordinate real symmetric
      char type[16], symmetric[16];
      if (char dum[20];
          sscanf(contents, "%s %s %s %s %s", dum, dum, dum, type, symmetric) != 5)
        exit(1);
      bool is_symmetric = (std::string(symmetric) == "symmetric");
      if (std::is_same_v<value_t, std::complex<double>> &&
          std::string(type) != "complex") {
        std::cerr << "Mismatch between MM type and assumed data type" << std::endl;
        exit(1);
      }
      // skip comments starting with '%'
      char *ptr = contents;
      while (*(ptr++) == '%') {
        while (*(ptr++) != '\n')
          ;
      }
      ptr--;

      auto nrow = _read_num_and_move_ptr<index_t>(ptr);
      auto ncol = _read_num_and_move_ptr<index_t>(ptr);
      auto nlines = _read_num_and_move_ptr<index_t>(ptr);

      // allocate the three arrays
      // if symmetric, the size is doubled to store the implicit upper part
      auto val = (is_symmetric) ? vector<value_t, host>(nlines * 2)
                                : vector<value_t, host>(nlines);
      auto row = (is_symmetric) ? vector<index_t, host>(nlines * 2)
                                : vector<index_t, host>(nlines);
      auto col = (is_symmetric) ? vector<index_t, host>(nlines * 2)
                                : vector<index_t, host>(nlines);

      // read row, col, and val line by line
      auto nz = 0;
      for (index_t i = 0; i < nlines; i++, nz++) {
        row[nz] = _read_num_and_move_ptr<index_t>(ptr) - 1;
        col[nz] = _read_num_and_move_ptr<index_t>(ptr) - 1;
        if constexpr (std::is_same_v<value_t, std::complex<double>>)
          val[nz] = std::complex<double>(_read_num_and_move_ptr<double>(ptr),
              _read_num_and_move_ptr<double>(ptr));
        else
          val[nz] = _read_num_and_move_ptr<double>(ptr);
        // if symmetric and not diagonal, duplicate the element in the upper
        // triangular part
        if (val[nz] == 0.0 && col[nz] != row[nz]) {
          --nz;
          continue;
        }
        if (is_symmetric && (row[nz] != col[nz])) {
          row[nz + 1] = col[nz];
          col[nz + 1] = row[nz];
          val[nz + 1] = val[nz];
          nz++;
        }
      }
      spmat::Type t = spmat::Type::real;
      spmat::Sym s = (is_symmetric) ? spmat::Sym::symmetric : spmat::Sym::general;
      spmat::Form f = spmat::Form::square;
      spmat::Pattern p = spmat::Pattern::general;
      return COO{spmat{{nrow, ncol}, nz, t, s, f, p}, row, col, val};
    }
    static vector<double, host> _read_matrix_market_b(char *contents) {
      // Must be like %%MatrixMarket matrix array real general
      char array[16], type[16], symmetric[16];
      if (char dum[20];
          sscanf(contents, "%s %s %s %s %s", dum, dum, array, type, symmetric) != 5)
        exit(1);
      // skip comments starting with '%'
      char *ptr = contents;
      while (*(ptr++) == '%') {
        while (*(ptr++) != '\n')
          ;
      }
      ptr--;
      auto nrow = _read_num_and_move_ptr<index_t>(ptr);
      auto ncol = _read_num_and_move_ptr<index_t>(ptr);
      if (std::string(array) != "array" || std::string(symmetric) != "general" ||
          ncol != 1) {
        exit(1);
      }
      auto vec = vector<double, host>(nrow);
      for (index_t i = 0; i < nrow; i++) {
        vec[i] = _read_num_and_move_ptr<double>(ptr);
      }
      return vec;
    }
  };

  struct jsol {
    jsol() = delete;
    static std::pair<CSR<double>, vector<double, host>> read_system(
        const std::string &filename) {
      std::string path_index = filename;
      std::string path_coeff = filename;
      path_index.insert(filename.length() - 8, "index_");
      path_coeff.insert(filename.length() - 8, "coeff_");
      auto contents_index = _file_to_chars(path_index);
      auto mat = _read_matrix_index(contents_index);
      if (contents_index)
        delete[] contents_index;

      auto contents_coeff = _file_to_chars(path_coeff);
      auto vec = _read_matrix_coeff(contents_coeff, mat);
      if (contents_coeff)
        delete[] contents_coeff;

      if (mat.nrows() != vec.size(0))
        exit(1);
      return {mat, vec};
    }

  private:
    static CSR<double> _read_matrix_index(char *contents) {
      char *ptr = contents;
      while (*(ptr) == ' ' || *(ptr) == '\n')
        ptr++;
      auto nrow = _read_num_and_move_ptr<index_t>(ptr);
      while (*(ptr) == ' ' || *(ptr) == '\n')
        ptr++;
      auto nnz = _read_num_and_move_ptr<index_t>(ptr);
      while (*(ptr) == ' ' || *(ptr) == '\n')
        ptr++;
      _read_num_and_move_ptr<index_t>(ptr);
      while (*(ptr) == ' ' || *(ptr) == '\n')
        ptr++;
      _read_num_and_move_ptr<index_t>(ptr);
      auto csr = CSR<double>({nrow, nrow}, nnz);

      for (int i = 0; i < nnz; i++) {
        while (*(ptr) == ' ' || *(ptr) == '\n')
          ptr++;
        csr.idx[i] = _read_num_and_move_ptr<index_t>(ptr) - 1;
      }
      for (int i = 0; i < nrow; i++) {
        while (*(ptr) == ' ' || *(ptr) == '\n')
          ptr++;
        _read_num_and_move_ptr<index_t>(ptr);
      }
      for (int i = 0; i < nrow; i++) {
        while (*(ptr) == ' ' || *(ptr) == '\n')
          ptr++;
        csr.rptr[i] = _read_num_and_move_ptr<index_t>(ptr) - 1;
      }
      csr.rptr[nrow] = nnz;
      return csr;
    }
    static vector<double, host> _read_matrix_coeff(
        char *contents, CSR<double> &csr) {
      char *ptr = contents;
      auto vec = vector<double, host>(csr.nrows());
      for (int i = 0; i < csr.nrows(); i++) {
        while (*(ptr) == ' ' || *(ptr) == '\n')
          ptr++;
        csr.val[i] = _read_num_and_move_ptr<double>(ptr);
      }
      for (int i = 0; i < csr.nrows(); i++) {
        while (*(ptr) == ' ' || *(ptr) == '\n')
          ptr++;
        vec[i] = _read_num_and_move_ptr<double>(ptr);
      }
      return vec;
    }
  };

private:
  static char *_file_to_chars(std::string path) {
    auto file = std::ifstream(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open())
      return nullptr;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    auto *contents = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(contents, size);
    file.close();
    return contents;
  }
  template <typename T> static T _read_num_and_move_ptr(char *&ptr) {
    char *end;
    T res{};
    // auto [p, ec] = std::from_chars(ptr, ptr + 80, res);
    // ptr += (p - ptr) + 1;
    res = std::strtod(ptr, &end);
    ptr = end + 1;
    return res;
  }

  template <typename T1, typename T2, typename T3>
std::tuple<T1, T2, T3> read_three_nums(char *&ptr) {
  char *end;
  auto a = std::strtod(ptr, &end);
  ptr = end + 1;
  auto b = std::strtod(ptr, &end);
  ptr = end + 1;
  auto c = std::strtod(ptr, &end);
  ptr = end + 1;
  return {a, b, c};
}

};

} // namespace kmm

#endif // UMINEKO_SPARSE_MATRIX_IO_HPP
