#include "network/Tensor.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

Tensor::Tensor() : _data{}, _shape{}, _stride{} {}

Tensor::Tensor(float data) : _data{data}, _shape{}, _stride{} {};

Tensor::Tensor(std::vector<float> data)
    : _data(data), _shape{data.size()}, _stride{1} {};

Tensor::Tensor(std::vector<std::vector<float>> data)
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1} {
  std::size_t n_expected_columns = data[0].size();
  for (std::size_t i = 0; i < data.size(); i++) {
    if (data[i].size() != n_expected_columns) {
      throw std::invalid_argument("Dimensions are inconsistent.");
    }
  }
  // Store in Row Major Format
  for (std::size_t i = 0; i < data.size(); i++) {
    for (std::size_t j = 0; j < data[0].size(); j++) {
      _data.push_back(data[i][j]);
    }
  }
}

const float &Tensor::item() const {
  if (_shape.empty() && !_data.empty()) {
    return _data[0];
  }
  if (_shape.size() == 1 && _shape[0] == 1) {
    return _data[0];
  }
  throw std::runtime_error("item(): Tensor must be scalar (shape [] or [1])");
}

float &Tensor::item() {
  if (_data.empty())
    throw std::runtime_error("empty tensor");
  return _data[0];
}

// 1D Indexing
const float &Tensor::operator()(std::size_t i) const {
  if (_shape.size() != 1) {
    throw std::invalid_argument(
        "1D index does not fit tensor shape of the tensor");
  }
  if (_shape[0] <= i) {
    throw std::invalid_argument("Index + " + std::to_string(i) +
                                " is out of bounds for array of size " +
                                std::to_string(_shape[0]));
  }
  return _data[i];
}

float &Tensor::operator()(std::size_t i) {
  if (_shape.size() != 1) {
    throw std::invalid_argument(
        "1D index does not fit tensor shape of the tensor");
  }
  if (_shape[0] <= i) {
    throw std::invalid_argument("Index + " + std::to_string(i) +
                                " is out of bounds for array of size " +
                                std::to_string(_shape[0]));
  }
  return _data[i];
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const {
  if (_shape.size() != 2) {
    throw std::invalid_argument(
        "2D index does not fit tensor shape of the tensor");
  }
  if (_shape[0] <= i || _shape[1] <= j) {
    throw std::invalid_argument(
        "Index (" + std::to_string(i) + "," + std::to_string(j) +
        ") out of bounds for shape[" + std::to_string(_shape[0]) + "," +
        std::to_string(_shape[1]) + "]");
  }
  std::size_t flat_index = i * _stride[0] + j * _stride[1];
  return _data[flat_index];
}

float &Tensor::operator()(std::size_t i, std::size_t j) {
  if (_shape.size() != 2) {
    throw std::invalid_argument(
        "2D index does not fit tensor shape of the tensor");
  }
  if (_shape[0] <= i || _shape[1] <= j) {
    throw std::invalid_argument(
        "Index (" + std::to_string(i) + "," + std::to_string(j) +
        ") out of bounds for shape[" + std::to_string(_shape[0]) + "," +
        std::to_string(_shape[1]) + "]");
  }
  std::size_t flat_index = i * _stride[0] + j * _stride[1];
  return _data[flat_index];
}

// Addition
Tensor Tensor::add_scalar_vector(const Tensor &scalar, const Tensor &vec) {
  std::vector<float> result;
  result.reserve(vec._data.size());
  for (std::size_t i = 0; i < vec._data.size(); ++i) {
    result.push_back(scalar.item() + vec._data[i]);
  }
  return Tensor(result);
}

Tensor Tensor::add_scalar_matrix(const Tensor &scalar, const Tensor &matrix) {
  std::vector<std::vector<float>> result;
  for (std::size_t i = 0; i < matrix._shape[0]; i++) {
    std::vector<float> result_i;
    for (std::size_t j = 0; j < matrix._shape[1]; j++) {
      result_i.push_back(scalar.item() + (matrix)(i, j));
    }
    result.push_back(result_i);
  }
  return Tensor(result);
}

Tensor Tensor::operator+(const Tensor &other) {
  // Scalar + Scalar
  if (_shape.size() == 0 && other._shape.size() == 0) {
    float result = item() + other.item();
    return Tensor(result);
  }
  // Scalar + 1D
  if (_shape.size() == 0 && other._shape.size() == 1) {

    return add_scalar_vector(*this, other);
  }
  // 1D + Scalar
  if (_shape.size() == 1 && other._shape.size() == 0) {
    return add_scalar_vector(other, *this);
  }
  // Scalar + 2D
  if (_shape.size() == 0 && other._shape.size() == 2) {
    return add_scalar_matrix(*this, other);
  }
  // 2D + Scalar
  if (_shape.size() == 2 && other._shape.size() == 0) {
    return add_scalar_matrix(other, *this);
  }
  // 1D + 1D
  if (_shape.size() == 1 && other._shape.size() == 1) {
    if (_shape[0] != other._shape[0]) {
      throw std::invalid_argument("Vectors need same length for addition.");
    }
    std::vector<float> result;
    for (std::size_t i = 0; i < _shape[0]; i++) {
      result.push_back(_data[i] + other._data[i]);
    }
    return result;
  }
  // 2D + 2D
  if (_shape.size() == 2 && other._shape.size() == 2) {
    if (_shape[0] != other._shape[0] || _shape[1] != other._shape[1]) {
      throw std::invalid_argument(
          "Matrices need same dimensions for addition.");
    }
    std::vector<float> flat_result(_data.size());
    for (std::size_t idx = 0; idx < _data.size(); ++idx) {
      flat_result[idx] = _data[idx] + other._data[idx];
    }

    std::vector<std::vector<float>> result_2d(_shape[0],
                                              std::vector<float>(_shape[1]));
    for (std::size_t i = 0; i < _shape[0]; ++i) {
      for (std::size_t j = 0; j < _shape[1]; ++j) {
        std::size_t flat_idx = i * _stride[0] + j * _stride[1];
        result_2d[i][j] = flat_result[flat_idx];
      }
    }
    return Tensor(result_2d);
  }
  return *this;
}

// Multiplication Hadamard (element-wise)
Tensor Tensor::multiply_scalar_vector(const Tensor &scalar, const Tensor &vec) {
  std::vector<float> result;
  result.reserve(vec._data.size());
  for (std::size_t i = 0; i < vec._data.size(); ++i) {
    result.push_back(scalar.item() * vec._data[i]);
  }
  return Tensor(result);
}

Tensor Tensor::multiply_scalar_matrix(const Tensor &scalar,
                                      const Tensor &matrix) {
  std::vector<std::vector<float>> result;
  for (std::size_t i = 0; i < matrix._shape[0]; i++) {
    std::vector<float> result_i;
    for (std::size_t j = 0; j < matrix._shape[1]; j++) {
      result_i.push_back(scalar.item() * (matrix)(i, j));
    }
    result.push_back(result_i);
  }
  return Tensor(result);
}

Tensor Tensor::multiply_vector_matrix(const Tensor &vector,
                                      const Tensor &matrix) {
  std::size_t n = vector._shape[0];
  std::size_t rows = matrix._shape[0];
  std::size_t cols = matrix._shape[1];

  if (n != cols)
    throw std::invalid_argument(
        "Vector length must equal number of columns for broadcasting.");

  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      std::size_t flat_idx = i * matrix._stride[0] + j * matrix._stride[1];
      result[i][j] = matrix._data[flat_idx] *
                     vector._data[j]; // multiply column j by vector[j]
    }
  }
  return Tensor(result);
}

Tensor Tensor::operator*(const Tensor &other) {
  // Scalar * Scalar
  if (_shape.size() == 0 && other._shape.size() == 0) {
    float result = item() * other.item();
    return Tensor(result);
  }
  // Scalar * 1D
  if (_shape.size() == 0 && other._shape.size() == 1) {
    return multiply_scalar_vector(*this, other);
  }
  // 1D * Scalar
  if (_shape.size() == 1 && other._shape.size() == 0) {
    return multiply_scalar_vector(other, *this);
  }
  // Scalar * 2D
  if (_shape.size() == 0 && other._shape.size() == 2) {
    return multiply_scalar_matrix(*this, other);
  }
  // 2D * Scalar
  if (_shape.size() == 2 && other._shape.size() == 0) {
    return multiply_scalar_matrix(other, *this);
  }
  // 1D * 1D
  if (_shape.size() == 1 && other._shape.size() == 1) {
    if (_shape[0] != other._shape[0]) {
      throw std::invalid_argument("Vectors need same length for addition.");
    }
    std::vector<float> result;
    for (std::size_t i = 0; i < _shape[0]; i++) {
      result.push_back(_data[i] * other._data[i]);
    }
    return result;
  }
  // 1D * 2D
  if (_shape.size() == 1 && other._shape.size() == 2) {
    return multiply_vector_matrix(*this, other);
  }
  // 2D * 1D
  if (_shape.size() == 2 && other._shape.size() == 1) {
    return multiply_vector_matrix(other, *this);
  }
  // 2D * 2D
  if (_shape.size() == 2 && other._shape.size() == 2) {
    if (_shape[0] != other._shape[0] || _shape[1] != other._shape[1]) {
      throw std::invalid_argument(
          "Matrices need same dimensions for multiplication.");
    }
    std::vector<float> flat_result(_data.size());
    for (std::size_t idx = 0; idx < _data.size(); ++idx) {
      flat_result[idx] = _data[idx] * other._data[idx];
    }

    std::vector<std::vector<float>> result_2d(_shape[0],
                                              std::vector<float>(_shape[1]));
    for (std::size_t i = 0; i < _shape[0]; ++i) {
      for (std::size_t j = 0; j < _shape[1]; ++j) {
        std::size_t flat_idx = i * _stride[0] + j * _stride[1];
        result_2d[i][j] = flat_result[flat_idx];
      }
    }
    return Tensor(result_2d);
  }
  return *this;
}

Tensor Tensor::matmul(const Tensor &other) {
  // Scalar * Scalar
  if (_shape.size() == 0 && other._shape.size() == 0) {
    float result = item() * other.item();
    return Tensor(result);
  }
  // Scalar * 1D
  if (_shape.size() == 0 && other._shape.size() == 1) {
    return multiply_scalar_vector(*this, other);
  }
  // 1D * Scalar
  if (_shape.size() == 1 && other._shape.size() == 0) {
    return multiply_scalar_vector(other, *this);
  }
  // Scalar * 2D
  if (_shape.size() == 0 && other._shape.size() == 2) {
    return multiply_scalar_matrix(*this, other);
  }
  // 2D * Scalar
  if (_shape.size() == 2 && other._shape.size() == 0) {
    return multiply_scalar_matrix(other, *this);
  }
  // 1D * 1D
  if (_shape.size() == 1 && other._shape.size() == 1) {
    if (_shape[0] != other._shape[0]) {
      throw std::invalid_argument("Vectors need same length for addition.");
    }
    float result = 0;
    for (std::size_t i = 0; i < _shape[0]; i++) {
      result += _data[i] * other._data[i];
    }
    return result;
  }
  // 1D * 2D
  if (_shape.size() == 1 && other._shape.size() == 2) {
    std::size_t n = _shape[0];
    std::size_t rows = other._shape[0];
    std::size_t cols = other._shape[1];

    if (n != rows) {
      throw std::invalid_argument("Shapes of vector and matrix do not match "
                                  "for matrix multiplication (1D*2D).");
    }
    std::vector<float> result;
    float entry = 0.0f;
    for (std::size_t i = 0; i < cols; i++) {
      entry = 0.0f;
      for (std::size_t j = 0; j < rows; j++) {
        entry += _data[i] * other(j, i);
      }
      std::cout << entry;
      result.push_back(entry);
    }
    return result;
  }
  // 2D * 1D
  if (_shape.size() == 2 && other._shape.size() == 1) {
    std::size_t n = other._shape[0];
    std::size_t rows = _shape[0];
    std::size_t cols = _shape[1];

    if (cols != n) {
      throw std::invalid_argument("Shapes of matrix and vector do not match "
                                  "for matrix multiplication (2D*1D).");
    }
    std::vector<float> result;
    float entry = 0.0f;
    for (std::size_t i = 0; i < rows; i++) {
      entry = 0.0f;
      for (std::size_t j = 0; j < cols; j++) {
        entry += (*this)(i, j) * other._data[j];
      }
      result.push_back(entry);
    }
    return result;
  }
  // 2D * 2D
  if (_shape.size() == 2 && other._shape.size() == 2) {
    std::size_t rows = _shape[0];
    std::size_t cols = _shape[1];

    std::size_t other_rows = other._shape[0];
    std::size_t other_cols = other._shape[1];

    if (cols != other_rows) {
      throw std::invalid_argument("Dimensions must match for matmul (2D*2D).");
    }
    std::vector<std::vector<float>> result;
    float entry = 0.0f;
    for (std::size_t i = 0; i < rows; i++) {
      std::vector<float> result_i;
      for (std::size_t j = 0; j < other_cols; j++) {
        entry = 0.0f;
        for (std::size_t k = 0; k < cols; k++) {
          entry += (*this)(i, k) * other(k, j);
        }
        result_i.push_back(entry);
      }
      result.push_back(result_i);
    }
    return Tensor(result);
  }
  return *this;
}

Tensor Tensor::row(std::size_t row_idx) const {
  if (_shape.size() != 2) {
    throw std::invalid_argument("row(): Tensor must be 2D");
  }
  if (row_idx >= _shape[0]) {
    throw std::invalid_argument("Row index out of bounds");
  }

  // Row-major offset: row_idx * num_cols
  std::size_t offset = row_idx * _shape[1];
  std::vector<float> row_data(_data.begin() + offset,
                              _data.begin() + offset + _shape[1]);

  return Tensor(row_data);
}

Tensor Tensor::column(std::size_t col_idx) const {
  if (_shape.size() != 2) {
    throw std::invalid_argument("column(): Tensor must be 2D");
  }
  if (col_idx >= _shape[1]) {
    throw std::invalid_argument("Column index out of bounds");
  }

  std::vector<float> col_data(_shape[0]);
  for (std::size_t row = 0; row < _shape[0]; ++row) {
    col_data[row] = (*this)(row, col_idx);
  }
  return Tensor(col_data);
}

Tensor Tensor::transpose() const {
  if (_shape.size() != 2) {
    throw std::invalid_argument("transpose(): Nur für 2D-Tensoren");
  }

  size_t rows = _shape[0];
  size_t cols = _shape[1];

  // Neues Shape: [cols, rows]
  std::vector<std::vector<float>> transposed_data(cols,
                                                  std::vector<float>(rows));

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      transposed_data[j][i] = (*this)(i, j); // (i,j) -> (j,i)
    }
  }

  return Tensor(transposed_data); // Automatisch Shape [cols, rows]
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  // Print shape
  os << "Tensor(shape=[";
  for (size_t i = 0; i < t._shape.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << t._shape[i];
  }
  os << "])\n";

  // Handle different ranks
  if (t._shape.empty() || (t._shape.size() == 1 && t._shape[0] == 1)) {
    // Scalar
    os << t.item() << "\n";
  } else if (t._shape.size() == 1) {
    // 1D vector
    os << "[";
    for (size_t i = 0; i < t._shape[0]; ++i) {
      if (i > 0)
        os << " ";
      os << t(i);
      if (t._shape[0] > 10 && i == 9) {
        os << " ...";
        break;
      }
    }
    os << "]\n";
  } else if (t._shape.size() == 2) {
    // 2D matrix - print as grid
    size_t rows = t._shape[0], cols = t._shape[1];
    size_t max_rows = std::min(rows, size_t(10));
    size_t max_cols = std::min(cols, size_t(10));

    for (size_t i = 0; i < max_rows; ++i) {
      os << "[";
      for (size_t j = 0; j < max_cols; ++j) {
        if (j > 0)
          os << "  ";
        os << std::fixed << std::setprecision(4) << t(i, j);
        if (cols > 10 && j == 8) {
          os << " ...";
          break;
        }
      }
      os << "]";
      if (rows > 10 && i == 9) {
        os << " ...";
      }
      os << "\n";
    }
  } else {
    // Higher dimensions - flat data preview
    os << "data=[";
    size_t max_print = std::min(t._data.size(), size_t(20));
    for (size_t i = 0; i < max_print; ++i) {
      if (i > 0)
        os << ", ";
      os << t._data[i];
    }
    if (t._data.size() > 20)
      os << ", ...";
    os << "]\n";
  }
  return os;
}
