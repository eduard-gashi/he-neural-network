#pragma once
#include <memory>
#include <ostream>
#include <span>
#include <vector>

class Tensor {
private:
  std::vector<float> _data; // Stores Tensor in Row Major Order
  std::vector<std::size_t> _shape;
  std::vector<std::size_t> _stride;

  // Helper Methods
  static Tensor add_scalar_vector(const Tensor &scalar, const Tensor &vec_data);

  static Tensor add_scalar_matrix(const Tensor &scalar, const Tensor &vec_data);

  static Tensor multiply_scalar_vector(const Tensor &scalar,
                                       const Tensor &vec_data);

  static Tensor multiply_scalar_matrix(const Tensor &scalar,
                                       const Tensor &vec_data);

  static Tensor multiply_vector_matrix(const Tensor &vector,
                                       const Tensor &matrix);

public:
  // Constructors
  Tensor();
  Tensor(float data);
  Tensor(std::vector<float> data);
  Tensor(std::vector<std::vector<float>> data);

  // Getter
  const std::vector<float> flatten() const {
    return _data;
  } // Returns data in Row Major order
  const std::vector<std::size_t> &shape() const { return _shape; }
  const std::vector<std::size_t> &stride() const { return _stride; }

  // Scalar
  const float &item() const; // Getter
  float &item();             // Setter

  // Indexing
  const float &operator()(std::size_t i) const;
  float &operator()(std::size_t i);
  const float &operator()(std::size_t i, std::size_t j) const;
  float &operator()(std::size_t i, std::size_t j);

  // Addition
  Tensor operator+(const Tensor &other);

  // Multiplication
  Tensor operator*(const Tensor &other);
  Tensor matmul(const Tensor &other);

  // Access Row
  Tensor row(std::size_t row_idx) const;

  // Access Column
  Tensor column(std::size_t col_idx) const;

  // Transpose
  Tensor transpose() const;

  // Print
  friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};