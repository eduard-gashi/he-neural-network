#pragma once

#include "openfhe.h"
#include <Eigen/Dense>

class Encoder {
public:
  Encoder(uint32_t mult_depth = 3, uint32_t scale_mod_size = 50,
          uint32_t batch_size = 64,
          lbcrypto::SecurityLevel security_level = lbcrypto::HEStd_128_classic)
      : batch_size(batch_size), mult_depth(mult_depth),
        scale_mod_size(scale_mod_size), security_level(security_level) {
    // 1. Setup CryptoContext
    setupCryptoContext(batch_size, mult_depth, scale_mod_size, security_level);

    // 2. Key Generation
    generateKeys();
  }

  // Getters
  const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &getCryptoContext() const;

  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &getPublicKey() const;

  // Decryption (only possible with secret_key)
  Eigen::MatrixXd decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
                          size_t length) const;
  Eigen::MatrixXd decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
                          size_t n_rows, size_t n_cols) const;

  // Addition on encrypted Ciphertexts
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  add(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt1,
      const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt2) const;

  // Subtraction on encrypted Ciphertexts
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  sub(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt1,
      const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt2) const;

  // Elementwise multiplication on encrypted Ciphertexts
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  mult(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt1,
       const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt2) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  mult(double number,
       const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt) const;

  // Elementwise multiplication on encrypted Ciphertexts
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  sumSlots(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
           size_t slots) const;

  // Matrix encoding
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  encodeMatrixPadded(const std::vector<std::vector<double>> &matrix) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  encodeMatrixOnce(const std::vector<std::vector<double>> &matrix) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  encodeMatrixOnce(const Eigen::MatrixXd &matrix) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  encodeUserData(const Eigen::MatrixXd &matrix,
                 lbcrypto::PublicKey<lbcrypto::DCRTPoly> public_key) const;

  // Matrix multiplication
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  matmulXW(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &X,
           const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &W,
           size_t num_features) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  matmulXtDelta(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &Xt,
                const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &delta,
                size_t num_features, size_t num_rows) const;

  // Column extraction and summation
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  extractColumn(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &matrix,
                size_t col_idx, size_t num_features, size_t num_rows) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  sumColumn(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &matrix, size_t rows,
            size_t cols) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  rotate(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
         int32_t slots) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> applyBootstrapping(
      const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt) const;

  // Chebychev approximation
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> applyChebyshevApproximation(
      const std::function<double(double)> &func,
      const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt, double lower_bound,
      double upper_bound, uint32_t poly_degree) const;

private:
  void setupCryptoContext(uint32_t batch_size, uint32_t mult_depth,
                          uint32_t scale_mod_size,
                          lbcrypto::SecurityLevel security_level);

  void generateKeys();

  // Helper methods
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  _repeatBlock(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
               size_t slots, size_t block) const;

  std::vector<double>
  _flattenMatrix(const std::vector<std::vector<double>> &mat) const;

  std::vector<double> _flattenMatrix(const Eigen::MatrixXd &M) const;

  std::vector<double> _repeatToSlots(const std::vector<double> &base,
                                     size_t total_slots) const;

private:
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys;
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
  std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>
      sum_cols_keys;

  uint32_t batch_size;
  uint32_t mult_depth;
  uint32_t scale_mod_size;
  lbcrypto::SecurityLevel security_level;
};
