#include "Encoder.h"
#include "openfhe.h"

void Encoder::setupCryptoContext(uint32_t batch_size, uint32_t mult_depth,
                                 uint32_t scale_mod_size,
                                 lbcrypto::SecurityLevel security_level) {

  // Setup Parameters
  lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;

  // Set security level, controls minimum ring dimension and modulus chain
  parameters.SetSecurityLevel(security_level);

  // Multiplicative Depth, e.g. for x1*x2*x3 = 2
  parameters.SetMultiplicativeDepth(mult_depth);

  // Bit-Length of the scaling factor D, determines the precision of stored data
  parameters.SetScalingModSize(scale_mod_size);

  // Number of slots used in the ciphertext, has to be < RingDimension / 2
  parameters.SetBatchSize(batch_size);

  // Create CryptoContext Object
  cc = GenCryptoContext(parameters);

  // Enable features
  cc->Enable(lbcrypto::PKE);
  cc->Enable(lbcrypto::KEYSWITCH);
  cc->Enable(lbcrypto::LEVELEDSHE);
  cc->Enable(lbcrypto::ADVANCEDSHE);

  std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension()
            << std::endl;
}

void Encoder::generateKeys() {
  // Generate Encryption Keys
  keys = cc->KeyGen();

  // Enable ciphertext multiplication
  cc->EvalMultKeyGen(keys.secretKey);

  // Enable summation
  cc->EvalSumKeyGen(keys.secretKey);

  // Enable rotation with needed keys
  cc->EvalRotateKeyGen(keys.secretKey,
                       {-12, -9, -6, -5, -3, 0, 1, 2, 4, 5, 6, 8});
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::extractColumn(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &matrix,
                       size_t col_idx, size_t num_features,
                       size_t num_rows) const {
  auto matrix_rotated = cc->EvalAtIndex(matrix, col_idx);

  // Extracts a column in a matrix with rotation and masks
  auto result = encodeMatrixPadded(std::vector<std::vector<double>>({{0.0}}));

  for (size_t i = 0; i < num_rows; i++) {
    // Mask out the rotated entry
    std::vector<std::vector<double>> mask_vec(
        1, std::vector<double>(num_rows, 0.0));
    mask_vec[0][i] = 1.0;

    auto mask = encodeMatrixOnce(mask_vec);

    // Rotate ciphertext to extract a column
    auto rotated = cc->EvalRotate(matrix_rotated, (num_features - 1) * i);
    rotated = cc->EvalMult(rotated, mask);

    // Add to result
    result = cc->EvalAdd(result, rotated);
  }
  return result;
}

// Decryption (only possible with secret_key)
Eigen::MatrixXd
Encoder::decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &matrix,
                 size_t length) const {
  lbcrypto::Plaintext result_plain;
  cc->Decrypt(keys.secretKey, matrix, &result_plain);
  result_plain->SetLength(length);

  std::vector<std::complex<double>> cleartext =
      result_plain->GetCKKSPackedValue();
  Eigen::VectorXd real_vals(cleartext.size());

  for (size_t i = 0; i < cleartext.size(); i++) {
    real_vals(i) = cleartext[i].real();
  }

  return real_vals;
}

Eigen::MatrixXd
Encoder::decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
                 size_t n_rows, size_t n_cols) const {
  lbcrypto::Plaintext result_plain;
  cc->Decrypt(keys.secretKey, ctxt, &result_plain);
  result_plain->SetLength(n_rows * n_cols);

  std::vector<std::complex<double>> cleartext =
      result_plain->GetCKKSPackedValue();
  Eigen::MatrixXd real_vals(n_rows, n_cols);

  for (size_t i = 0; i < n_rows; i++) {
    for (std::size_t j = 0; j < n_cols; j++) {
      real_vals(i, j) = cleartext[i * n_cols + j].real(); // Row Major Order
    }
  }

  return real_vals;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::add(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt1,
             const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt2) const {
  return cc->EvalAdd(ctxt1, ctxt2);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::sub(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt1,
             const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt2) const {
  return cc->EvalSub(ctxt1, ctxt2);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::mult(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt1,
              const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt2) const {
  return cc->EvalMult(ctxt1, ctxt2);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::mult(double number,
              const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt) const {
  return cc->EvalMult(number, ctxt);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::sumSlots(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
                  size_t slots) const {
  return cc->EvalSum(ctxt, slots);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::matmulXW(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &X,
                  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &W,
                  size_t num_features) const {
  auto ct_prod = cc->EvalMult(X, W);
  auto sum_cols_key = cc->EvalSumColsKeyGen(keys.secretKey);
  auto ct_sum_matrix = cc->EvalSumCols(ct_prod, num_features, *sum_cols_key);
  return ct_sum_matrix;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::_repeatBlock(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
                      size_t slots, size_t block) const {
  auto ctxt_repeated = ctxt;
  for (size_t shift = block; shift < slots; shift += block) {
    ctxt_repeated = cc->EvalAdd(
        ctxt_repeated, cc->EvalRotate(ctxt, static_cast<int32_t>(-shift)));
  }
  return ctxt_repeated;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::matmulXtDelta(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &Xt,
                       const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &delta,
                       size_t num_features, size_t num_rows) const {
  auto delta_repeated =
      _repeatBlock(delta, num_features * num_rows, num_features);

  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> results;
  auto ct_prod = cc->EvalMult(Xt, delta_repeated);

  std::vector<double> mask_vec({1, 1, 1, 1, 1, 0, 0, 0, 0, 0});
  auto mask = cc->MakeCKKSPackedPlaintext(mask_vec);
  auto ct_masked = cc->EvalMult(ct_prod, mask);
  auto ct_sum = cc->EvalSum(ct_masked, num_features);
  results.push_back(ct_sum);

  ct_prod = cc->EvalRotate(ct_prod, 5);
  auto ctMasked2 = cc->EvalMult(ct_prod, mask);
  auto ct_sum2 = cc->EvalSum(ctMasked2, num_features);
  results.push_back(ct_sum2);

  auto ct_sum_matrix = cc->EvalMerge(results);
  ct_sum_matrix = cc->EvalRotate(ct_sum_matrix, -1);
  ct_sum_matrix = _repeatBlock(ct_sum_matrix, 15, 3);

  return ct_sum_matrix;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Encoder::sumColumn(
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt, // batch x features
    size_t rows, size_t cols) const {
  auto result = cc->EvalSum(ctxt, rows);
  std::vector<double> mask_vec({1.0, 0.0, 0.0, 0.0, 0.0});
  auto mask = cc->MakeCKKSPackedPlaintext(mask_vec);
  result = cc->EvalMult(result, mask);
  result = _repeatBlock(result, rows, 1);
  return result;
}

std::vector<double>
Encoder::_flattenMatrix(const std::vector<std::vector<double>> &mat) const {
  if (mat.empty() || mat[0].empty())
    return {};

  const size_t n = mat.size();
  const size_t m = mat[0].size();

  std::vector<double> out;
  out.reserve(n * m);

  for (const auto &row : mat) {
    out.insert(out.end(), row.begin(), row.end());
  }

  return out;
}

std::vector<double> Encoder::_repeatToSlots(const std::vector<double> &base,
                                            size_t total_slots) const {
  if (base.empty() || total_slots == 0)
    return {};

  const size_t blocks = total_slots / base.size();

  std::vector<double> out;
  out.reserve(total_slots);

  for (size_t i = 0; i < blocks; ++i) {
    out.insert(out.end(), base.begin(), base.end());
  }

  return out;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Encoder::encodeMatrixOnce(
    const std::vector<std::vector<double>> &matrix) const {
  auto flat = _flattenMatrix(matrix);
  lbcrypto::Plaintext pt = cc->MakeCKKSPackedPlaintext(flat);
  return cc->Encrypt(keys.publicKey, pt);
}

std::vector<double> Encoder::_flattenMatrix(const Eigen::MatrixXd &M) const {
  std::vector<double> result;
  result.reserve(M.rows() * M.cols());

  for (Eigen::Index i = 0; i < M.rows(); ++i) {
    for (Eigen::Index j = 0; j < M.cols(); ++j) {
      result.push_back(M(i, j));
    }
  }

  return result;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
Encoder::encodeMatrixOnce(const Eigen::MatrixXd &matrix) const {
  auto flat = _flattenMatrix(matrix);
  lbcrypto::Plaintext pt = cc->MakeCKKSPackedPlaintext(flat);
  return cc->Encrypt(keys.publicKey, pt);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Encoder::encodeMatrixPadded(
    const std::vector<std::vector<double>> &matrix) const {
  auto flat = _flattenMatrix(matrix);
  auto padded = _repeatToSlots(flat, this->batch_size);
  lbcrypto::Plaintext pt = cc->MakeCKKSPackedPlaintext(padded);
  return cc->Encrypt(keys.publicKey, pt);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Encoder::applyChebyshevApproximation(
    const std::function<double(double)> &func,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt, double lower_bound,
    double upper_bound, uint32_t poly_degree) const {
  return cc->EvalChebyshevFunction(func, ctxt, lower_bound, upper_bound,
                                   poly_degree);
}
