#include "openfhe.h"
#include <Eigen/Dense>
#include <chrono>

size_t cleartext_memory(const Eigen::MatrixXd &matrix);

size_t ciphertext_memory(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt);

size_t
ciphertext_memory_ser(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt);

void save_ciphertext(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
                     const std::string &filename);

template <typename F> double measure_ms(F &&fn) {
  const auto start = std::chrono::steady_clock::now();
  std::forward<F>(fn)();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}