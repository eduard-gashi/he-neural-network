#include "Encoder.h"
#include "config.h"
#include "network/NeuralNetwork.h"
#include "network/NeuralNetworkCipher.h"
#include "network/NeuralNetworkClear.h"
#include "openfhe.h"
#include "utils.h"
#include <Eigen/Dense>
#include <iostream>

void test_prediction_user_data(const NeuralNetworkCipher &nn_encoded,
                               CryptoConfig crypto_cfg) {
  Encoder encoder_user =
      Encoder(crypto_cfg.mult_depth, crypto_cfg.scaled_mod_size,
              crypto_cfg.batch_size, crypto_cfg.security_level);

  Eigen::MatrixXd X_test(1, 2);
  X_test << 1.0, 2.0;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> X_test_ctxt =
      encoder_user.encodeMatrixOnce(X_test);

  auto y_pred = nn_encoded.predict(X_test_ctxt, encoder_user);
  auto y_pred_clear = encoder_user.decrypt(y_pred, 1)(0);
  bool y_pred_class = y_pred_clear > 0.5;

  std::cout << "Input data: " << X_test << std::endl;
  std::cout << "Raw Prediction: " << y_pred_clear << std::endl;
  std::cout << "Class: " << y_pred_class << std::endl;
}

void test_bootstrapping(Eigen::MatrixXd X, int ring_dimension) {
  /* Bootstrapping workflow
   * 1. Modulus Raising
   * 2. Coefficients to Slots
   * 3. Evaluate the Approximate Modular Reduction
   * 4. Slots to Coefficients
   * The number of levels required to be available before bootstrapping is 1 to
   * support a multiplication (and one more for FLEXIBLEAUTOEXT).
   */
  lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
  lbcrypto::SecretKeyDist secretKeyDist = lbcrypto::UNIFORM_TERNARY;
  parameters.SetSecretKeyDist(secretKeyDist);

  parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
  parameters.SetRingDim(ring_dimension); // Set ring rimdension

  parameters.SetScalingModSize(59);
  parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
  parameters.SetFirstModSize(60);

  std::vector<uint32_t> levelBudget = {4, 4};

  uint32_t levelsAvailableAfterBootstrap = 10;
  uint32_t depth =
      levelsAvailableAfterBootstrap +
      lbcrypto::FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
  parameters.SetMultiplicativeDepth(depth);

  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cryptoContext =
      lbcrypto::GenCryptoContext(parameters);

  cryptoContext->Enable(lbcrypto::PKE);
  cryptoContext->Enable(lbcrypto::KEYSWITCH);
  cryptoContext->Enable(lbcrypto::LEVELEDSHE);
  cryptoContext->Enable(lbcrypto::ADVANCEDSHE);
  cryptoContext->Enable(lbcrypto::FHE);

  uint32_t ringDim = cryptoContext->GetRingDimension();
  uint32_t numSlots = ringDim / 2;
  std::cout << "CKKS scheme ring dimension: " << ringDim << "\n\n";

  cryptoContext->EvalBootstrapSetup(levelBudget, {0, 0}, numSlots);

  auto keyPair = cryptoContext->KeyGen();
  cryptoContext->EvalMultKeyGen(keyPair.secretKey);
  cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

  Eigen::VectorXd col = X.col(0);
  std::vector<double> x(col.data(), col.data() + col.size());
  size_t encodedLength = x.size();

  // Start with a depleted ciphertext that has used up all of its levels
  lbcrypto::Plaintext ptxt =
      cryptoContext->MakeCKKSPackedPlaintext(x, 1, depth - 1);

  ptxt->SetLength(encodedLength);
  std::cout << "Input: " << ptxt << "\n";

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciph =
      cryptoContext->Encrypt(keyPair.publicKey, ptxt);

  std::cout << "Initial number of levels remaining: "
            << depth - ciph->GetLevel() << "\n\n";

  auto start = std::chrono::high_resolution_clock::now();

  // Perform the bootstrapping operation. The goal is to increase the number of
  // levels remaining for HE computation.
  auto ciphertextAfter = cryptoContext->EvalBootstrap(ciph);

  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << "Bootstrapping time: "
            << std::chrono::duration<double>(stop - start).count() << " s\n\n";

  std::cout << "Number of levels remaining after bootstrapping: "
            << depth - ciphertextAfter->GetLevel() -
                   (ciphertextAfter->GetNoiseScaleDeg() - 1)
            << "\n\n";

  lbcrypto::Plaintext result;
  cryptoContext->Decrypt(keyPair.secretKey, ciphertextAfter, &result);
  result->SetLength(encodedLength);
  std::cout << "Output after bootstrapping: " << result << "\n";

  cryptoContext->ClearStaticMapsAndVectors();
}

int main() {
  // Hyperparameters
  CryptoConfig crypto_cfg;
  TrainConfig train_cfg;

  // Training data
  Eigen::MatrixXd X(5, 2);
  X << 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 2.0;
  Eigen::VectorXd y(5);
  y << 0.0, 1.0, 0.0, 1.0, 1.0;

  // Combine to one matrix [y^T, X]
  Eigen::MatrixXd training_data(X.rows(), X.cols() + 1);
  training_data.col(0) = y.transpose();
  training_data.rightCols(X.cols()) = X;

  Encoder encoder(crypto_cfg.mult_depth, crypto_cfg.scaled_mod_size,
                  crypto_cfg.batch_size, crypto_cfg.security_level);

  // Encode training data
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> training_data_ctxt =
      encoder.encodeMatrixOnce(training_data);

  size_t clear_memory = cleartext_memory(training_data);
  std::cout << "[CLEARTEXT] Memory of training data: " << clear_memory << " B."
            << std::endl
            << std::endl;
  size_t cipher_memory = ciphertext_memory(training_data_ctxt);
  std::cout << "[CIPHERTEXT] Memory of training data: " << cipher_memory
            << " B." << std::endl
            << std::endl;

  // Encode Xt for gradient computation
  Eigen::MatrixXd Xt = X.transpose();
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Xt_ctxt =
      encoder.encodeMatrixOnce(Xt);

  // Cleartext NN
  NeuralNetworkClear nn_clear(train_cfg.layers);
  nn_clear.setData(X, y);
  double clear_train_ms = measure_ms(
      [&] { nn_clear.train(train_cfg.epochs, train_cfg.learning_rate); });

  // Ciphertext NN
  NeuralNetworkCipher nn_encoded(train_cfg.layers, encoder);
  nn_encoded.setData(
      training_data_ctxt, Xt_ctxt,
      std::vector<size_t>{static_cast<size_t>(training_data.rows()),
                          static_cast<size_t>(training_data.cols())});
  double cipher_train_ms = measure_ms(
      [&] { nn_encoded.train(train_cfg.epochs, train_cfg.learning_rate); });

  std::cout << "[CLEARTEXT] Training duration: " << clear_train_ms << "ms\n";
  std::cout << "[CIPHERTEXT] Training duration: " << cipher_train_ms << "ms\n";

  //   test_prediction_user_data(nn_encoded, crypto_cfg);

  return 0;
}
