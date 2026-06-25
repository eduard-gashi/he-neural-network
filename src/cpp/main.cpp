#include "Encoder.h"
#include "config.h"
#include "network/NeuralNetwork.h"
#include "network/NeuralNetworkCipher.h"
#include "network/NeuralNetworkClear.h"
#include "openfhe.h"
#include "utils.h"
#include <Eigen/Dense>
#include <iostream>
void test_prediction() {
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

  Eigen::MatrixXd X_test(1, 2);
  X_test << 1.0, 2.0; // 1.0, 0.0

  Encoder encoder(crypto_cfg.mult_depth, crypto_cfg.scaled_mod_size,
                  crypto_cfg.batch_size, crypto_cfg.security_level);
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> X_test_ctxt =
      encoder.encodeMatrixOnce(X_test);

  // Encode training data
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> training_data_ctxt =
      encoder.encodeMatrixOnce(training_data);

  // Encode Xt for gradient computation
  Eigen::MatrixXd Xt = X.transpose();
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Xt_ctxt =
      encoder.encodeMatrixOnce(Xt);

  // Ciphertext NN
  NeuralNetworkCipher nn_encoded(train_cfg.layers, encoder);
  nn_encoded.setData(
      training_data_ctxt, Xt_ctxt,
      std::vector<size_t>{static_cast<size_t>(training_data.rows()),
                          static_cast<size_t>(training_data.cols())});

  double cipher_train_ms = measure_ms(
      [&] { nn_encoded.train(train_cfg.epochs, train_cfg.learning_rate); });

  std::cout << "Train duration" << cipher_train_ms << std::endl;
  auto y_pred = nn_encoded.predict(X_test_ctxt);
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

  size_t clear_memory = cleartext_memory(X) + cleartext_memory(y);
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
  return 0;
}
