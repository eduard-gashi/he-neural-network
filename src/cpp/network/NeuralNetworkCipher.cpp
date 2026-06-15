
#include "network/NeuralNetworkCipher.h"
#include "network/NeuralNetwork.h"
#include "openfhe.h"
#include <tuple>

void NeuralNetworkCipher::initializeWeightsAndBias() {
  size_t prev = shape[1]; // Amount input features

  for (size_t i = 0; i < layers.size(); i++) {
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> weights_ctxts;

    size_t rows = prev;
    size_t cols = layers[i];

    std::vector<std::vector<double>> base_w(
        rows, std::vector<double>(
                  cols, 1.0)); // prev = COls, layers[i] = Rows = Neurons
    base_w[0] = std::vector<double>(cols, 0.0);
    auto weights_ctxt =
        encoder.encodeMatrixPadded(base_w); // Encrypt every column of weights
    weights_encoded.push_back(weights_ctxt);

    std::vector<std::vector<double>> bias(shape[0],
                                          std::vector<double>(cols, 1.0));
    auto bias_ctxt = encoder.encodeMatrixOnce(bias);
    bias_encoded.push_back(bias_ctxt);

    prev = layers[i];
  }
}

void NeuralNetworkCipher::setData(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> data,
                                  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Xt_,
                                  std::vector<size_t> shape_) {
  shape = shape_;

  y = encoder.extractColumn(data, 0, shape[1], shape[0]);

  X = data;

  Xt = Xt_;

  initializeWeightsAndBias();

  scale = encoder.encodeMatrixPadded({{2.0 / shape[0]}});
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
NeuralNetworkCipher::relu(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> y_pred,
                          size_t slots) const {
  double lower_bound = 0;
  double upper_bound = 5;

  uint32_t poly_degree = 3; // Depends on multiplicative depth

  auto result = encoder.applyChebyshevApproximation(
      [](double x) -> double {
        if (x < 0.0)
          return 0.0;
        else
          return x;
      },
      y_pred, lower_bound, upper_bound, poly_degree);

  return result;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
NeuralNetworkCipher::reluDeriv(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> z,
                               size_t slots) const {
  double lower_bound = 0.0;
  double upper_bound = 5.0;
  uint32_t poly_degree = 3; // Depends on the multiplicative depth

  auto result = encoder.applyChebyshevApproximation(
      [](double x) -> double {
        if (x < 0.0)
          return 0.0;
        else
          return 1;
      },
      z, lower_bound, upper_bound, poly_degree);

  return result;
}

double NeuralNetworkCipher::mseLoss(
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &error,
    size_t num_rows) const {
  // Compute square product
  auto error_sq = encoder.mult(error, error);

  // Compute summation
  auto error_sq_sum = encoder.sumSlots(error_sq, shape[0]);

  // Divide by number of rows
  auto loss_ctxt = encoder.mult(1.0 / num_rows, error_sq_sum);

  return encoder.decrypt(loss_ctxt, 1)(0);
}

std::tuple<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>,
           std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
NeuralNetworkCipher::forward(
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &X) const {
  // Lists of results
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> z_layers;
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> h_layers;

  // Current input to be multiplied with weights
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> current_input = X;

  // Temp variables
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> z;
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> h;

  size_t num_rows = shape[0];
  size_t num_cols = shape[1];

  // Iterate through every layer
  for (size_t i = 0; i < layers.size(); i++) {
    // X * W
    z = encoder.matmulXW(current_input, weights_encoded[i], num_cols);
    z = encoder.extractColumn(z, 1, shape[1], shape[0]);

    // Add bias
    z = encoder.add(z, bias_encoded[i]);

    // Apply relu activation
    h = relu(z, num_rows);

    z_layers.push_back(z);
    h_layers.push_back(h);

    // Amount of columns is amount of neurons from previous layer
    num_cols = layers[i];
  }

  return {z_layers, h_layers};
}

void NeuralNetworkCipher::train(int epochs, double learning_rate) {
  std::cout << "[CIPHERTEXT] Starting training process..." << std::endl;
  double loss = 0.0f;
  auto lr = encoder.encodeMatrixPadded({{learning_rate}});

  for (int epoch = 1; epoch <= epochs; epoch++) {
    // Forward Pass
    auto [z_layers, h_layers] = forward(X);

    // Extract final predictions
    auto h_out = h_layers.back();
    auto z_out = z_layers.back();

    // Compute error and loss
    auto ct_error = encoder.sub(h_out, y);
    loss = mseLoss(ct_error, shape[0]);

    // Backprop
    for (int l = layers.size() - 1; l >= 0; l--) {
      // Compute delta term: 2/N * error
      auto delta = encoder.mult(scale, ct_error);

      // Compute relu derivative: Relu'(z)
      auto relu_deriv = reluDeriv(z_out, shape[0]);

      // Multiply delta term with relu' and extract result
      delta = encoder.mult(delta, relu_deriv);

      // Compute gradient of bias: Summation of every column of delta
      auto grad_b = encoder.sumColumn(delta, shape[0], shape[1]);

      // Compute gradient of weights: Xt * delta
      auto grad_w = encoder.matmulXtDelta(Xt, delta, shape[0], shape[1] - 1);

      // Update bias and weights
      auto update_b = encoder.mult(lr, grad_b);
      auto update_w = encoder.mult(lr, grad_w);

      bias_encoded[l] = encoder.sub(bias_encoded[l], update_b);
      weights_encoded[l] = encoder.sub(weights_encoded[l], update_w);
    }
    std::cout << "[CIPHERTEXT] Training process running.. Epoch " << epoch
              << "/" << epochs << std::endl
              << "Loss " << loss << std::endl;
  }
  std::cout << "[CIPHERTEXT] Final bias: \n"
            << encoder.decrypt(bias_encoded[0], layers[0]) << std::endl
            << "Final weights: \n"
            << encoder.decrypt(weights_encoded[0], shape[1]) << std::endl
            << "Loss:\n"
            << loss << std::endl
            << std::endl;
}