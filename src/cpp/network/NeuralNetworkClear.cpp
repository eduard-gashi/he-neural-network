#include "network/NeuralNetworkClear.h"
#include "network/NeuralNetwork.h"
#include <Eigen/Dense>
#include <tuple>

void NeuralNetworkClear::initializeWeightsAndBias() {
  size_t prev = static_cast<size_t>(X.cols());

  for (size_t i = 0; i < layers.size(); i++) {
    // Create weight matrix: Rows=Amount of inputs, clumns=Amount of Neurons
    Eigen::MatrixXd w(prev, layers[i]);
    w.setZero();
    weights.emplace_back(w);

    // Create Bias: Amount of Neurons
    Eigen::VectorXd b(layers[i]);
    b.setConstant(-1.0);
    bias.emplace_back(b);

    prev = layers[i];
  }
}

void NeuralNetworkClear::setData(const Eigen::MatrixXd &X_,
                                 const Eigen::VectorXd &y_) {
  shape = {static_cast<size_t>(X.rows()), static_cast<size_t>(X.cols())};

  X = X_;

  y = y_;

  Xt = X.transpose();

  initializeWeightsAndBias();
}

std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>
NeuralNetworkClear::forward(const Eigen::MatrixXd &X) const {
  // Lists of results
  std::vector<Eigen::MatrixXd> z_layers;
  std::vector<Eigen::MatrixXd> h_layers;

  // Current input to be multiplied with weights
  Eigen::MatrixXd current_input = X;

  // Temp variables
  Eigen::MatrixXd z;
  Eigen::MatrixXd h;

  // Iterate through every layer
  for (size_t i = 0; i < layers.size(); i++) {
    // X * W
    z = current_input * weights[i];

    // Add bias
    z.rowwise() += bias[i].transpose();

    // Apply relu activation
    h = sigmoid(z);

    z_layers.push_back(z);
    h_layers.push_back(h);
  }

  return {z_layers, h_layers};
}

Eigen::MatrixXd NeuralNetworkClear::relu(const Eigen::MatrixXd &y_pred) const {
  return y_pred.cwiseMax(0.0);
}

double NeuralNetworkClear::mseLoss(const Eigen::MatrixXd &error) const {
  double loss = 0.0;

  for (size_t i = 0; i < static_cast<size_t>(error.rows()); ++i) {
    loss += std::pow(error(i), 2);
  }
  loss /= error.rows(); // Get Mean
  return loss;
}

Eigen::MatrixXd NeuralNetworkClear::reluDeriv(const Eigen::MatrixXd &z) const {
  return (z.array() > 0.0).cast<double>();
}

Eigen::MatrixXd
NeuralNetworkClear::sigmoid(const Eigen::MatrixXd &z_out) const {
  return (1.0 / (1.0 + (-z_out.array()).exp())).matrix();
}

Eigen::MatrixXd
NeuralNetworkClear::sigmoidDeriv(const Eigen::MatrixXd &h_out) const {
  return (h_out.array() * (1.0 - h_out.array())).matrix();
}

// Whole training process
void NeuralNetworkClear::train(int epochs, double learning_rate) {
  std::cout << "[Cleartext] Starting training process..." << std::endl;
  double loss = 0.0f;

  for (int epoch = 1; epoch <= epochs; epoch++) {
    // Forward Pass
    auto [z_layers, h_layers] = forward(X);

    // Extract final predictions
    Eigen::MatrixXd h_out = h_layers.back();
    Eigen::MatrixXd z_out = z_layers.back();

    // Compute error and loss
    Eigen::MatrixXd error = h_out - y;
    loss = mseLoss(error);

    size_t N = static_cast<size_t>(h_out.rows());

    // Backprop
    for (int layer = layers.size() - 1; layer >= 0; layer--) {
      // Compute delta term: 2/N * error
      Eigen::MatrixXd delta = (2.0 / N) * error;

      // Compute relu derivative: Relu'(z)
      Eigen::MatrixXd relu_deriv = sigmoidDeriv(h_out);

      // Multiply delta term with relu' and extract result
      delta = delta.cwiseProduct(relu_deriv);

      // Compute gradient of bias: Summation of every column of delta
      Eigen::MatrixXd grad_b = delta.colwise().sum();

      // Compute gradient of weights: Xt * delta
      Eigen::MatrixXd grad_w = Xt * delta;

      // Update bias and weights
      bias[layer] -= learning_rate * grad_b;
      weights[layer] -= learning_rate * grad_w;
    }
    std::cout << "[CLEARTEXT] Training process running.. Epoch " << epoch << "/"
              << epochs << std::endl
              << "Loss " << loss << std::endl;
  }
  std::cout << "[CLEARTEXT] Final bias: \n"
            << bias[0] << std::endl
            << "Final weights: \n"
            << weights[0] << std::endl
            << "Loss: " << loss << std::endl
            << std::endl;
}
