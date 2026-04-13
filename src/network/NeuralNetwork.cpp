#include "network/NeuralNetwork.h"
#include <cmath>
#include <random>

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layer_sizes)
    : layers(layer_sizes) {}

void NeuralNetwork::set_data(const Tensor &X, const Tensor &y) {
  throw std::runtime_error("set_data() not implemented");
}

Tensor NeuralNetwork::forward(const Tensor &input) {
  throw std::runtime_error("forward() not implemented");
}

Tensor NeuralNetwork::sigmoid(const Tensor &z) {
  throw std::runtime_error("sigmoid() not implemented");
}
