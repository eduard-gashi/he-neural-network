#pragma once
#include "Tensor.h"
#include <iostream>
#include <vector>

class NeuralNetwork {
public:
  std::vector<size_t> layers;
  std::vector<Tensor> weights;
  std::vector<Tensor> biases;

  NeuralNetwork(const std::vector<size_t> &layers);

  void set_data(const Tensor &X, const Tensor &y);
  Tensor forward(const Tensor &X);
  double compute_loss(const Tensor &y_pred);
  void train(int epochs, double learning_rate);

private:
  Tensor _X_train;
  Tensor _y_train;

  Tensor sigmoid(const Tensor &z);
  Tensor sigmoid_deriv(const Tensor &s);
  Tensor matmul(const Tensor &a, const Tensor &b);
};
