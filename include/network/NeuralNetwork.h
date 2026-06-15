#pragma once

#include <vector>

class NeuralNetwork {
public:
  // Destructor
  virtual ~NeuralNetwork() = default;

  // Constructor
  NeuralNetwork(const std::vector<size_t> &layers_) : layers(layers_) {};

  virtual void train(int epochs, double learning_rate) = 0;

protected:
  virtual void initializeWeightsAndBias() = 0;
  std::vector<size_t> layers; // Contains the amount of neurons for each layer
  std::vector<size_t> shape;  // [rows, col]
};
