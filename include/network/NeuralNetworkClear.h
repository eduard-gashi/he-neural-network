#pragma once
#include "network/NeuralNetwork.h"
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>

class NeuralNetworkClear : public NeuralNetwork {
public:
  // Constructor
  explicit NeuralNetworkClear(const std::vector<size_t> &layers_)
      : NeuralNetwork(layers_) {}

  std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>
  forward(const Eigen::MatrixXd &X) const;

  void train(int epochs, double learning_rate) override;
  void setData(const Eigen::MatrixXd &X_, const Eigen::VectorXd &y_);

  double mseLoss(const Eigen::MatrixXd &error) const;

  Eigen::MatrixXd relu(const Eigen::MatrixXd &y_pred) const;
  Eigen::MatrixXd reluDeriv(const Eigen::MatrixXd &z) const;

protected:
  void initializeWeightsAndBias() override;

private:
  Eigen::MatrixXd X;
  Eigen::VectorXd y;
  Eigen::MatrixXd Xt;

  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::VectorXd> bias;
};
