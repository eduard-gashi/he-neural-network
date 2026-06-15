#pragma once
#include "Encoder.h"
#include "network/NeuralNetwork.h"
#include "openfhe.h"
#include <Eigen/Dense>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

class NeuralNetworkCipher : public NeuralNetwork {
public:
  // Constructor
  explicit NeuralNetworkCipher(const std::vector<size_t> &layers_,
                               Encoder encoder_)
      : NeuralNetwork(layers_), encoder(encoder_) {}

  std::tuple<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>,
             std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
  forward(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &X) const;

  void train(int epochs, double learning_rate) override;
  void setData(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> data,
               lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Xt_,
               std::vector<size_t> shape);

  double mseLoss(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &error,
                 size_t num_rows) const;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  relu(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> y_pred, size_t slots) const;
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
  reluDeriv(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> y_pred,
            size_t slots) const;

protected:
  void initializeWeightsAndBias() override;

private:
  Encoder encoder;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> X;
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> y;
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Xt;

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> scale;
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> lr_ctxt;

  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> weights_encoded;
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> bias_encoded;
};
