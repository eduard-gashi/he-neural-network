#pragma once

#include "openfhe.h"

struct CryptoConfig {
  uint32_t mult_depth = 16;
  uint32_t scaled_mod_size = 28;
  uint32_t batch_size = 16;
  lbcrypto::SecurityLevel security_level = lbcrypto::HEStd_128_classic;
};

struct TrainConfig {
  int epochs = 1;
  double learning_rate = 1.3;
  std::vector<size_t> layers{1};
};
