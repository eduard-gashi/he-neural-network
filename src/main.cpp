#include "DataProcessor.h"
#include "Encoder.h"
#include "network/NeuralNetwork.h"
#include "network/Tensor.h"
#include "openfhe.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

// Hyperparameters
uint32_t MULT_DEPTH = 1;
uint32_t SCALED_MOD_SIZE = 50;
uint32_t BATCH_SIZE = 8;

std::string DATAFOLDER = "saved_data/";

int main() {
  std::cout << "Start of script" << std::endl;
  DataProcessor processor;
  Dataset data = processor.load_csv("data/heart_disease_data.csv");
  Dataset sliced = processor.slice_data({"thalach", "cp"}, data); // Add cp

  Encoder encoder(MULT_DEPTH, SCALED_MOD_SIZE, BATCH_SIZE);
  std::vector<Plaintext> features_plaintext =
      encoder.encode_tensor(sliced.features);

  std::vector<Ciphertext<DCRTPoly>> features_ciphertext =
      encoder.encrypt(features_plaintext);

  size_t m_memory = tensor_memory(
      sliced.features); // Compute memory usage of data in cleartext

  size_t cipher_memory = ciphertext_memory(
      features_ciphertext); // memory usage of data in ciphertext

  std::cout << "Cleartext Storage: " << m_memory << std::endl;
  std::cout << "Ciphertext Storage: " << cipher_memory << std::endl;

  size_t cipher_memory_ser = ciphertext_memory_ser(features_ciphertext);
  std::cout << "Ciphertext Storage ser: " << cipher_memory_ser << std::endl;

  processor.save_data(sliced.features, sliced.feature_names,
                      DATAFOLDER + "cleartext.csv");
  processor.save_data(features_ciphertext, DATAFOLDER + "serialzied.txt");

  return 0;
}
