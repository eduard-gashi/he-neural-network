#include "openfhe.h"
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

// Folder to save serialized Ciphertexts
const std::filesystem::path DATAFOLDER =
    std::filesystem::path(__FILE__).parent_path() / "saved_data";

size_t cleartext_memory(const Eigen::MatrixXd &matrix) {
  size_t size =
      static_cast<size_t>(matrix.size()) * sizeof(double) + sizeof(matrix);
  return size;
}

size_t ciphertext_memory(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt) {
  size_t size = 0;
  for (auto &element : ctxt->GetElements()) {
    for (auto &subelements : element.GetAllElements()) {
      auto length = subelements.GetLength();
      size += length * sizeof(subelements[0]);
    }
  }
  return size;
}

size_t
ciphertext_memory_ser(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt) {
  size_t total = 0;
  std::stringstream stream;
  lbcrypto::Serial::Serialize(ctxt, stream, lbcrypto::SerType::BINARY);
  total += stream.tellp();
  stream.str("");
  stream.clear();
  return total;
}

void save_ciphertext(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ctxt,
                     const std::string &filename) {
  std::filesystem::create_directories(DATAFOLDER);
  std::filesystem::path fullpath = DATAFOLDER / filename;
  lbcrypto::Serial::SerializeToFile(fullpath.string(), ctxt,
                                    lbcrypto::SerType::JSON);
  std::cout << "Serialized ciphertext to JSON file: " << fullpath << std::endl;
}
