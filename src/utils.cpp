#include "DataProcessor.h"
#include "Encoder.h"
#include "network/NeuralNetwork.h"
#include "network/Tensor.h"
#include "openfhe.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

size_t tensor_memory(const Tensor &t) {
  size_t data_bytes = t.flatten().size() * sizeof(float);
  size_t shape_bytes = t.shape().size() * sizeof(size_t);
  size_t stride_bytes = t.stride().size() * sizeof(size_t);
  size_t total = data_bytes + shape_bytes + stride_bytes;

  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TENSOR MEMORY" << std::endl;
  std::cout << std::string(60, '=') << "\n";

  std::cout << "Data:     " << t.flatten().size() << " floats x "
            << sizeof(float) << " B = **" << data_bytes << " B**" << std::endl;
  std::cout << "Shape:    " << t.shape().size() << " x " << sizeof(size_t)
            << " B = **" << shape_bytes << " B**" << std::endl;
  std::cout << "Stride:   " << t.stride().size() << " x " << sizeof(size_t)
            << " B = **" << stride_bytes << " B**" << std::endl;
  std::cout << std::string(40, '-') << "\n";
  std::cout << "TOTAL:    **" << total << " BYTES**" << std::endl;
  std::cout << std::string(60, '=') << "\n";

  return total;
}

size_t plaintexts_memory(const std::vector<Plaintext> &ptxts) {
  // std::cout << "Plaintxt Capacity " << ptxts.capacity();
  size_t total = ptxts.size() * sizeof(Plaintext); // Vector overhead
  std::cout << "Total amount of storage for plaintext: " << total << std::endl;
  return total;
}

// size_t ciphertext_memory(const std::vector<Ciphertext<DCRTPoly>> &ctxts) {
//   std::stringstream ss;

//   for (const auto &ct : ctxts) {
//     Serial::Serialize(ct, ss, SerType::BINARY);
//   }

//   return ss.str().size();
// }

size_t ctxt_size(const Ciphertext<DCRTPoly> &ctxt) {
  size_t size = 0;
  for (auto &element : ctxt->GetElements()) {
    for (auto &subelements : element.GetAllElements()) {
      auto length = subelements.GetLength();
      size += length * sizeof(subelements[0]);
    }
  }
  return size;
}

size_t ciphertext_memory(const std::vector<Ciphertext<DCRTPoly>> &ctxts) {
  size_t total = 0;
  for (const auto &cc : ctxts) {
    // Grobe Schätzung: 2 Polys (c0,c1) pro CT
    total += ctxt_size(cc);
  }
  std::cout << "Ciphertexts: " << ctxts.size() << " x ~"
            << (ctxts.empty() ? 0 : total / ctxts.size()) << " B = **" << total
            << " B**\n";
  return total;
}

size_t ciphertext_memory_ser(const std::vector<Ciphertext<DCRTPoly>> &ctxts) {
  size_t total = 0;

  std::stringstream stream;

  for (const auto &cc : ctxts) {
    Serial::Serialize(cc, stream, SerType::BINARY);
    total += stream.tellp();
    stream.str("");
    stream.clear();
  }
  return total;
}

void serialize_to_json(const std::vector<Ciphertext<DCRTPoly>> &ctxts) {
  if (!Serial::SerializeToFile("demoData/cryptoContext.txt", ctxts[0],
                               SerType::JSON)) {
    std::cerr << "Error serializing the cryptocontext" << std::endl;
  }
}