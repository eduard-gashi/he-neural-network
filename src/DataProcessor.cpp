#include "DataProcessor.h"
#include "openfhe.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using CiphertextType = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using PublicKeyType = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
using PrivateKeyType = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

using namespace lbcrypto;

Dataset DataProcessor::load_csv(const std::string &file_path) {
  Dataset data;

  std::ifstream file;
  file.open(file_path);

  if (!file.is_open()) {
    throw std::invalid_argument("Could not open file.");
  }

  // Get feature names (first line)
  std::string line;
  if (!std::getline(file, line)) {
    throw std::invalid_argument("Empty file or no header line.");
  }
  std::stringstream ss_header(line);
  std::string feature_name;
  while (std::getline(ss_header, feature_name, ',')) {
    data.feature_names.push_back(feature_name);
  }

  // Get all data
  std::vector<std::vector<float>> features;
  while (std::getline(file, line)) {
    std::stringstream ss_row(line);
    std::string cell;
    std::vector<float> row;
    while (std::getline(ss_row, cell, ',')) {
      // Convert string to float
      row.push_back(std::stof(cell));
    }
    features.push_back(row);
  }
  std::vector<float> targets;
  targets.reserve(features.size());
  for (auto &row : features) {
    if (!row.empty()) {
      targets.push_back(row.back());
      row.erase(row.end() - 1);
    }
  }

  data.features = Tensor(features);
  data.targets = Tensor(targets);

  return data;
}

Tensor DataProcessor::normalize_tensor(Tensor &tensor) {
  if (tensor.shape().size() != 2)
    return tensor;

  return tensor;
}

Dataset
DataProcessor::slice_data(const std::vector<std::string> &features_to_extract,
                          const Dataset &input_data) {
  std::vector<std::string> feature_names =
      input_data.feature_names; // Korrigierter Name
  const Tensor &data_tensor =
      input_data.features; // Referenz, kein Überschreiben!

  if (data_tensor.shape().size() != 2) { // _shape direkt zugreifen
    throw std::invalid_argument("Data must be 2D tensor");
  }

  std::vector<size_t> col_indices;
  for (const auto &fname : features_to_extract) {
    bool found = false;
    for (size_t i = 0; i < feature_names.size(); ++i) {
      if (feature_names[i] == fname) {
        col_indices.push_back(i);
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::invalid_argument("Feature '" + fname + "' not found");
    }
  }

  // Submatrix bauen
  size_t n_rows = data_tensor.shape()[0];
  size_t n_cols = col_indices.size();
  std::vector<std::vector<float>> submatrix(n_rows, std::vector<float>(n_cols));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < n_cols; ++c) {
      submatrix[r][c] = data_tensor(r, col_indices[c]);
    }
  }

  // Neues Dataset erstellen
  Dataset sliced;
  sliced.feature_names = features_to_extract; // Nur gewählte Features
  sliced.features = Tensor(submatrix);        // [606, 2]
  sliced.targets = input_data.targets;        // Targets unverändert
  return sliced;
}

void DataProcessor::save_data(const std::vector<Plaintext> &encoded_data,
                              const std::string &filename) {
  // std::ofstream file(filename, std::ios::binary);
  // if (!file)
  // {
  //     throw std::runtime_error("Konnte Datei nicht öffnen: " + filename);
  // }

  // // 1. Header schreiben
  // uint32_t magic = 0xFHEPLAIN; // Magic Number
  // uint32_t version = 1;
  // uint64_t num_ptxts = encoded_data.size();

  // file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  // file.write(reinterpret_cast<const char *>(&version), sizeof(version));
  // file.write(reinterpret_cast<const char *>(&num_ptxts), sizeof(num_ptxts));

  // // 2. Jedes Plaintext serialisieren
  // for (size_t i = 0; i < encoded_data.size(); ++i)
  // {
  //     std::string serialized;
  //     encoded_data[i]->Serialize(serialized); // OpenFHE Magic!

  //     uint64_t ptxt_size = serialized.size();
  //     file.write(reinterpret_cast<const char *>(&ptxt_size),
  //     sizeof(ptxt_size)); file.write(serialized.data(), ptxt_size);

  //     std::cout << "  PTXT " << (i + 1) << "/" << encoded_data.size()
  //               << ": " << ptxt_size / 1024 << " KB" << std::endl;
  // }

  // file.close();
  // std::cout << "✅ Gespeichert: " << encoded_data.size()
  //           << " Plaintexts → " << filename << std::endl;
}

void DataProcessor::save_data(const Tensor &data,
                              const std::vector<std::string> &feature_names,
                              const std::string &filename) {

  size_t n_rows = data.shape()[0];
  size_t n_cols = data.shape()[1];

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open CSV-file: " + filename);
  }

  // Column index
  for (size_t col = 0; col < n_cols; ++col) {
    file << feature_names[col];
    if (col < n_cols - 1)
      file << ",";
  }
  file << "\n";

  // Iterate through every row
  for (size_t row = 0; row < n_rows; ++row) {
    for (size_t col = 0; col < n_cols; ++col) {
      file << std::fixed << std::setprecision(6) << data(row, col);
      if (col < n_cols - 1)
        file << ",";
    }
    file << "\n";
  }

  file.close();
  std::cout << "CSV saved: " << filename << std::endl;
}

void DataProcessor::save_data(const std::vector<Ciphertext<DCRTPoly>> &ctxts,
                              const std::string &filename) {
  if (ctxts.empty()) {
    std::cout << "No ciphertexts to serialize." << std::endl;
    return;
  }

  Serial::SerializeToFile(filename, ctxts, SerType::JSON);
  std::cout << "Serialized " << ctxts.size()
            << " ciphertexts to JSON file: " << filename << std::endl;
}