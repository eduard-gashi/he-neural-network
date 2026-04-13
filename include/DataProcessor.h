#include "iostream"
#include "network/Tensor.h"
#include "openfhe.h"
#include <fstream>
#include <string>
#include <vector>

using namespace lbcrypto;

struct Dataset {
  std::vector<std::string> feature_names;
  Tensor features;
  Tensor targets;
};

class DataProcessor {
public:
  DataProcessor() = default;
  Dataset load_csv(const std::string &file_path);
  Tensor normalize_tensor(Tensor &tensor); // Z-standardization
  Dataset slice_data(const std::vector<std::string> &features_to_extract,
                     const Dataset &input_data);

  void save_data(const std::vector<Plaintext> &encoded_data,
                 const std::string &filename);

  void save_data(const Tensor &data,
                 const std::vector<std::string> &feature_names,
                 const std::string &filename);

  void save_data(const std::vector<Ciphertext<DCRTPoly>> &ctxts,
                 const std::string &filename);
};
