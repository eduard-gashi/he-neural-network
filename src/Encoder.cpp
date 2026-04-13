#include "Encoder.h"
#include "network/Tensor.h"
#include "openfhe.h"

using namespace lbcrypto;

void Encoder::setup_crypto_context(uint32_t batch_size, uint32_t mult_depth,
                                   uint32_t scale_mod_size) {
  // Setup Parameters
  CCParams<CryptoContextCKKSRNS> parameters;
  parameters.SetMultiplicativeDepth(
      mult_depth); // Multiplicative Depth, e.g. for x1*x2*x3 = 2
  parameters.SetScalingModSize(
      scale_mod_size); // Bit-Length of the scaling factor D
  parameters.SetBatchSize(
      batch_size); // Number of plaintext slots used in the ciphertext, has to
                   // be < RingDimension/2

  // Create CryptoContext Object
  cc = GenCryptoContext(parameters);

  // Enable features
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);

  std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension()
            << std::endl
            << std::endl;
}

void Encoder::generate_keys() {
  // Generate Encryption Keys
  keys = cc->KeyGen();

  // Generate the digit size (Relinearization keys)
  cc->EvalMultKeyGen(keys.secretKey);

  // Generate the rotation keys
  cc->EvalRotateKeyGen(keys.secretKey, {1, -2});
}

std::vector<Plaintext> Encoder::encode_tensor(const Tensor &tensor) {
  std::vector<std::size_t> shape = tensor.shape(); // [rows, columns]
  std::cout << "\nEncoding tensor with shape " << shape << std::endl;
  std::size_t rows = shape[0];
  std::size_t features = shape[1];

  // Compute number of Plaintexts based on BatchSize and shape of the dataset
  std::size_t slots_per_ptxt = cc->GetEncodingParams()->GetBatchSize();
  std::size_t samples_per_ptxt = slots_per_ptxt / features;
  std::size_t num_ptxts = (rows + samples_per_ptxt - 1) / samples_per_ptxt;

  std::cout << "Crypto Parameters: " << cc->GetEncodingParams() << std::endl;

  std::vector<Plaintext> ptxts;
  ptxts.reserve(num_ptxts);

  std::cout << "Creating Batches for plaintext encoding:"
            << "\nData Points: " << rows << "\nBatch Size: " << slots_per_ptxt
            << "\nSamples per Plaintext: " << samples_per_ptxt
            << "\nAmount Plaintexts: " << num_ptxts << std::endl;

  // Create batches
  for (size_t ptxt_idx = 0; ptxt_idx < num_ptxts; ++ptxt_idx) {
    size_t start_sample = ptxt_idx * samples_per_ptxt;
    size_t end_sample = std::min(start_sample + samples_per_ptxt, rows);

    std::vector<double> ptxt_data;
    ptxt_data.reserve(slots_per_ptxt);

    // Samples [start:end] → flatten row-major
    for (size_t i = start_sample; i < end_sample; ++i) {
      Tensor row = tensor.row(i); // [2]
      for (float f : row.flatten()) {
        ptxt_data.push_back(static_cast<double>(f));
      }
    }

    ptxt_data.resize(slots_per_ptxt, 0.0); // Pad
    ptxts.emplace_back(cc->MakeCKKSPackedPlaintext(ptxt_data));
  }

  return ptxts;
}

std::vector<Ciphertext<DCRTPoly>>
Encoder::encrypt(std::vector<Plaintext> &ptxts) {
  std::size_t ptxts_len = size(ptxts);
  std::cout << "Encrypting " << ptxts_len << " plaintexts to ciphertexts."
            << std::endl;

  std::vector<Ciphertext<DCRTPoly>> ctxts;
  ctxts.reserve(ptxts_len);

  for (size_t ptxt_idx = 0; ptxt_idx < ptxts_len; ++ptxt_idx) {
    auto c = cc->Encrypt(keys.publicKey, ptxts[ptxt_idx]);
    ctxts.push_back(std::move(c));
  }
  return ctxts;
}

std::vector<Ciphertext<DCRTPoly>>
Encoder::encrypt_with_user_key(std::vector<Plaintext> &ptxts,
                               const PublicKey<DCRTPoly> &user_pk) {
  std::size_t ptxts_len = size(ptxts);
  std::cout << "Encrypting " << ptxts_len << " plaintexts to ciphertexts."
            << std::endl;

  std::vector<Ciphertext<DCRTPoly>> ctxts;
  ctxts.reserve(ptxts_len);

  return ctxts;
}

// Decryption (only possible with secret_key)
Tensor Encoder::decrypt(const Ciphertext<DCRTPoly> &ctxt) {
  Tensor a;
  return a;
}
