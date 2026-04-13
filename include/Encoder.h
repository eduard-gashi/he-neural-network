#include "network/Tensor.h"
#include "openfhe.h"

using namespace lbcrypto;

class Encoder {
private:
  CryptoContext<DCRTPoly> cc;
  KeyPair<DCRTPoly> keys;

  void setup_crypto_context(uint32_t batch_size, uint32_t mult_depth,
                            uint32_t scale_mod_size);

  void generate_keys();

public:
  Encoder(uint32_t mult_depth = 3, uint32_t scale_mod_size = 50,
          uint32_t batch_size = 64) {
    // 1. Setup CryptoContext
    setup_crypto_context(batch_size, mult_depth, scale_mod_size);

    // 2. Key Generation
    generate_keys();
  }

  // Plaintext-Encoding (for Training)
  std::vector<Plaintext> encode_tensor(const Tensor &tensor);

  // Ciphertext-Encryption (for Training)
  std::vector<Ciphertext<DCRTPoly>> encrypt(std::vector<Plaintext> &ptxts);

  // Encryption with user keys (for Inference)
  std::vector<Ciphertext<DCRTPoly>>
  encrypt_with_user_key(std::vector<Plaintext> &ptxts,
                        const PublicKey<DCRTPoly> &user_pk);

  // Decryption (only possible with secret_key)
  Tensor decrypt(const Ciphertext<DCRTPoly> &ctxt);
};
