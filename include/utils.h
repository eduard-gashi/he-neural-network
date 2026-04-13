#include "network/Tensor.h"
#include "openfhe.h"
#include <vector>

size_t tensor_memory(const Tensor &t);

size_t plaintexts_memory(const std::vector<Plaintext> &ptxts);

size_t ciphertext_memory(const std::vector<Ciphertext<DCRTPoly>> &ctxts);

size_t ctxt_size(const Ciphertext<DCRTPoly> &ctxt);

size_t ciphertext_memory_ser(const std::vector<Ciphertext<DCRTPoly>> &ctxts);