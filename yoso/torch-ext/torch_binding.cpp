#include <torch/all.h>
#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

at::Tensor lsh_cumulation_wrapper(
    at::Tensor query_mask,         // [batch_size, num_query]
    at::Tensor query_hash_code,    // [batch_size, num_query, num_hash_f]
    at::Tensor key_mask,           // [batch_size, num_key]
    at::Tensor key_hash_code,      // [batch_size, num_key, num_hash_f]
    at::Tensor value,              // [batch_size, num_key, value_dim]
    int64_t hashtable_capacity,
    bool use_cuda,
    int64_t version
) {
  return lsh_cumulation(
    query_mask,
    query_hash_code,
    key_mask,
    key_hash_code,
    value,
    static_cast<int>(hashtable_capacity),
    use_cuda,
    static_cast<int>(version)
  );
}

std::vector<at::Tensor> fast_hash_wrapper(
    at::Tensor query_mask,
    at::Tensor query_vector,
    at::Tensor key_mask,
    at::Tensor key_vector,
    int64_t num_hash_f,
    int64_t hash_code_len,
    bool use_cuda,
    int64_t version
) {
  return fast_hash(
    query_mask,
    query_vector,
    key_mask,
    key_vector,
    static_cast<int>(num_hash_f),
    static_cast<int>(hash_code_len),
    use_cuda,
    static_cast<int>(version)
  );
}

at::Tensor lsh_weighted_cumulation_wrapper(
    at::Tensor query_mask,         // [batch_size, num_query]
    at::Tensor query_hash_code,    // [batch_size, num_query, num_hash_f]
    at::Tensor query_weight,       // [batch_size, num_query, weight_dim]
    at::Tensor key_mask,           // [batch_size, num_key]
    at::Tensor key_hash_code,      // [batch_size, num_key, num_hash_f]
    at::Tensor key_weight,         // [batch_size, num_key, weight_dim]
    at::Tensor value,              // [batch_size, num_key, value_dim]
    int64_t hashtable_capacity,
    bool use_cuda,
    int64_t version
) {
  return lsh_weighted_cumulation(
    query_mask,
    query_hash_code,
    query_weight,
    key_mask,
    key_hash_code,
    key_weight,
    value,
    static_cast<int>(hashtable_capacity),
    use_cuda,
    static_cast<int>(version)
  );
}
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("lsh_cumulation(Tensor query_mask, Tensor query_hash_code, Tensor key_mask, Tensor key_hash_code, Tensor value, int hashtable_capacity, bool use_cuda, int version) -> Tensor");
  ops.impl("lsh_cumulation", torch::kCUDA, &lsh_cumulation_wrapper);

  ops.def("fast_hash(Tensor query_mask, Tensor query_vector, Tensor key_mask, Tensor key_vector, int num_hash_f, int hash_code_len, bool use_cuda, int version) -> Tensor[]");
  ops.impl("fast_hash", torch::kCUDA, &fast_hash_wrapper);

  ops.def("lsh_weighted_cumulation(Tensor query_mask, Tensor query_hash_code, Tensor query_weight, Tensor key_mask, Tensor key_hash_code, Tensor key_weight, Tensor value, int hashtable_capacity, bool use_cuda, int version) -> Tensor");
  ops.impl("lsh_weighted_cumulation", torch::kCUDA, &lsh_weighted_cumulation_wrapper);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)