#include <torch/torch.h>
#include <ATen/ATen.h>
#include <vector>

at::Tensor lsh_cumulation(
    at::Tensor query_mask,         // [batch_size, num_query]
    at::Tensor query_hash_code,    // [batch_size, num_query, num_hash_f]
    at::Tensor key_mask,           // [batch_size, num_key]
    at::Tensor key_hash_code,      // [batch_size, num_key, num_hash_f]
    at::Tensor value,              // [batch_size, num_key, value_dim]
    int hashtable_capacity,
    bool use_cuda,
    int version
);

std::vector<at::Tensor> fast_hash(
    at::Tensor query_mask,
    at::Tensor query_vector,
    at::Tensor key_mask,
    at::Tensor key_vector,
    int num_hash_f,
    int hash_code_len,
    bool use_cuda,
    int version
);

at::Tensor lsh_weighted_cumulation(
    at::Tensor query_mask,         // [batch_size, num_query]
    at::Tensor query_hash_code,    // [batch_size, num_query, num_hash_f]
    at::Tensor query_weight,       // [batch_size, num_query, weight_dim]
    at::Tensor key_mask,           // [batch_size, num_key]
    at::Tensor key_hash_code,      // [batch_size, num_key, num_hash_f]
    at::Tensor key_weight,         // [batch_size, num_key, weight_dim]
    at::Tensor value,              // [batch_size, num_key, value_dim]
    int hashtable_capacity,
    bool use_cuda,
    int version
);