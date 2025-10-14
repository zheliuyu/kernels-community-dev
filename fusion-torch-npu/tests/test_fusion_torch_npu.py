# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch_npu
from fusion_torch_npu import (
    RMSNorm,
    MLPWithSwiGLU,
    flash_attn_func,
    flash_attn_varlen_func,
)


def test_rmsnorm():
    device = torch.device("npu")
    dtype = torch.bfloat16
    x = torch.randn(1024, 1024, device=device, dtype=dtype)
    weight = torch.randn(1024, device=device, dtype=dtype)
    variance_epsilon = 1e-6

    rmsnorm_layer = RMSNorm()
    output = rmsnorm_layer(x)

    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    ref_out = weight * x.to(torch.bfloat16)
    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=1e-2)


def test_mlp_with_swiglu():
    device = torch.device("npu")
    dtype = torch.bfloat16
    batch_size = 4
    seq_length = 128
    hidden_size = 512
    intermediate_size = 2048

    hidden_state = torch.randn(
        batch_size, seq_length, hidden_size, device=device, dtype=dtype
    )

    mlp_layer = MLPWithSwiGLU()
    gate_proj = torch.nn.Linear(hidden_size, intermediate_size).to(device)
    up_proj = torch.nn.Linear(hidden_size, intermediate_size).to(device)
    down_proj = torch.nn.Linear(intermediate_size // 2, hidden_size).to(device)

    output = mlp_layer(hidden_state)

    gate_up = torch.cat((gate_proj(hidden_state), up_proj(hidden_state)), dim=-1)
    swish = gate_up[:, :, :intermediate_size] * torch.sigmoid(
        gate_up[:, :, :intermediate_size]
    )
    ref_out = down_proj(swish)

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=1e-2)


def test_flash_attention():
    device = torch.device("npu")
    dtype = torch.bfloat16
    batch_size = 2
    seq_length = 128
    num_heads = 8
    head_dim = 64

    query = torch.randn(
        batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    value = torch.randn(
        batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
    )

    output = flash_attn_func(query, key, value)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(attn_weights, value)

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=1e-2)


def test_flash_attention_varlen():
    device = torch.device("npu")
    dtype = torch.bfloat16
    batch_size = 2
    max_seq_length = 128
    num_heads = 8
    head_dim = 64

    query = torch.randn(
        batch_size, max_seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        batch_size, max_seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    value = torch.randn(
        batch_size, max_seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    cu_seqlens_q = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
    total_q_len = cu_seqlens_q[-1].item()
    total_k_len = cu_seqlens_k[-1].item()

    output = flash_attn_varlen_func(
        query[:total_q_len],
        key[:total_k_len],
        value[:total_k_len],
        cu_seqlens_q,
        cu_seqlens_k,
    )

    ref_out = torch.zeros_like(output)
    for i in range(batch_size):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()

        q_slice = query[q_start:q_end]
        k_slice = key[k_start:k_end]
        v_slice = value[k_start:k_end]

        scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        ref_out[q_start:q_end] = torch.matmul(attn_weights, v_slice)

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=1e-2)
