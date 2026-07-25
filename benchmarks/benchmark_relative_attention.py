"""二维相对位置偏置注意力的 Triton 在线 softmax 原型。"""

import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def relative_attention_forward_kernel(
    query,
    key,
    value,
    bias_table,
    output,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_on: tl.constexpr,
    scale: tl.constexpr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, HEAD_DIM)

    query_ptrs = (
        query
        + batch * stride_qb
        + head * stride_qh
        + query_offsets[:, None] * stride_qn
        + dim_offsets[None, :]
    )
    query_values = tl.load(query_ptrs, mask=query_offsets[:, None] < TOKENS, other=0.0)

    running_max = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    log2e = 1.4426950408889634

    for key_start in range(0, TOKENS, BLOCK_N):
        current_keys = key_start + key_offsets
        key_ptrs = (
            key
            + batch * stride_kb
            + head * stride_kh
            + current_keys[:, None] * stride_kn
            + dim_offsets[None, :]
        )
        value_ptrs = (
            value
            + batch * stride_vb
            + head * stride_vh
            + current_keys[:, None] * stride_vn
            + dim_offsets[None, :]
        )
        key_values = tl.load(key_ptrs, mask=current_keys[:, None] < TOKENS, other=0.0)
        value_values = tl.load(value_ptrs, mask=current_keys[:, None] < TOKENS, other=0.0)

        scores = tl.dot(query_values, tl.trans(key_values)) * scale
        query_row = query_offsets // 19
        query_col = query_offsets % 19
        key_row = current_keys // 19
        key_col = current_keys % 19
        relative_index = (
            (query_row[:, None] - key_row[None, :] + 18) * 37
            + query_col[:, None]
            - key_col[None, :]
            + 18
        )
        valid = (query_offsets[:, None] < TOKENS) & (current_keys[None, :] < TOKENS)
        relative_bias = tl.load(
            bias_table + relative_index * HEADS + head,
            mask=valid,
            other=0.0,
        )
        scores += relative_bias
        scores = tl.where(valid, scores, -float('inf'))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp2((running_max - new_max) * log2e)
        probabilities = tl.exp2((scores - new_max[:, None]) * log2e)
        new_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(tl.float16), value_values)
        running_max = new_max
        running_sum = new_sum

    accumulator /= running_sum[:, None]
    output_ptrs = (
        output
        + batch * stride_ob
        + head * stride_oh
        + query_offsets[:, None] * stride_on
        + dim_offsets[None, :]
    )
    tl.store(output_ptrs, accumulator, mask=query_offsets[:, None] < TOKENS)


def triton_relative_attention(
    query,
    key,
    value,
    bias_table,
    block_m=64,
    block_n=64,
    num_warps=4,
    num_stages=3,
):
    batch, heads, tokens, head_dim = query.shape
    output = torch.empty_like(query)
    grid = (triton.cdiv(tokens, block_m), batch * heads)
    relative_attention_forward_kernel[grid](
        query,
        key,
        value,
        bias_table,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        head_dim ** -0.5,
        HEADS=heads,
        TOKENS=tokens,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def benchmark(callable_fn, rounds=100):
    for _ in range(10):
        callable_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        callable_fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=100)
    args = parser.parse_args()

    batch, heads, tokens, head_dim = args.batch_size, 8, 361, 32
    query = torch.randn(batch, heads, tokens, head_dim, device='cuda', dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    bias_table = torch.randn(37 * 37, heads, device='cuda', dtype=torch.float16) * 0.02

    positions = torch.arange(tokens, device='cuda')
    row, col = positions // 19, positions % 19
    relative_index = (
        (row[:, None] - row[None, :] + 18) * 37
        + col[:, None]
        - col[None, :]
        + 18
    )
    dense_bias = bias_table[relative_index].permute(2, 0, 1).unsqueeze(0).contiguous()

    reference = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=dense_bias,
        dropout_p=0.0,
        scale=head_dim ** -0.5,
    )
    candidate = triton_relative_attention(query, key, value, bias_table)
    max_error = float((candidate - reference).abs().max())
    mean_error = float((candidate - reference).abs().mean())

    torch_ms = benchmark(
        lambda: F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=dense_bias,
            dropout_p=0.0,
            scale=head_dim ** -0.5,
        ),
        args.rounds,
    )
    triton_ms = benchmark(
        lambda: triton_relative_attention(query, key, value, bias_table),
        args.rounds,
    )
    print(
        {
            'batch_size': batch,
            'torch_sdpa_ms': torch_ms,
            'triton_ms': triton_ms,
            'speedup': torch_ms / triton_ms,
            'max_error': max_error,
            'mean_error': mean_error,
        }
    )


if __name__ == '__main__':
    main()
