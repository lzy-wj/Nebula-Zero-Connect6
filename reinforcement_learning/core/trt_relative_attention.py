"""Connect6 专用的二维相对位置注意力 TensorRT AOT 插件。

这个内核针对当前模型的固定几何尺寸设计：19×19 个 token、8 个头、
每头 32 维。它在在线 softmax 循环中直接计算二维相对位置索引，因此
既不创建 361×361×8 的稠密偏置，也不保存完整注意力概率矩阵。
"""

from typing import Tuple, Union

import tensorrt as trt
import tensorrt.plugin as trtp
import triton
import triton.language as tl


PLUGIN_ID = "nebula::relative_attention"
BOARD_WIDTH = 19
HEADS = 8
TOKENS = BOARD_WIDTH * BOARD_WIDTH
HEAD_DIM = 32
RELATIVE_WIDTH = BOARD_WIDTH * 2 - 1
BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def relative_attention_forward_kernel(
    query,
    key,
    value,
    bias_table,
    output,
    SCALE: tl.constexpr,
    HEADS_CONST: tl.constexpr,
    TOKENS_CONST: tl.constexpr,
    HEAD_DIM_CONST: tl.constexpr,
    BOARD_WIDTH_CONST: tl.constexpr,
    RELATIVE_WIDTH_CONST: tl.constexpr,
    BLOCK_M_CONST: tl.constexpr,
    BLOCK_N_CONST: tl.constexpr,
):
    """分块在线 softmax；一个 Triton program 处理一个头的 64 行。"""

    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS_CONST
    head = batch_head % HEADS_CONST

    query_offsets = query_block * BLOCK_M_CONST + tl.arange(0, BLOCK_M_CONST)
    key_offsets = tl.arange(0, BLOCK_N_CONST)
    dim_offsets = tl.arange(0, HEAD_DIM_CONST)

    head_stride = TOKENS_CONST * HEAD_DIM_CONST
    batch_stride = HEADS_CONST * head_stride
    query_ptrs = (
        query
        + batch * batch_stride
        + head * head_stride
        + query_offsets[:, None] * HEAD_DIM_CONST
        + dim_offsets[None, :]
    )
    query_values = tl.load(
        query_ptrs,
        mask=query_offsets[:, None] < TOKENS_CONST,
        other=0.0,
    )

    running_max = tl.full((BLOCK_M_CONST,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M_CONST,), tl.float32)
    accumulator = tl.zeros((BLOCK_M_CONST, HEAD_DIM_CONST), tl.float32)
    log2e = 1.4426950408889634

    for key_start in range(0, TOKENS_CONST, BLOCK_N_CONST):
        current_keys = key_start + key_offsets
        key_ptrs = (
            key
            + batch * batch_stride
            + head * head_stride
            + current_keys[:, None] * HEAD_DIM_CONST
            + dim_offsets[None, :]
        )
        value_ptrs = (
            value
            + batch * batch_stride
            + head * head_stride
            + current_keys[:, None] * HEAD_DIM_CONST
            + dim_offsets[None, :]
        )
        key_values = tl.load(
            key_ptrs,
            mask=current_keys[:, None] < TOKENS_CONST,
            other=0.0,
        )
        value_values = tl.load(
            value_ptrs,
            mask=current_keys[:, None] < TOKENS_CONST,
            other=0.0,
        )

        scores = tl.dot(query_values, tl.trans(key_values)) * SCALE

        # 相对位置索引与 PyTorch 模型完全一致：先行差，再列差。
        query_row = query_offsets // BOARD_WIDTH_CONST
        query_col = query_offsets % BOARD_WIDTH_CONST
        key_row = current_keys // BOARD_WIDTH_CONST
        key_col = current_keys % BOARD_WIDTH_CONST
        relative_index = (
            (query_row[:, None] - key_row[None, :] + BOARD_WIDTH_CONST - 1)
            * RELATIVE_WIDTH_CONST
            + query_col[:, None]
            - key_col[None, :]
            + BOARD_WIDTH_CONST
            - 1
        )
        valid = (query_offsets[:, None] < TOKENS_CONST) & (
            current_keys[None, :] < TOKENS_CONST
        )
        relative_bias = tl.load(
            bias_table + relative_index * HEADS_CONST + head,
            mask=valid,
            other=0.0,
        )
        scores = tl.where(valid, scores + relative_bias, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp2((running_max - new_max) * log2e)
        probabilities = tl.exp2((scores - new_max[:, None]) * log2e)
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(tl.float16), value_values)
        running_max = new_max

    accumulator /= running_sum[:, None]
    output_ptrs = (
        output
        + batch * batch_stride
        + head * head_stride
        + query_offsets[:, None] * HEAD_DIM_CONST
        + dim_offsets[None, :]
    )
    tl.store(
        output_ptrs,
        accumulator,
        mask=query_offsets[:, None] < TOKENS_CONST,
    )


def _kernel_source():
    """生成固定形状的 Triton AOT 编译描述。"""

    return triton.compiler.ASTSource(
        fn=relative_attention_forward_kernel,
        signature={
            "query": "*fp16",
            "key": "*fp16",
            "value": "*fp16",
            "bias_table": "*fp16",
            "output": "*fp16",
        },
        constexprs={
            "SCALE": HEAD_DIM**-0.5,
            "HEADS_CONST": HEADS,
            "TOKENS_CONST": TOKENS,
            "HEAD_DIM_CONST": HEAD_DIM,
            "BOARD_WIDTH_CONST": BOARD_WIDTH,
            "RELATIVE_WIDTH_CONST": RELATIVE_WIDTH,
            "BLOCK_M_CONST": BLOCK_M,
            "BLOCK_N_CONST": BLOCK_N,
        },
    )


@trtp.register(PLUGIN_ID)
def relative_attention_desc(
    query: trtp.TensorDesc,
    key: trtp.TensorDesc,
    value: trtp.TensorDesc,
    bias_table: trtp.TensorDesc,
) -> trtp.TensorDesc:
    """输出布局与 query 完全相同。输入形状由模型和构建期检查保证。"""

    return query.like()


@trtp.aot_impl(PLUGIN_ID)
def relative_attention_aot_impl(
    query: trtp.TensorDesc,
    key: trtp.TensorDesc,
    value: trtp.TensorDesc,
    bias_table: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    """构建引擎时编译 PTX，运行引擎时不再依赖 Python 或 Triton。"""

    for tensor in (query, key, value, bias_table, outputs[0]):
        if tensor.dtype != trt.float16:
            raise ValueError("Nebula 相对位置注意力 AOT 插件目前只支持 FP16")

    compiled = triton.compile(
        _kernel_source(),
        options={"num_warps": NUM_WARPS, "num_stages": NUM_STAGES},
    )
    launch_params = trtp.KernelLaunchParams(
        grid_x=(TOKENS + BLOCK_M - 1) // BLOCK_M,
        grid_y=query.shape_expr[0] * HEADS,
        block_x=compiled.metadata.num_warps * 32,
        shared_mem=compiled.metadata.shared,
    )

    # 内核没有动态标量参数，动态 batch 只参与 launch grid 的计算。
    extra_args = trtp.SymIntExprs(0)
    return compiled.metadata.name, compiled.asm["ptx"], launch_params, extra_args


def add_relative_attention_plugin(network, query, key, value, bias_table):
    """向 TensorRT 网络添加 AOT 插件层并返回输出张量。"""

    create_plugin = trtp.op.nebula.relative_attention(
        query,
        key,
        value,
        bias_table,
    )
    inputs, shape_inputs, plugin = create_plugin(trt.QuickPluginCreationRequest.STRICT_AOT)
    layer = network.add_plugin_v3(inputs, shape_inputs, plugin)
    if layer is None:
        raise RuntimeError("TensorRT 无法创建 Nebula 相对位置注意力插件层")
    return layer.get_output(0)
