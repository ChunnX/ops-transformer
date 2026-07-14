"""临时脚本：调用 torch_extension chunk_gated_delta_rule 接口，支持 eager / graph 两种模式。

参考 attention/flash_attn/tests/pytests/core/backends/npu.py 的封装风格。
"""

import argparse
import torch
import torch_npu

try:
    from cann_ops_transformer.ops import chunk_gated_delta_rule
    _HAS_OP = True
except ImportError:
    _HAS_OP = False

try:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    _HAS_TORCHAIR = True
except ImportError:
    _HAS_TORCHAIR = False


class ChunkGatedDeltaRuleGraphNetwork(torch.nn.Module):
    """torch.compile 图模式包装器。"""

    def forward(self, q, k, v, beta, initial_state, actual_seq_lengths, g, scale):
        out, final_state = chunk_gated_delta_rule(
            q, k, v, beta, initial_state, actual_seq_lengths,
            g=g, scale=scale,
        )
        return out, final_state


def build_inputs(B, seqlen, nk, nv, dk, dv, device):
    """构造随机输入，shape / dtype 对齐算子约束。"""
    T = B * seqlen
    q = torch.randn(T, nk, dk, dtype=torch.bfloat16, device=device)
    k = torch.randn(T, nk, dk, dtype=torch.bfloat16, device=device)
    v = torch.randn(T, nv, dv, dtype=torch.bfloat16, device=device)
    beta = torch.rand(T, nv, dtype=torch.bfloat16, device=device)
    initial_state = torch.randn(B, nv, dv, dk, dtype=torch.bfloat16, device=device)
    actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device=device)
    g = torch.rand(T, nv, dtype=torch.float32, device=device) * -1.0
    scale = 1.0 / (dk ** 0.5)
    return q, k, v, beta, initial_state, actual_seq_lengths, g, scale


def get_graph_net():
    """构建并缓存图模式网络，CompilerConfig 配置参考 npu.py。"""
    torch._dynamo.reset()
    net = ChunkGatedDeltaRuleGraphNetwork().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(net, fullgraph=False, backend=npu_backend, dynamic=False)


def run_eager(inputs):
    q, k, v, beta, initial_state, actual_seq_lengths, g, scale = inputs
    out, final_state = chunk_gated_delta_rule(
        q, k, v, beta, initial_state, actual_seq_lengths,
        g=g, scale=scale,
    )
    torch_npu.npu.synchronize()
    return out, final_state


def run_graph(inputs):
    if not _HAS_TORCHAIR:
        raise ImportError("graph 模式需要 torchair，请先安装")
    net = get_graph_net()
    q, k, v, beta, initial_state, actual_seq_lengths, g, scale = inputs
    out, final_state = net(q, k, v, beta, initial_state, actual_seq_lengths, g, scale)
    torch_npu.npu.synchronize()
    return out, final_state


def main():
    parser = argparse.ArgumentParser(description="chunk_gated_delta_rule 临时调用脚本")
    parser.add_argument("--mode", choices=["eager", "graph"], default="eager",
                        help="调用模式：eager 单算子 / graph 图模式")
    parser.add_argument("--device", type=int, default=0, help="NPU 设备号")
    parser.add_argument("--B", type=int, default=2, help="batch size")
    parser.add_argument("--seqlen", type=int, default=200, help="单 batch 序列长度")
    parser.add_argument("--nk", type=int, default=4, help="query/key 头数 Nk")
    parser.add_argument("--nv", type=int, default=8, help="value 头数 Nv")
    parser.add_argument("--dk", type=int, default=32, help="key 维度 Dk")
    parser.add_argument("--dv", type=int, default=32, help="value 维度 Dv")
    args = parser.parse_args()

    if not _HAS_OP:
        raise ImportError("无法导入 cann_ops_transformer.ops.chunk_gated_delta_rule")

    torch_npu.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    if args.nv % args.nk != 0:
        raise ValueError(f"Nv={args.nv} 必须是 Nk={args.nk} 的整数倍")

    print(f"[配置] mode={args.mode} device={device} "
          f"B={args.B} seqlen={args.seqlen} Nk={args.nk} Nv={args.nv} "
          f"Dk={args.dk} Dv={args.dv}")

    inputs = build_inputs(args.B, args.seqlen, args.nk, args.nv,
                          args.dk, args.dv, device)

    if args.mode == "graph":
        out, final_state = run_graph(inputs)
    else:
        out, final_state = run_eager(inputs)

    T = args.B * args.seqlen
    assert out.shape == (T, args.nv, args.dv), f"out shape 不符: {out.shape}"
    assert final_state.shape == (args.B, args.nv, args.dv, args.dk), \
        f"final_state shape 不符: {final_state.shape}"

    print(f"[结果] out.shape={tuple(out.shape)} "
          f"dtype={out.dtype}")
    print(f"[结果] final_state.shape={tuple(final_state.shape)} "
          f"dtype={final_state.dtype}")
    print(f"[结果] out 含 nan: {torch.isnan(out).any().item()}  "
          f"含 inf: {torch.isinf(out).any().item()}")
    print("[完成] chunk_gated_delta_rule 调用成功")


if __name__ == "__main__":
    main()
